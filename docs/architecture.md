# 設計メモ

企画（[plan.md](plan.md)）を実装に落とすときに決めたこと。判断が変わったらこちらを更新する。

## 1. なぜモデルを 3 つに割るのか

拡散パイプラインは推論の形が 3 つで違う。

| モデル | 実行回数 | 入力 | 出力 |
| --- | --- | --- | --- |
| TextEncoder (CLIP ViT-L/14) | 生成ごとに 1 回 | `input_ids [1, 77] int32` | `last_hidden_state [1, 77, 768]` |
| UNet | 生成ごとに **ステップ数だけ** | `sample [B, 4, 64, 64]`, `timestep`, `encoder_hidden_states [B, 77, 768]` | `noise_pred [B, 4, 64, 64]` |
| VAEDecoder | 生成ごとに 1 回 | `latent [1, 4, 64, 64]` | `image [1, 3, 512, 512]`（値域 `[-1, 1]`） |

1 つにまとめると「毎ステップ回る部分」だけを最適化できない。UNet だけが 4 回走るので、
量子化もプロファイルもここに集中させたい。`apple/ml-stable-diffusion` が旧 Core ML でこの
割り方をしていたのと同じ理由。

VAE の潜在は `scaling_factor = 0.18215` でスケールされている。デコード前に割り戻すこと。

## 2. CFG を使うか — オンデバイスで一番効く判断

classifier-free guidance は「条件つき」と「無条件」の 2 回 UNet を回す。つまり **4 ステップの
生成が UNet 8 回になる**。

| 設定 | UNet 実行回数 | 備考 |
| --- | --- | --- |
| `guidanceScale = 1.0` | 4 | CFG なし。LCM-LoRA は低 CFG 前提で蒸留されているので破綻しにくい |
| `guidanceScale = 1.2`（企画メモの値） | 8 | Mac での質感確認向け。品質はわずかに上がる |

企画メモの Stage 3 スニペットは `guidance_scale=1.2` なので、diffusers 側では CFG が
**有効**（`do_classifier_free_guidance = guidance_scale > 1.0`）。Mac での確認はそれでよいが、
**端末では 1.0（CFG なし）を既定にする**。速度が素直に 2 倍違う。

`IconDiffusionPipeline` は両方に対応し、`guidanceScale <= 1.0` のときはバッチを 1 のまま
1 回だけ UNet を回す。

## 3. LCM スケジューラを Swift で持つ

Core AI が動かすのは UNet の 1 回分だけ。「ノイズ予測 → 少し綺麗にする」の後半は自分で書く。
diffusers の `LCMScheduler` と同じ式を [`LCMScheduler.swift`](../ios/IconDiffusionKit/Sources/IconDiffusionKit/LCMScheduler.swift) に置いた。

**タイムステップの選び方**（`original_inference_steps = 50` が LCM の学習時設定）

```
k                  = 1000 / 50 = 20
lcm_origin         = [19, 39, 59, ..., 999]        (50 個)
skipping_step      = 50 / num_inference_steps
timesteps          = lcm_origin[::-skipping_step][:num_inference_steps]
```

4 ステップなら `[999, 759, 519, 279]` になる。これは diffusers と一致する既知の値なので、
テストで固定値として突き合わせている。

**1 ステップの計算**

```
c_skip = σ² / ((t·10)² + σ²)                       σ = 0.5 (sigma_data)
c_out  = (t·10) / sqrt((t·10)² + σ²)

x0_pred  = (x_t - sqrt(1 - ᾱ_t) · ε_pred) / sqrt(ᾱ_t)
denoised = c_out · x0_pred + c_skip · x_t

最終ステップ以外:  x_{t-1} = sqrt(ᾱ_prev) · denoised + sqrt(1 - ᾱ_prev) · z,  z ~ N(0, I)
最終ステップ:      x_{t-1} = denoised
```

`ᾱ` は SD1.5 の scaled-linear β スケジュール（`β_start=0.00085`, `β_end=0.012`, 1000 ステップ）
から作る。`init_noise_sigma = 1.0` なので初期潜在のスケーリングは不要。

**中間ステップにノイズを足し直すのが LCM の肝**なので、ここを省くと 4 ステップでも眠い絵になる。

## 4. バックエンド抽象 — Core AI の API 不確実性を 1 ファイルに閉じ込める

Core AI（WWDC26）の Swift API は企画メモに書かれた形しか手元に情報がない。実装がずれる前提で、
依存を 1 枚のプロトコルの裏に隔離した。

```
IconDiffusionPipeline          ← 純 Swift。ループ・CFG・潜在の扱い
  ├─ CLIPTokenizer             ← 純 Swift。テストあり
  ├─ LCMScheduler              ← 純 Swift。テストあり
  └─ DiffusionBackend (protocol)
       ├─ CoreAIBackend        ← ★ Core AI 依存はここだけ。TODO(core-ai) 付き
       └─ StubBackend          ← テスト用。恒等・ゼロ出力で形だけ返す
```

こうしておくと:

- API 名が違っても直すのは `CoreAIBackend.swift` だけ
- Core AI が無い環境（Linux CI、非対応端末）でもスケジューラとトークナイザは検証できる
- 非対応端末向けの逃げ道（サーバー呼び出し・プリセット画像）を別 backend として足せる

`CoreAIBackend.swift` は `#if canImport(CoreAI)` で囲ってある。フレームワークが無い環境では
コンパイル対象から外れ、`swift test` はスケジューラ／トークナイザのテストだけを走らせる。

## 5. トークナイザ

CLIP の byte-level BPE をそのまま Swift で実装した（`CLIPTokenizer.swift`）。

- HF の `tokenizer/vocab.json`（token → id）と `tokenizer/merges.txt`（マージ規則）を読む
- `bytes_to_unicode()` によるバイト → 疑似文字マッピングを再現
- 単語末尾に `</w>` を付ける CLIP 方式
- 前処理は小文字化 + 連続空白の畳み込み
- `<|startoftext|>` = 49406, `<|endoftext|>` = 49407、系列長 77 に EOS でパディング

ftfy（Python 側の Unicode 修復）までは移植していない。生成プロンプトは自分のアプリが作るので、
壊れた Unicode が入ってくる経路が無いという判断。ここは割り切り。

`vocab.json` / `merges.txt` は Stage 5 の `05_export_coreai.py` がベースモデルから書き出す。
サイズが大きいので Git には入れず、`.gitignore` している。

## 6. メモリの当たり（SD1.5）

| モデル | パラメータ | fp16 | INT8 |
| --- | --- | --- | --- |
| UNet | ~860M | ~1.7GB | ~860MB |
| TextEncoder | ~123M | ~250MB | ~125MB |
| VAEDecoder | ~50M | ~100MB | ~50MB |

fp16 のまま 3 つ載せると 2GB を超える。**INT8 量子化はオプションではなく前提**。
TextEncoder は生成の頭で 1 回使ったら解放できるので、UNet ループ中に常駐させない。

## 7. 非対応端末

Core AI は Apple Intelligence 対応機が前提。`IconDiffusionPipeline` の初期化を失敗させて
`IconDiffusionError.unsupportedDevice` を返し、アプリ側で分岐する想定。
判定ロジックはアプリの要件が決まってから足す（現状は backend の初期化失敗をそのまま伝える）。
