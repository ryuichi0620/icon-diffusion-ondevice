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

4 ステップなら `[999, 759, 499, 259]` になる。これは diffusers と一致する値なので、
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

当初の見積もりと、2026-08-22 に実際に `.aimodel` へ変換して測った値。

| モデル | パラメータ | fp16（見積） | INT8（見積） | **fp32（実測）** | **INT8（実測）** |
| --- | --- | --- | --- | --- | --- |
| UNet | ~860M | ~1.7GB | ~860MB | 3.2GB | **829MB** |
| TextEncoder | ~123M | ~250MB | ~125MB | 470MB | **118MB** |
| VAEDecoder | ~50M | ~100MB | ~50MB | 189MB | **47MB** |
| 合計 | | | ~1.0GB | 3.9GB | **994MB** |

**見積もりはほぼ当たっていた。** INT8 で合計 1GB を切り、2GB の上限に収まる。
**INT8 量子化はオプションではなく前提**。
TextEncoder は生成の頭で 1 回使ったら解放できるので、UNet ループ中に常駐させない。

### 量子化したのは重みだけ

`05_export_coreai.py` の `_quantize_module()` は **重みのみ int8 にし、activation には触らない**。
activation を量子化するには代表的な入力を流して値の範囲を観測する校正（PTQ）が必要で、
形だけ合わせた乱数で代用すると活性値の範囲を誤って学習し、絵の品質が黙って落ちる。
サイズの大半は重みなので、校正なしで安全に効く範囲に留めてある。

### 変換したモデルの検証結果（2026-08-23）

Core AI には **Python のランタイム**（`coreai.runtime`）がある。Xcode を用意しなくても
Mac 上で `.aimodel` を読んで推論でき、変換の正しさをここで確かめられる。

同じ入力を PyTorch と Core AI に与えて出力を比べた（モジュール単位）。

| モジュール | fp32 の相対 MAE | int8 の相対 MAE | Mac での 1 回 |
| --- | --- | --- | --- |
| TextEncoder | 0.0005% | **8.98%** | 1s |
| UNet | 0.0002% | 0.34% | 27〜29s |
| VAEDecoder | 1.90% | 1.96% | 108〜114s |

- **変換そのものは正しい。** UNet と TextEncoder の fp32 は実質完全一致
- **量子化の damage は TextEncoder に集中する。** CLIP のテキストエンコーダは埋め込みに
  外れ値を持つため、重みのみ int8 でも崩れやすい。UNet と VAEDecoder はほぼ無傷
- **遅さのボトルネックは VAEDecoder。** 17 倍大きい UNet より遅い（512×512 への引き伸ばし）

4 ステップの生成を通しで回して PyTorch と見比べた結果:

| 版 | 画素 MAE |
| --- | --- |
| Core AI 全 int8 | 0.0718 |
| Core AI（TextEncoder だけ fp32） | 0.0643 |

**絵は壊れない。** 細部（蓋の描画など）は変わるが、同じ被写体・同じ配色・同じ構図の
アイコンとして成立する。TextEncoder を fp32 にすると誤差が 1 割減るが同一にはならず、
+352MB かかる。**全 int8（994MB）で十分**と判断した。

**未検証**: Mac のランタイムは PyTorch CPU より 30 倍遅い（fp32 VAEDecoder で 114s 対 3.3s）。
計算ユニットの指定も `is_supported()` が false を返す。**速度は実機でしか測れない。**
また歩留まり 70% が Core AI でも保たれるかは未確認（1 枚 4 分かかるので 192 枚の評価は
この Mac では現実的でない）。

## 5.5 モデルの入出力契約

Swift 側が入出力を名前で引くための契約は **`pipeline_config.json` の `io` が唯一の情報源**。
Swift にハードコードしない。

| モデル | 入力 | 出力 |
| --- | --- | --- |
| TextEncoder | `input_ids` [1,77] int32 | `text_embeddings` [1,77,768] f32 |
| UNet | `sample` [1,4,64,64] f32<br>`timestep` [1] **f32**<br>`encoder_hidden_states` [1,77,768] f32 | `noise_pred` [1,4,64,64] f32 |
| VAEDecoder | `latent` [1,4,64,64] f32 | `image` [1,3,512,512] f32（値域 -1〜1） |

**`timestep` は float32。** 整数で渡すとランタイムが弾く。

**出力名は明示しないと自動生成名になる。** 何も指定しないと `convolution_97` のような
グラフ内部の名前が付き、モデルが少し変わるだけで名前が変わる。Stage 5 は
`add_exported_program(..., input_names=..., output_names=...)` で固定している。

## 6.5 タイムステップの取り違え（2026-08-23 に修正）

**Swift 側だけが古い diffusers の挙動を実装していた。**

| | 4 ステップのタイムステップ |
| --- | --- |
| 旧実装（Swift・企画メモ） | `[999, 759, 519, 279]` |
| **現行の diffusers** | **`[999, 759, 499, 259]`** |

diffusers は途中でタイムステップの選び方を変えている。旧来は
`lcm_origin_timesteps[::-skipping_step]`（skipping = 50/4 = 12）だったが、
現行は**降順に並べた候補から等間隔インデックスを取る**
（`np.linspace(0, 50, 4, endpoint=False)` を floor して `[0, 12, 25, 37]`）。

**なぜ気づけなかったか。** 3 つ重なっていた。

1. `05_export_coreai.py` がタイムステップを**ハードコード**していた（スケジューラから導出していなかった）
2. `smoke_test.py` の検査が、その定数と `pipeline_config.json` に書かれた同じ定数を
   比べるだけの**空虚な比較**になっていた。だから常に通った
3. Swift のテスト期待値（`expected_scheduler.json`）に**再生成手段が無かった**。
   手作業で作られていたので、diffusers の変更に追随できなかった

**直した内容。** タイムステップは実スケジューラから導出する。smoke_test は diffusers に
実際に計算させた値と突き合わせる。期待値は `training/scripts/gen_scheduler_fixture.py`
で再生成できるようにし、`--check` でずれを検出できるようにした。

## 6.6 Core AI の Swift API（2026-08-23 に実物で検証）

**Core AI は iOS 27 / macOS 27 以降。** iOS 26 SDK に `CoreAI.framework` は無い。
ビルドには Xcode 27 以降が要る。`.aimodel` も `OSVersion.v27` 向けに書き出される。

iOS 27.0 SDK に対して型チェックを通して確認した実際の呼び出し方。

```swift
let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)  // 省略可
let model = try await AIModel(contentsOf: url, options: options)
guard let fn = try model.loadFunction(named: "main") else { … }

let input = NDArray(scalars: values, shape: shape)
var outputs = try await fn.run(inputs: ["latent": input])

guard let value = outputs.remove("image"), let nd = value.ndArray else { … }
let scalars = nd.view(as: Float.self).withUnsafePointer { p, _, _ in
    Array(UnsafeBufferPointer(start: p, count: nd.shape.reduce(1, *)))
}
```

### 当初の推測とどう違ったか

| 項目 | 企画メモの推測 | 実物 |
| --- | --- | --- |
| 読み込み | `AIModel(contentsOf:)` | ✅ 正しい（`options:` 既定付き） |
| 関数の取得 | （無し） | **`model.loadFunction(named: "main")`** が要る |
| 推論 | `model.prediction(from:)` | **`fn.run(inputs: [String: NDArray])`** |
| 入力生成 | `NDArray(shape:values:)` | **`NDArray(scalars:shape:)`** |
| 出力取得 | `outputs.values.first` | **`outputs.remove(name) -> InferenceValue?`** |
| 値の読み出し | `array.scalars(of:)` | **`nd.view(as:).withUnsafePointer`** |

ほかに使えるもの:

- **`AIModel.specialize(contentsOf:options:cache:cachePolicy:)`** — コンパイル結果を
  キャッシュする。端末で毎回コンパイルさせないために要る
- **`ComputeUnitKind.availableKinds`** — 実行できる計算ユニットの確認

### 検証の限界

**通ったのは型チェックまで。実機では動かしていない。**
`./ios/typecheck-coreai.sh` で再現できる。`swift test` では `CoreAIBackend.swift` は
ビルド対象外なので、これを別に通さないと壊れても気づけない。

## 7. 非対応端末

Core AI は Apple Intelligence 対応機が前提。`IconDiffusionPipeline` の初期化を失敗させて
`IconDiffusionError.unsupportedDevice` を返し、アプリ側で分岐する想定。
判定ロジックはアプリの要件が決まってから足す（現状は backend の初期化失敗をそのまま伝える）。
