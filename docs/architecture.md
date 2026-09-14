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

### ANE は fp32 を扱えない（2026-08-24 に実機で踏んだ）

<!-- ここを間違えて実機がクラッシュした。二度と踏まないこと -->

**書き出す dtype は fp16 にする。** fp32 のまま端末に載せると起動時に落ちる。

```
ane_validation_message("Incompatible element type for ANE:
  expected fp16, f8E4M3, si8, ui8, si16, or ui16")
failed: ANE I/O op isn't a squeeze / expandDims
...
libc++abi: terminating due to uncaught exception of type std::bad_alloc
```

Attention の各所で ANE 検証に落ちて GPU/CPU にフォールバックし、UNet の
コンパイル中にメモリを使い切って死ぬ。

**INT8 量子化しても足りない。** 量子化しているのは**重みだけ**で、入出力と活性値は
fp32 のまま残る。ANE が見るのはそちらなので、別途 fp16 にする必要がある。

| 版 | サイズ（VAEDecoder） | 入出力 dtype | ANE |
| --- | --- | --- | --- |
| fp32 | 198MB | float32 | ✗ |
| int8（重みのみ） | 47MB | float32 | ✗ |
| fp16 | 99MB | float16 | ✓ |
| **fp16 + int8** | **50MB** | **float16** | **✓** |

`05_export_coreai.py --precision fp16`（既定）で fp16 になる。
**トレースは fp32 のまま**行い、書き出す直前に落とす。ここを混同しないこと。
トレースを fp16 でやると数値が荒れるが、配布 dtype とは別の判断。

**品質は落ちない。** 4 ステップの生成を PyTorch と比べると、画素 MAE は
0.0718（int8/fp32 IO）→ 0.0587（fp16+int8）と**むしろ改善した**。

**Swift 側も合わせること。** `NDArray` を `[Float]` で作ると fp32 になる。
`CoreAIBackend` は `pipeline_config.json` の `io` に書かれた dtype を見て
`Float16` と `Float` を切り替える。ここをハードコードしない。

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

- **`ComputeUnitKind.availableKinds`** — 実行できる計算ユニットの確認

### コンパイル結果のキャッシュ

`.aimodel` は端末向けにコンパイルしてから動く。**そのコンパイルが重い。**
Mac の Python 側で VAEDecoder 1 回に 100 秒かかったことを思うと、端末でも無視できない。

`CoreAIBackend` はまずキャッシュを見て、無ければ `specialize` して保存する。

```swift
if let cached = try? AIModelCache.default.model(for: url, options: options) {
    return cached
}
return try await AIModel.specialize(
    contentsOf: url, options: options, cache: .default, cachePolicy: .default
)
```

破棄条件は既定のまま。**ストレージ逼迫時**と、**元の `.aimodel` が変わった / 消えたとき**に
破棄される。後者があるので、モデルを作り直したときに古いコンパイル結果を掴む事故は起きない。

`CoreAIBackend.prewarm(resources:preferredComputeUnit:onProgress:)` で 3 モデルを
まとめて先に済ませられる。**生成ボタンを押してから待たせないため**にあり、
アプリは画面が出た時点で呼ぶ。2 回目以降はキャッシュから読むだけになる。

### 検証の限界

型チェックは `./ios/typecheck-coreai.sh` で再現できる。`swift test` では
`CoreAIBackend.swift` はビルド対象外なので、これを別に通さないと壊れても気づけない。

**2026-08-24 に実機（iPhone・iOS 27）で通しで動作を確認した。** 到達点は 6.7 節。

## 6.7 実機で絵が壊れた 2 つの原因（2026-08-24）

変換も型チェックも通り、4 ステップも回り切るのに、出てくる絵が灰色の帯だった。
原因は 2 つあり、**どちらも「変換は成功しているのに値が化ける」**という形で出た。
形の検証（`requireShape`）では検出できない種類の壊れ方なので、各段の値の統計を
出して Python 側と突き合わせて特定した（`ICON_DEBUG=1`）。

### 原因 1: int8 量子化が CLIP の TextEncoder を壊す

    TextEncoder 出力    fp32:  [-28.05, 33.09]
                        int8:  [-14.20,  6.77]   ← 振幅が半分以下に潰れる

CLIP の埋め込みは一部の次元に極端な外れ値を持つ。per-channel 対称 int8 でも
その範囲を表現しきれず、全体が潰れる。潰れた埋め込みは UNet の 4 ステップすべてに
効くので、最終的に VAE の出力が [-1, 1] を大きく外れて飽和する。

**対処: TextEncoder だけ量子化から外す**（`configs/icon_lora.toml` の
`export.quantize_skip`）。118MB → 235MB と増えるが、UNet（821MB）に比べれば小さい。
UNet と VAEDecoder は int8 のままで問題ない（相対誤差 0.34% / 1.45%）。

### 原因 2: VAEDecoder は ANE 上でだけ壊れる

    decodeLatent 出力   ANE:  [-13.27, 26.59]   ← 発散
                        GPU:  [ -0.85,  1.13]
                        Python 参照:  [-0.78, 1.09]

**モデルは正しい。** 実機に載せているのと同じ `.aimodel` を Mac の Core AI ランタイムで
実行すると fp32 の PyTorch に対して相対誤差 1.45% で一致する（`coreai.runtime.AIModel`）。
壊れるのは ANE で実行したときだけ。Mac 側では `SpecializationOptions` に
`neural_engine` を指定しても値が変わらなかったので、**この現象は Mac では再現できない**。

**対処: VAEDecoder だけ GPU で動かす**（`CoreAIBackend.ComputeUnitPlan.recommended`）。
全体を GPU に落とせば直るが、4 ステップ回る UNet まで巻き添えになる。UNet は ANE で
参照と一致しているので、VAE だけ逃がす。VAE は 1 回しか回らないので全体時間への影響は小さい。

TODO(core-ai): ANE で何が起きているかは未解明。fp16 のオーバーフローなら NaN/Inf が
出るはずだが出ていない（統計に NaN/Inf の検出を入れてある）。ANE 側の中間精度か、
特定の op の扱いの違いと思われる。Core AI の正式リリース後に再確認すること。

### 実機での到達点

    端末          iPhone（iOS 27）
    プロンプト     "a coffee cup"
    ステップ数     4（LCM）
    生成時間       1.4 秒（3 モデルともコンパイル済みキャッシュあり）
                  4.9 秒（VAE を GPU 用に初回コンパイルしたとき込み）
    バンドル       1.1GB（TextEncoder 235 + UNet 821 + VAEDecoder 47）

### 切り分けに要った道具

どれも「壊れてから作った」もので、次に同じことが起きたときの初手になる。

- `ICON_DEBUG=1` — 各段の入出力の min/max/mean と NaN/Inf を出す。
  **これが無いと「絵が変」以上のことが分からない**
- `scratchpad/ref_stats.py` 相当 — Python 側で同じ道筋の値を出す。突き合わせる相手が要る
- `CoreAIBackend.selfTestMarshalling()` — モデルを触らずに NDArray の往復だけ試す。
  受け渡しが原因かモデルが原因かを最初に切り分ける
- Mac 上で `.aimodel` を直接実行する（`coreai.runtime`）。**モデルが悪いのか
  実行環境が悪いのかは、これでしか分けられない**
- `ICON_UNIT` / `ICON_UNIT_<モデル名>` — 計算ユニットをモデル単位で差し替える

## 7. 非対応端末

Core AI は Apple Intelligence 対応機が前提。`IconDiffusionPipeline` の初期化を失敗させて
`IconDiffusionError.unsupportedDevice` を返し、アプリ側で分岐する想定。
判定ロジックはアプリの要件が決まってから足す（現状は backend の初期化失敗をそのまま伝える）。

## 8. 後処理 — 何を学習で直し、何を後処理で直すか

生成直後の 512px ラスタには、アイコンとして見ると三種類の粗さが乗る。**どれを学習で
追い込み、どれを後処理で消すかを最初に決めておかないと、直せないものを学習で追いかけて
時間を溶かす**（実際に溶かした記録が experiments.md の Trial 9）。

| 症状 | 直し方 | 実装 |
|---|---|---|
| 背景が塗りつぶし | 後処理（決定的） | `remove_flat_background` |
| 中間色・グレーの塗り | 後処理（決定的） | パレット量子化 |
| 破片・ヒゲ・塗り残しの穴 | 後処理（決定的） | `clean_labels` |
| 輪郭のガタつき | 後処理（決定的） | `render_smooth` |
| **構造の破綻**（持ち手 2 つ） | **学習データ**（複数例を見せる） | Trial 12 |
| **意味の誤り**（虫めがねに見えない） | どちらでも直らない | 候補から人が選ぶ |

判断の基準は「**入力から一意に決まるか**」。背景色も、パレットの最近色も、破片の面積も、
輪郭の階段も、絵を見れば一意に決まる。決まるものを確率的な生成に任せる理由はない。
一方で「持ち手が何本あるべきか」は絵からは決まらない（何が描かれるべきかの知識が要る）。
そこは学習データの仕事で、それでも残る分は人の目で落とす。

### 滑らか化のやり方

色ごとのマスクを 1.2px ぼかした濃度場に変え、**二値化する前に出力解像度へ拡大**してから
0.5 で切る。512px の階段は振幅がちょうど 1px なので、ぼかすと階段だけが消えて角は残る
（ガウスぼかしが動かす量は曲率に比例するため）。拡大を二値化の前に置くのが肝で、
これにより 512px の情報量のまま 1024px の綺麗な輪郭が得られる。

ベクタに起こして描き直す手もある（姉妹プロジェクトの `Vectorizer` は Vision の
輪郭抽出を使っている）が、こちらは numpy と Foundation だけで書けて移植が軽い。

### 使われ方に合わせる（SNS アイコン）

形が綺麗になっても、**使われる場所で成立しなければ意味がない**。SNS のプロフィール画像は
ほぼ例外なく円形に切り抜かれ、タイムラインでは 48px 前後まで縮む。`icon_fit.py` はそこを扱う。

| 基準 | 根拠 | 実装 |
|---|---|---|
| 四隅は切れる前提で中央に寄せる | 各 SNS が円クロップ | `plan_fit`（視覚重心 + 塗りの 98% が安全領域に入る倍率） |
| 隣接色と 3:1 以上 | WCAG 2.1 の 1.4.11 | `choose_background`（**コーラル×白は 2.82:1 で未達**） |
| 端の切れ端は被写体ではない | 学習データは四辺に等しい余白を指定して作っている | `drop_scraps` |
| 48px で判別できる | X のタイムライン表示が約 48px | `legibility_report`（測るだけ） |

**背景を敷くのは透過を捨てるためではない。** 透過のままだとタイムラインの地の色に
左右され、コントラストを保証できない。敷く色は輪郭に出ている色から決める
（面積ではなく**輪郭の長さ**で数える。WCAG が求めるのは隣接色との比なので、
カップの内側の白のように背景と接しない色まで数えると、どの背景でも基準を満たせなくなる）。

### 参照実装と移植

`training/src/icon_diffusion/shapes.py` と `icon_fit.py` が参照実装、
`IconShapes.swift` の `IconShapes` / `IconFit` がその移植。
**採点式は一字一句同じであること。** Mac で測った数字（形の合格率）がそのまま端末の
候補選びに繋がるのが、この二重実装の目的。`IconShapesTests` が Python の出力値を
そのまま期待値に置いて突き合わせている。

## 9. ControlNet を端末に載せる（2026-09-15・変換可能性の確認まで）

Trial 19〜21 で ControlNet が効くと分かったので、Core AI へ持っていけるかを先に確かめた。
**2 つの関門とも通った。**

| 関門 | 結果 |
|---|---|
| ControlNet 単体を `torch.export` できるか | **通る。** 361M パラメータ、出力は **13 本のタプル**（down 12 + mid 1） |
| UNet が残差を受け取る形で書き出せるか | **通る。** 入力 3 本 → **16 本**（sample / timestep / encoder_hidden_states + 残差 13） |

### 残っている作業

1. **Stage 5**: ControlNet と「残差つき UNet」を書き出す。出力名は 13 本ぶん明示する
   （名前を付けないとグラフ内部の自動生成名になり、モデルが少し変わるだけで名前が変わる。
   6.5 節と同じ轍）
2. **`pipeline_config.json`**: 2 モデルぶんの入出力契約を足す。Swift にハードコードしない
3. **Swift**: 毎ステップ ControlNet → UNet の順に回し、**13 本のテンソルを受け渡す**。
   いまの `DiffusionBackend` は 1 モデル 1 出力を前提にしているので、ここは設計変更になる
4. **グリフの同梱**: SVG なら 1.8MB、512px の PNG なら 29MB。端末では 256px で足りる見込み
5. **引き当てをどう動かすか** — 下記の判断が要る

### 引き当ての置き場所（未決）

Mac 側は SigLIP で「被写体の文章 → グリフの絵」を引いている。端末で同じことをするには
**テキスト側のエンコーダを 1 つ載せる**（int8 で 50MB 前後）。載せない場合は、
**あらかじめ計算した「語 → グリフ」の表**を同梱して文字列で引くことになる。

| 方式 | 追加サイズ | 任意の入力 | 備考 |
|---|---|---|---|
| SigLIP テキスト側を載せる | 約 50MB | **効く** | Mac と同じ結果になる |
| 語 → グリフの表を同梱 | 約 100KB | 語彙の中だけ | 外れたら条件なしに落ちる |

### バンドルの見積もり

    現行                    1.1GB
    + ControlNet（int8）    約 361MB
    + グリフ（256px PNG）   約 8MB
    + 引き当て（方式による） 0.1〜50MB
    = 約 1.5GB（Apple の指針は 2GB 未満）
