# IconDiffusionKit

企画メモ Stage 6 の「iOS アプリに組み込む」部分。**Core AI 依存は 1 ファイルに閉じ込めてある。**

```
IconDiffusionPipeline          ループ・CFG・潜在の扱い          純 Swift
  ├─ CLIPTokenizer             テキスト -> input_ids            純 Swift・テストあり
  ├─ LCMScheduler              ノイズ予測 -> 次の潜在            純 Swift・テストあり
  └─ DiffusionBackend          3 モデルを動かす口（protocol）
       ├─ CoreAIBackend        ★ Core AI 依存はここだけ（既定では無効）
       └─ StubBackend          モデル無しで配線を検証する
```

## いま動くもの / 動かないもの

| | 状態 |
| --- | --- |
| `CLIPTokenizer` | 実装済み。Python 参照実装との一致をテストで固定 |
| `LCMScheduler` | 実装済み。diffusers の `LCMScheduler` との一致をテストで固定 |
| `IconDiffusionPipeline` | 実装済み。`StubBackend` で配線を検証済み |
| `CoreAIBackend` | **未検証・既定で無効**。下記参照 |

`swift test` は Core AI 無しで通る。**ただしこのリポジトリを作った環境には Swift ツールチェーンが
無かったため、コンパイル自体はまだ一度も通していない。** Mac で `swift test` を通すのが最初の作業。

## テスト

```bash
swift test
```

Core AI もモデルファイルも要らない。期待値は `Tests/IconDiffusionKitTests/Fixtures/` に
JSON で置いてあり、中身は Python の参照実装（diffusers の `LCMScheduler`、
openai/CLIP の `simple_tokenizer`）で生成したもの。

テストが押さえているのは主にこの 3 点:

- **タイムステップの並び** — 4 ステップなら `[999, 759, 499, 259]`。ここがずれると絵が破綻する。
  期待値は手で書かず `training/scripts/gen_scheduler_fixture.py` で diffusers から生成する
- **BPE の最小ランク優先** — 左から貪欲にマージする実装だと `icon` の分解が変わる。
  そこを踏むテストを入れてある
- **CFG の実行回数** — `guidanceScale > 1.0` で UNet 呼び出しが 2 倍になることを回数で確認

## Core AI を有効にする

`CoreAIBackend.swift` は既定でビルド対象から外してある。中身が WWDC26 の実 API に対して
**未検証**だからで、推測で書いたコードが既定で混ざると、ビルドが通らないか静かにズレるかの
どちらかになる。

有効にするには `Package.swift` のターゲットに定義を足す:

```swift
.target(
    name: "IconDiffusionKit",
    swiftSettings: [.define("ICON_DIFFUSION_COREAI")]
)
```

その前に `CoreAIBackend.swift` の `TODO(core-ai)` を実物と突き合わせること:

- WWDC26「Meet Core AI」(324) /「Dive into Core AI model authoring and optimization」(325)
- `apple/coreai-models` — パッケージの組み方の見本
- `apple/ml-stable-diffusion` — 旧 Core ML 版。モデルの分け方と Swift ループの下敷き

API がずれていても、直すのは `CoreAIBackend.swift` だけで済む。`DiffusionBackend` さえ
満たせばパイプラインもスケジューラも変更不要。

## 使い方

Stage 5 の出力（3 つの `.aimodel` と `vocab.json` / `merges.txt` / `pipeline_config.json`）を
アプリのバンドルに入れる。モデルは GB 級になるので Swift Package には同梱しない。

```swift
import IconDiffusionKit

let resources = try PipelineResources.bundled(in: .main)
let configuration = try PipelineConfiguration.load(contentsOf: resources.configurationURL)
let backend = try await CoreAIBackend(resources: resources, configuration: configuration)
let pipeline = try IconDiffusionPipeline(resources: resources, backend: backend)

let image = try await pipeline.generateIcon(
    subject: "a coffee cup",
    options: .init(steps: 4, guidanceScale: 1.0, seed: 42) { done, total in
        print("\(done)/\(total)")
    }
)
let cgImage = image.makeCGImage()
```

`generateIcon(subject:)` は学習時と同じテンプレート
（`an icon in {trigger} style, {subject}, flat, minimal`）に被写体を埋める。
トリガーワードの付け忘れが起きないので、アプリからは基本こちらを使う。

## 押さえておく点

**`guidanceScale` は 1.0 が既定。** 1.0 を超えると CFG が有効になり、UNet の実行回数が
2 倍になる。企画メモ Stage 3 の `guidance_scale=1.2` は Mac での質感確認向けの値で、
端末でそのまま使うと 4 ステップの生成が UNet 8 回になる。

**`seed` は端末内での再現性のためのもの。** Mac の Stage 3 プレビューと同じ seed を指定しても
同じ絵にはならない（`torch.randn` と別の乱数列を使っているため）。

**非対応端末。** `CoreAIBackend` の初期化が `IconDiffusionError.unsupportedDevice` を投げる。
アプリ側でここを捕まえて逃げ道に分岐する。

**メモリ。** fp16 のまま 3 モデル載せると 2GB を超える。INT8 量子化は前提。
内訳は [../../docs/architecture.md](../../docs/architecture.md) を参照。
