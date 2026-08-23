# IconDiffusionApp

`.aimodel` 3 つを端末で回して 1 枚生成するだけの最小アプリ。
`IconDiffusionKit` の動作確認が目的で、製品ではない。

## Core AI は iOS 27 以降・実機のみ

- **iOS 26 の SDK に `CoreAI.framework` は無い。** Xcode 27 以降が要る
- **シミュレータの SDK にも無い。** 実機でしか動かない。
  シミュレータ向けにビルドは通るが、生成しようとすると
  `unsupportedDevice` で落ちる（`ContentView` の `#if` で切ってある）

## 動かすまで

```bash
# 1. Stage 5 で .aimodel を作る（リポジトリのルートで）
cd training
.venv-coreai/bin/python scripts/05_export_coreai.py --quantize int8

# 2. アプリのバンドルにコピーする（約 1GB）
cd ../ios/IconDiffusionApp
./sync-resources.sh

# 3. プロジェクトを生成して開く
brew install xcodegen     # 未導入なら
xcodegen
open IconDiffusionApp.xcodeproj
```

`IconDiffusionApp.xcodeproj` と `Resources/` の中身は**生成物なので追跡していない**。
`project.yml` が唯一の情報源。プロジェクト設定を変えるときは `project.yml` を直す。

## 確認できている範囲

- **実機（iOS 27）向けのビルドが通る。** `CoreAI.framework` へのリンクも確認済み
- **実機で動かしてはいない。** 署名と実機が要るため

ビルドだけ確かめたいとき:

```bash
xcodebuild -project IconDiffusionApp.xcodeproj -scheme IconDiffusionApp \
  -sdk iphoneos27.0 -destination "generic/platform=iOS" \
  CODE_SIGNING_ALLOWED=NO build
```

`IconDiffusionKit` 側だけなら `../typecheck-coreai.sh` のほうが速い。

## 設計上の注意

- **`.aimodel` はディレクトリ。** バンドルにはフォルダ参照で入れる（`project.yml` の
  `type: folder`）。中身をバラしてコピーされるとモデルとして読めなくなる
- そのため直下ではなく `Resources/` の下に入る。`PipelineResources.bundled(in:)` は
  `Bundle.url(forResource:)` を使うのでサブフォルダを見てくれない。
  アプリ側は `PipelineResources.directory(_:)` を使っている
- **モデルの読み込みは重い**（コンパイルが走る）。`CoreAIBackend` は
  `AIModel.specialize` のキャッシュを使うので、コンパイルは初回だけ。
  アプリは画面が出た時点で `CoreAIBackend.prewarm` を呼び、生成ボタンを
  押してから待たせないようにしている（`docs/architecture.md` 6.6 節）
