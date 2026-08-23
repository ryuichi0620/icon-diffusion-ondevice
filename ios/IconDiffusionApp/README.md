# IconDiffusionApp

`.aimodel` 3 つを端末で回して 1 枚生成するだけの最小アプリ。
`IconDiffusionKit` の動作確認が目的で、製品ではない。

## Core AI は iOS 27 以降・実機のみ

- **iOS 26 の SDK に `CoreAI.framework` は無い。** Xcode 27 以降が要る
- **シミュレータの SDK にも無い。** 実機でしか動かない。
  シミュレータ向けにビルドは通るが、生成しようとすると
  `unsupportedDevice` で落ちる（`ContentView` の `#if` で切ってある）

## 実機に入れるまで

```bash
# 1. Stage 5 で .aimodel を作る（リポジトリのルートで）
cd training
.venv-coreai/bin/python scripts/05_export_coreai.py --quantize int8

# 2. アプリのバンドルにコピーする（約 1GB）
cd ../ios/IconDiffusionApp
./sync-resources.sh

# 3. 端末を USB で繋いで、ビルドしてインストール
./install-device.sh
```

Xcode で開きたいときは `brew install xcodegen && xcodegen && open IconDiffusionApp.xcodeproj`。

### 前提

| 項目 | 内容 |
| --- | --- |
| 端末 | **iOS 27 以降**。Core AI が iOS 27 からなので、それ未満では動かない |
| Xcode | **27 以降**。`install-device.sh` が自動で探す |
| 署名 | Ryuichi Egoshi 個人チーム（`5JD92NU25P`）。`com.egoshi.icondiffusion` |
| 容量 | 約 1GB。端末の空き容量と転送時間に注意 |

署名は自動（`CODE_SIGN_STYLE: Automatic`）。端末を繋いだ状態で
`-allowProvisioningUpdates` 付きでビルドすれば、プロファイルへの端末登録も自動で入る。

### 確認済み

- 実機向けの**署名付きビルドが通る**。`codesign --verify` も通り、
  TeamIdentifier は `5JD92NU25P`、`get-task-allow` 付きの開発ビルド
- **実機に入れて動かしてはいない**（端末が繋がっていないため）

`IconDiffusionApp.xcodeproj` と `Resources/` の中身は**生成物なので追跡していない**。
`project.yml` が唯一の情報源。プロジェクト設定を変えるときは `project.yml` を直す。

## ビルドだけ確かめたいとき

```bash
xcodebuild -project IconDiffusionApp.xcodeproj -scheme IconDiffusionApp \
  -sdk iphoneos27.0 -destination "generic/platform=iOS" \
  CODE_SIGNING_ALLOWED=NO build
```

`IconDiffusionKit` 側だけなら `../typecheck-coreai.sh` のほうが速い。

## 設計上の注意

- **`.aimodel` はディレクトリ。** バンドルにはフォルダ参照で入れる（`project.yml` の
  `type: folder`）。中身をバラしてコピーされるとモデルとして読めなくなる
- そのため直下ではなく `Models/` の下に入る。`PipelineResources.bundled(in:)` は
  `Bundle.url(forResource:)` を使うのでサブフォルダを見てくれない。
  アプリ側は `PipelineResources.directory(_:)` を使っている
- **フォルダ名を `Resources` にしてはいけない。** バンドル直下に `Resources/` があると
  codesign のリソース封印が壊れ、`code object is not signed at all` で署名に失敗する
  （macOS の `Contents/Resources` と紛らわしいのが原因とみられる）。`Models/` にしてある
- **モデルの読み込みは重い**（コンパイルが走る）。`CoreAIBackend` は
  `AIModel.specialize` のキャッシュを使うので、コンパイルは初回だけ。
  アプリは画面が出た時点で `CoreAIBackend.prewarm` を呼び、生成ボタンを
  押してから待たせないようにしている（`docs/architecture.md` 6.6 節）
