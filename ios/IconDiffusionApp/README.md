# IconDiffusionApp

**SNS のプロフィール編集欄を模した 1 画面**のアプリ。画面遷移は無い。
`.aimodel` 4 つを端末で回してアイコンの候補を作り、丸く切り抜いて見せ、
選んだものをプロフィール画像として保存する。

このモデルの出口は SNS のアイコン — 丸く切り抜かれ、他人のタイムラインの
地色の上に 40pt 程度で載る。**選ぶときの条件を、使うときの条件に揃える**ために:

- プレビューも候補も最初から丸（正方形で見ると輪郭の欠けを見落とす）
- 明るい地と暗い地の両方に並べて見せる（背景 `auto` の 3:1 はここで効く）
- 形の下書き（ControlNet）は**既定で入り**。人の目で判定して
  なし 0/5・あり 3/5 だったため（docs/experiments.md Trial 19〜21）
- **機械判定の点は画面に出さない。** 人の判定と 10 枚中 3 枚で食い違い、
  ほぼ空の絵を 0.88 で「合格」にした。開発用の欄にだけ残してある

| ファイル | 役割 |
| --- | --- |
| `ContentView.swift` | 入口。`ProfileEditView` を出すだけ |
| `ProfileEditView.swift` | 画面。状態は持たない |
| `IconStudio.swift` | 生成・後処理・色の差し替え・書き出し・無人実行 |

## Core AI は iOS 27 以降・実機のみ

- **iOS 26 の SDK に `CoreAI.framework` は無い。** Xcode 27 以降が要る
- **シミュレータの SDK にも無い。** 生成は実機でしか動かない。
  シミュレータでも**ビルドと起動は通り、画面の見た目は確かめられる**が、
  「つくる」を押すと `unsupportedDevice` が画面に出る。
  Core AI の型に触る箇所は `IconStudio` の `#if` で切ってある

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
- **シミュレータ（iPhone 17 / iOS 27.0）で起動し、画面を目で確認した。**
  プロフィール編集欄・丸のプレビュー・タイムラインでの見え方（明/暗）・
  「つくる」を押したときの `unsupportedDevice` 表示まで
- **この画面のまま実機で生成してはいない。** 生成そのものは以前の画面で
  実機確認済み（1 枚 5.4 秒）だが、作り替えた後の実機実行はしていない

`IconDiffusionApp.xcodeproj` と `Resources/` の中身は**生成物なので追跡していない**。
`project.yml` が唯一の情報源。プロジェクト設定を変えるときは `project.yml` を直す。

## 写真に保存

選んだアイコンは「写真に保存」で写真アプリへ書き出せる。ここを通らないと
実際に SNS のアイコンに設定できないため入れてある。追加のみの許可
（`INFOPLIST_KEY_NSPhotoLibraryAddUsageDescription`）。
**この経路は実機で試していない**（シミュレータでは候補を作れないため）。

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
