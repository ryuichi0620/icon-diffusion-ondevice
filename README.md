# icon-diffusion-ondevice

キャッチーでシンプルなアイコンを、**iPhone の中だけで**生成するプロジェクト。

基本方針は一行で言うと **「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」**。
少ステップ化（LCM）を自前で蒸留するのは手数が多いので、そこは配布済みのものに乗る。

- 企画メモ: [docs/plan.md](docs/plan.md)（Notion 原本の内容をリポジトリに固定したもの）
- 設計: [docs/architecture.md](docs/architecture.md)
- データセット方針: [docs/dataset.md](docs/dataset.md)

## 全体像

```
[クラウドAIで生成したアイコン画像]
        │  Stage 1  training/scripts/01_prepare_dataset.py
        ▼
[512x512 + captions の学習セット]
        │  Stage 2  training/scripts/02_train_lora.sh
        ▼
[アイコン風スタイル LoRA]  ── ＋ 既製の LCM-LoRA（高速化）
        │  Stage 3  training/scripts/03_preview_lcm.py   ← Mac で 4 ステップの質感チェック
        │  Stage 4  training/scripts/04_fuse_lora.py     ← LoRA を UNet に焼き込む
        ▼
[一体型の SD1.5 パイプライン]
        │  Stage 5  training/scripts/05_export_coreai.py
        ▼
[TextEncoder.aimodel / UNet.aimodel / VAEDecoder.aimodel]
        │  Stage 6  ios/IconDiffusionKit
        ▼
[iOS アプリ内で 4 ステップ生成]
```

## リポジトリ構成

```
training/            Mac 側。データ整形 → LoRA 学習 → LCM 検証 → fuse → Core AI 変換
  configs/           全ステージが読む単一の設定ファイル
  scripts/           Stage 1〜5 を番号順に実行する
  src/icon_diffusion 各スクリプトが共有するロジック
ios/IconDiffusionKit Swift Package。トークナイザ・LCM スケジューラ・推論ループ
  Sources/           バックエンド抽象を挟んで Core AI 依存を 1 ファイルに閉じ込めている
  Tests/             スケジューラとトークナイザは Core AI 無しで検証できる
docs/                企画・設計・データセット方針
```

## 使い方

### Mac 側（Stage 1〜5）

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Stage 1: 生成済みアイコンを 512x512 + captions に整形
python scripts/01_prepare_dataset.py --input ~/Downloads/icons_raw

# Stage 2: スタイル LoRA を学習（MPS でも回るが、遅ければ時間貸し GPU）
./scripts/02_train_lora.sh

# Stage 3: LCM-LoRA と重ねて 4 ステップ生成。ここが最初の山
python scripts/03_preview_lcm.py --steps 4

# Stage 4: 納得したら UNet に焼き込む
python scripts/04_fuse_lora.py

# Stage 5: Core AI 形式へ変換（3 モデル）
python scripts/05_export_coreai.py
```

`make help` に同じことを短縮形で並べてある。

### iOS 側（Stage 6）

```bash
cd ios/IconDiffusionKit
swift test          # スケジューラ・トークナイザの検証（Core AI 不要）
```

アプリからは Swift Package として追加し、Stage 5 で出力した 3 つの `.aimodel` と
`vocab.json` / `merges.txt` をバンドルに入れて使う。

```swift
let pipeline = try IconDiffusionPipeline(resources: .bundled(in: .main))
let image = try await pipeline.generate(
    prompt: "a coffee cup, flat, minimal",
    options: .init(steps: 4, guidanceScale: 1.0, seed: 42)
)
```

## 現状と既知の未確定事項

このリポジトリは企画からの初期スキャフォールドで、**まだ一度も通しで動かしていない**。
特に以下は実機・実 API での確認が必要:

- **Core AI の API 名**（`coreai_torch.TorchConverter` / `import CoreAI` / `AIModel` / `NDArray`）は
  企画メモに書かれた形をそのまま写している。WWDC26 のセッション 324 / 325 と
  `apple/coreai-torch` の実物に対して要検証。該当箇所には `TODO(core-ai)` を付けてある。
  Swift 側は `DiffusionBackend` プロトコルを挟んであるので、API がずれても
  `CoreAIBackend.swift` の差し替えで済む。
- **Swift 側はこの環境ではコンパイルしていない**（Linux コンテナに Swift ツールチェーンが無い）。
  Mac で `swift test` を通すのが最初の作業になる。
- ベースモデルの選定（SD-Turbo か SD1.5＋LCM-LoRA）は未決。`configs/icon_lora.toml` の
  `base_model` を差し替えれば Stage 2〜5 は追従する。

## ライセンス・権利まわり

学習データは**クラウド生成で自前調達する**方針。市販アイコンセットをそのまま学習に使わない。
詳細は [docs/dataset.md](docs/dataset.md)。
