# icon-diffusion-ondevice

フラットでシンプルなアイコンを生成するモデルを自作し、iPhone のオンデバイスで動かす。
被写体の名前（`a coffee cup` など）を渡すと、2 色・影なし・背景が透過の 512×512 の PNG が
1 枚出てくる。生成は端末の中だけで完結する。

方針は一行で言うと **「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」**。
少ステップ化（LCM）を自前で蒸留すると手間がかかるので、そこは配布済みのものに乗る。

---

## まず動かしてみる

学習済みのモデルも GPU もネットワークも要らない確認が入っている。数秒で終わるので、
最初にこれを通すとよい。

```bash
make smoke
```

極小のモデルをその場で作り、生成から Core AI 形式への変換までを通しで実行する。
ここが通れば、スクリプトのつながりは壊れていない。各ステージの入口は `make help` で一覧できる。

### 必要なもの

| | |
| --- | --- |
| Mac | Python 3.11 以降。学習には Apple Silicon の GPU（MPS）を使う |
| 学習データ | アイコン画像を自前で用意する（集め方は `training/data/SOURCE.md`）|
| 端末に載せるなら | iPhone（**iOS 27 以降**）と **Xcode 27**。シミュレータでは生成できない |

---

## 全体像

```
 ~/icons_raw/                     学習データ 746 枚（373 被写体 × 2 枚）をクラウド AI で自前生成
      │
      │  make smoke               まず配管の確認（数秒・モデル不要）
      │  make dataset             Stage 1  512x512 ＋ captions に整形し、揃い具合を検査
      ▼
 training/data/processed/
      │
      │  make train               Stage 2  アイコン風スタイル LoRA を学習（唯一の自作物）
      ▼
 training/outputs/icon-lora/
      │
      │  make preview             Stage 3  既製の LCM-LoRA と重ねて少ステップ生成 ← 最初の山
      │  make generate            量産。透過・形の整え・採点まで
      ▼
 training/outputs/generated/      *.png ＋ contact_sheet.png ＋ manifest.json
      │
      │  make fuse                Stage 4  LoRA を UNet に焼き込んで一体型にする
      │  make export              Stage 5  Core AI 形式へ変換（int8・4 モデル）
      ▼
 training/outputs/coreai/*.aimodel
      │
      │  sync-resources.sh → install-device.sh
      ▼
 iPhone のアプリ内で生成（ios/IconDiffusionApp）
```

---

## 使い方

### 1. Mac でモデルを作る

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke_test.py                       # 配管の確認（数秒・モデル不要）
python scripts/01_prepare_dataset.py --input ~/icons_raw
./scripts/02_train_lora.sh
python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0
python scripts/generate.py --prompts prompts.example.txt --count 4 --polish
python scripts/score_shapes.py outputs/generated   # 形の採点で設定を比べる
```

形の下敷き（ControlNet）込みで試すなら、グリフ集を作ってから渡す。

```bash
python scripts/build_glyph_library.py              # Material Symbols 3,912 件＋引き当て表
python scripts/controlnet_style.py --prompts prompts.example.txt --out outputs/cn
```

設定値は [`training/configs/icon_lora.toml`](training/configs/icon_lora.toml) に集めてある。
CLI の引数はその場で試すための上書きなので、恒久的な値はこのファイルに書く。

### 2. iPhone に載せる

```bash
cd training
.venv-coreai/bin/python scripts/05_export_coreai.py --quantize int8   # Stage 5（別の venv）

cd ../ios/IconDiffusionApp
./sync-resources.sh                                # 約 1.4GB をバンドルへコピー
./install-device.sh                                # USB で繋いだ端末へ
```

Core AI はシミュレータの SDK に入っていないので、生成は実機でしか動かない。
venv を分ける理由は [training/requirements.txt](training/requirements.txt)、
端末に入れるまでの詳細は [ios/IconDiffusionApp/README.md](ios/IconDiffusionApp/README.md) にある。

### 3. 「完成」をどこで判定するか

決めておかないと終わらないので、線引きを置いてある。

> `prompts.example.txt` の 12 被写体 × 4 枚 = 48 枚を生成し、
> そのうち **7 割以上**が「そのままアイコンとして使える」なら合格。

`contact_sheet.png` を見て数えるだけでよい。手順は
[docs/verification.md](docs/verification.md) の 5。**機械の採点は設定どうしを比べるための
物差しであって、人の目の代わりにはならない。** 最後は必ず自分の目で確かめること。

---

## 成果物（アーティファクト）

### 発表資料

- **スライド資料** — リンクは後で追加

### リポジトリに入っているもの

| 成果物 | 置き場所 |
| --- | --- |
| 学習・生成・変換の一式（Stage 0〜5） | [`training/scripts/`](training/scripts) ＋ [`training/src/icon_diffusion/`](training/src/icon_diffusion) |
| 全ステージが読む単一の設定 | [`training/configs/icon_lora.toml`](training/configs/icon_lora.toml) |
| Swift の推論 Package（LCM スケジューラと CLIP BPE を純 Swift で実装）| [`ios/IconDiffusionKit/`](ios/IconDiffusionKit) |
| デモアプリ（SNS のプロフィール画面を模した 3 層）| [`ios/IconDiffusionApp/`](ios/IconDiffusionApp) |
| 実験ログ（Trial 0〜24。何を試し、なぜそれを選んだか）| [`docs/experiments.md`](docs/experiments.md) |
| 学習データの被写体リストと出所の記録 | [`training/data/`](training/data) |

### 生成物（大きいのでリポジトリでは追跡していない）

| 成果物 | 置き場所 | 実測 |
| --- | --- | --- |
| 学習データ 746 枚 | `training/data/raw/` | 373 被写体 × 2 枚。出所は `training/data/SOURCE.md` |
| アイコン風スタイル LoRA | `training/outputs/icon-lora/` | rank 16・800 ステップ |
| 焼き込み済みパイプライン | `training/outputs/fused/` | Stage 4 の出力 |
| Core AI モデル 4 つ | `training/outputs/coreai/*.aimodel` | UNetControlled 821MB / ControlNet 345MB / TextEncoder 235MB / VAEDecoder 47MB |
| グリフ 3,912 件と引き当て表 | `training/data/glyphs/` | 26MB。Material Symbols（Apache-2.0）|

---

## つまずきどころ

### 一番危ういところ

**LCM-LoRA（速くする）とスタイル LoRA（アイコンらしくする）の重ね掛けは、互いを弱める。**
どちらも同じ層に効くので、片方を強くするともう片方が薄まる。「少ないステップでアイコンらしく
出るか」が最初の山だというのは、つまりこの綱引きのこと。

そのために Stage 3 に `--sweep-style` がある。被写体を固定して重みだけを振り、
効きの良い範囲を見つけてから量産に進む。

### 症状別の手当て

| 症状 | 手当て |
| --- | --- |
| 4 ステップだと崩れる | `--steps 8` にする（既定も 8。形の採点で 4 ステップを上回った）|
| アイコンらしくない（写真のよう・立体的）| `--style-weight` を上げる。それでも変わらなければ、学習データのスタイルが揃っていない |
| スタイルは出るが被写体が出ない | `--style-weight` を下げる。過学習が疑われるなら `max_train_steps` を減らす |
| 全体にぼやける | `--lcm-weight` を下げる。LCM が強すぎる |
| 何の絵か分からない | 形の下敷き（ControlNet）を付ける。**ここが一番効いた** |
| 枚数を増やしたら背景が崩れた | `max_train_steps` を 800 に戻す。枚数に比例させてはいけない |

原因が学習データ側にあることも多い。Stage 1 が出す `report.json` の警告を先に見ること。

---

## リポジトリ構成

```
training/            Mac 側。ここだけでモデルは完成する
  configs/           全ステージが読む単一の設定ファイル
  scripts/           Stage 00〜05 と generate.py / score_shapes.py / build_glyph_library.py
  src/icon_diffusion 各スクリプトが共有するロジック
  prompts.example.txt 生成する被写体のリスト
ios/IconDiffusionKit Swift Package。推論・スケジューラ・トークナイザ
ios/IconDiffusionApp デモアプリ（SNS のプロフィール画面）
docs/                企画・設計・データセット方針・実験ログ
```

## 文書の地図

手を動かすなら `docs/verification.md`、決定の経緯を知りたいなら `docs/experiments.md` から読む。

| 文書 | 何が書いてあるか |
| --- | --- |
| **[docs/verification.md](docs/verification.md)** | **何を実行して何を見るか** |
| **[docs/experiments.md](docs/experiments.md)** | **歩留まりを上げるために何を試し、なぜそれを選んだか（Trial 0〜24）** |
| [docs/architecture.md](docs/architecture.md) | 設計。モデルを 3 つに割る理由、ANE で壊れた原因、後処理の線引き |
| [docs/dataset.md](docs/dataset.md) | データセット方針。枚数の効き方、権利まわり |
| [docs/base-model-comparison.md](docs/base-model-comparison.md) | ベースモデルの比較（SD1.5 / FLUX.2 klein / Sana Sprint）|
| [docs/plan.md](docs/plan.md) | 企画メモ。最初に決めた狙いと段取り |
| [training/README.md](training/README.md) | 学習パイプラインの詳細 |
| [ios/IconDiffusionApp/README.md](ios/IconDiffusionApp/README.md) | アプリの画面構成と、実機に入れるまで |
| [ios/IconDiffusionKit/README.md](ios/IconDiffusionKit/README.md) | Swift 側の実装状況 |
