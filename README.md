# icon-diffusion-ondevice

**キャッチーでシンプルなアイコンを生成するモデルを作り、iPhone の中だけで動かす。**

方針は一行で言うと **「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」**。
少ステップ化（LCM）を自前で蒸留するのは手数が多いので、そこは配布済みのものに乗る。

| | |
| --- | --- |
| 土台（借り物） | SD1.5 ＋ `latent-consistency/lcm-lora-sdv1-5` |
| 自作 | アイコン風スタイル LoRA（rank 16・800 ステップ・トリガー `ICONST`）|
| 形の下敷き | Material Symbols の輪郭を ControlNet に渡す（**既定で入り**・Apache-2.0）|
| 端末 | iPhone 17 / iOS 27 ＋ Core AI。**1 枚 10.5 秒**・バンドル約 1.4GB |
| 残っている課題 | **絵の質**。後処理では直らないと確かめたので、手当ては学習側 |

---

## いまの状態

**端末までは通っている。残っているのは絵の質。**

- **実機で生成できる** — iPhone 17 / iOS 27 で 1 枚 10.5 秒（2 回目以降）。
  TextEncoder → ControlNet → UNetControlled → VAEDecoder の 4 モデルを Core AI で回す
- **ControlNet が効いた** — 被写体判定 58.3% → **77.1%**（自動引き当て込み・既知 12 被写体）。
  手でグリフを選べば 87.5〜97.9%（[実験ログ](docs/experiments.md) Trial 20〜21）
- **ただし人の目では 5 枚中 3 枚** — 利用者に 10 枚を判定してもらった結果は
  下敷きあり 3 可 / 2 不可、下敷きなし **0 可 / 5 不可**（Trial 24）。
  **機械の点は人の判定と食い違う。** 完成判定の「7 割」にはまだ届いていない
- **指摘された欠陥は後処理では直せなかった** — 「謎の線」「塗り残し」「輪郭のぼやけ」。
  3 案試して 3 案とも撤回した。生成そのものの粗さなので、直すなら学習側
- **未確認** — UI を操作しての実機確認、写真アプリへの書き出し、
  TextEncoder の per-tensor int8（埋め込みの数値は合ったが、絵をまだ出していない）

優先順位は **モデルの質が先**。端末側は通っているので、いまは触らなくてよい。

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

`make help` で各ステージの入口を一覧できる。

---

## 成果物（アーティファクト）

### 発表資料

- **スライド資料 — （リンクは後で追加）**

### リポジトリに入っているもの

| 成果物 | 置き場所 |
| --- | --- |
| 学習・生成・変換の一式（Stage 0〜5） | [`training/scripts/`](training/scripts) ＋ [`training/src/icon_diffusion/`](training/src/icon_diffusion) |
| 全ステージが読む単一の設定 | [`training/configs/icon_lora.toml`](training/configs/icon_lora.toml) |
| Swift の推論 Package（LCM スケジューラ・CLIP BPE を純 Swift で実装） | [`ios/IconDiffusionKit/`](ios/IconDiffusionKit) |
| デモアプリ（SNS のプロフィール画面を模した 3 層） | [`ios/IconDiffusionApp/`](ios/IconDiffusionApp) |
| 実験ログ（Trial 0〜24。何を試し、なぜそれを選んだか） | [`docs/experiments.md`](docs/experiments.md) |
| 学習データの被写体リストと出所の記録 | [`training/data/`](training/data) |

### 生成物（大きいのでリポジトリでは追跡していない）

| 成果物 | 置き場所 | 実測 |
| --- | --- | --- |
| 学習データ 746 枚 | `training/data/raw/` | 373 被写体 × 2 枚。出所は `training/data/SOURCE.md` |
| アイコン風スタイル LoRA | `training/outputs/icon-lora/` | rank 16・800 ステップ |
| 焼き込み済みパイプライン | `training/outputs/fused/` | Stage 4 の出力 |
| Core AI モデル 4 つ | `training/outputs/coreai/*.aimodel` | UNetControlled 821MB / ControlNet 345MB / TextEncoder 235MB / VAEDecoder 47MB |
| グリフ 3,912 件 ＋ 引き当て表 | `training/data/glyphs/` | 26MB。Material Symbols（Apache-2.0） |

---

## 使い方

### Mac でモデルを作る

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke_test.py                       # 配管の確認（数秒・モデル不要）
python scripts/01_prepare_dataset.py --input ~/icons_raw
./scripts/02_train_lora.sh
python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0
python scripts/generate.py --prompts prompts.example.txt --count 4 --polish
python scripts/score_shapes.py outputs/generated   # 形の点で設定を比べる
```

形の下敷き（ControlNet）込みで試すなら、グリフ集を作ってから渡す:

```bash
python scripts/build_glyph_library.py              # Material Symbols 3,912 件＋引き当て表
python scripts/controlnet_style.py --prompts prompts.example.txt --out outputs/cn
```

### iPhone に載せる

```bash
cd training
.venv-coreai/bin/python scripts/05_export_coreai.py --quantize int8   # Stage 5（別の venv）

cd ../ios/IconDiffusionApp
./sync-resources.sh                                # 約 1.4GB をバンドルへコピー
./install-device.sh                                # USB で繋いだ端末へ
```

**iOS 27 以降・実機のみ**（Core AI がシミュレータの SDK に無い）。Xcode 27 が要る。
venv を分ける理由と前提は [training/requirements.txt](training/requirements.txt) と
[ios/IconDiffusionApp/README.md](ios/IconDiffusionApp/README.md)。

### 「完成」をどこで判定するか

決めておかないと終わらない。線引きは:

> `prompts.example.txt` の 12 被写体 × 4 枚 = 48 枚を生成し、
> そのうち **7 割以上**が「そのままアイコンとして使える」なら合格。

`contact_sheet.png` を眺めて数えるだけでよい。手順は
[docs/verification.md](docs/verification.md) の 5。**機械の採点は設定を比べる物差しで、
人の判定の代わりにはならない**（10 枚中 3 枚で食い違った）。

---

## つまずきどころ

### このプロジェクトで一番危ういところ

**LCM-LoRA（速くする）と スタイル LoRA（アイコンらしくする）の重ね掛けは、互いを弱める。**
どちらも同じ層に効くので、片方を強くするともう片方が薄まる。
「少ステップでアイコンらしく出るか」が最初の山だと企画メモが書いているのは、実質これのこと。

だから Stage 3 に `--sweep-style` がある。被写体を固定して重みだけ振り、
効きの当たりを取ってから量産に進む。

### 症状別の手当て

| 症状 | 手当て |
| --- | --- |
| 4 ステップだと崩れる | `--steps 8` にする（既定も 8。形の点で 4 步を上回った） |
| アイコンらしくない（写真的・立体的） | `--style-weight` を上げる。それでも駄目なら学習データのスタイルが揃っていない |
| スタイルは出るが被写体が出ない | `--style-weight` を下げる。焼き付きなら `max_train_steps` を減らす |
| 全体にぼやける | `--lcm-weight` を下げる。LCM が強すぎる |
| 何の絵か分からない | 形の下敷き（ControlNet）を付ける。**ここが一番効いた** |
| 枚数を増やしたら背景が崩れた | `max_train_steps` を 800 に戻す。枚数に比例させてはいけない |

学習データ側が原因のことも多い。Stage 1 が出す `report.json` の警告を先に見ること。

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

| 文書 | 何が書いてあるか |
| --- | --- |
| **[docs/verification.md](docs/verification.md)** | **何を実行して何を見るか。まずここ** |
| **[docs/experiments.md](docs/experiments.md)** | **歩留まりを上げるために何を試し、なぜそれを選んだか（Trial 0〜24）** |
| [docs/architecture.md](docs/architecture.md) | 設計。モデルを 3 つに割る理由、ANE で壊れた原因、後処理の線引き |
| [docs/dataset.md](docs/dataset.md) | データセット方針。枚数の効き方、権利まわり |
| [docs/base-model-comparison.md](docs/base-model-comparison.md) | ベースモデルの比較（SD1.5 / FLUX.2 klein / Sana Sprint） |
| [docs/plan.md](docs/plan.md) | 企画メモ。最初に決めた狙いと段取り |
| [training/README.md](training/README.md) | 学習パイプラインの詳細 |
| [ios/IconDiffusionApp/README.md](ios/IconDiffusionApp/README.md) | アプリの画面構成と、実機に入れるまで |
| [ios/IconDiffusionKit/README.md](ios/IconDiffusionKit/README.md) | Swift 側の実装状況 |

## ライセンス・権利まわり

- **学習データはクラウド生成で自前調達する。** 市販アイコンセットをそのまま学習に使わない
  （詳細は [docs/dataset.md](docs/dataset.md)）
- **形の下敷きに使う Material Symbols は Apache-2.0。** 学習には一切使わず、推論時の条件
  （輪郭）としてだけ使う。配布するならライセンス全文と著作権表示を画面から読める場所に置く
- ベースモデル（SD1.5）は CreativeML OpenRAIL-M。乗り換え候補との比較は
  [docs/base-model-comparison.md](docs/base-model-comparison.md)
