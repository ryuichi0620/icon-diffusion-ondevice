# icon-diffusion-ondevice

キャッチーでシンプルなアイコンを生成するモデルを作る。
最終的に iPhone のオンデバイスで動かすのが目標だが、**まずはモデルを完成させる**。

基本方針は一行で言うと **「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」**。
少ステップ化（LCM）を自前で蒸留するのは手数が多いので、そこは配布済みのものに乗る。

---

## いまやること — モデルを完成させる

**フォルダに素材を入れて、フォルダにアイコンが出てくる**ところまで。iOS は関係ない。

```
 ~/icons_raw/                     クラウドAIで作ったアイコン画像を放り込む
      │
      │  make smoke               まず配管の確認（数秒・モデル不要）
      │  make dataset             Stage 1  整形＋スタイルの揃い具合を検査
      ▼
 training/data/processed/         512x512 ＋ captions
      │
      │  make train               Stage 2  アイコン風スタイル LoRA を学習
      ▼
 training/outputs/icon-lora/      自作 LoRA（ここが唯一の自作物）
      │
      │  make preview             Stage 3  既製の LCM-LoRA と重ねて 4 ステップ生成
      ▼                                    ← ここが最初の山
 training/outputs/preview/        効き具合の確認用
      │
      │  make generate            量産。背景を透過にして書き出す
      ▼
 training/outputs/generated/      *.png ＋ contact_sheet.png ＋ manifest.json
```

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke_test.py                       # 配管の確認（数秒・モデル不要）
python scripts/01_prepare_dataset.py --input ~/icons_raw
./scripts/02_train_lora.sh
python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0
python scripts/generate.py --prompts prompts.example.txt --count 4 --transparent
```

### 「完成」をどこで判定するか

決めておかないと終わらない。おすすめの線引き:

> `prompts.example.txt` の 12 被写体 × 4 枚 = 48 枚を **4 ステップ**で生成し、
> そのうち **7 割以上**が「そのままアイコンとして使える」なら合格。

`contact_sheet.png` を眺めて数えるだけでよい。落ちたときの手当ては下の「詰まったら」を参照。

### このプロジェクトで一番危ういところ

**LCM-LoRA（速くする）と スタイル LoRA（アイコンらしくする）の重ね掛けは、互いを弱める。**
どちらも同じ層に効くので、片方を強くするともう片方が薄まる。
「4 ステップでアイコンらしく出るか」が最初の山だと企画メモが書いているのは、実質これのこと。

だから Stage 3 に `--sweep-style` がある。被写体を固定して重みだけ振り、
効きの当たりを取ってから量産に進む。

### 詰まったら

| 症状 | 手当て |
| --- | --- |
| 4 ステップだと崩れる | `--steps 8` にする。速度は端末に載せる段で詰めればよい |
| アイコンらしくない（写真的・立体的） | `--style-weight` を上げる。それでも駄目なら学習データのスタイルが揃っていない |
| スタイルは出るが被写体が出ない | `--style-weight` を下げる。焼き付きなら `max_train_steps` を減らす |
| 全体にぼやける | `--lcm-weight` を下げる。LCM が強すぎる |
| そもそも 4 ステップが厳しい | ベースを `stabilityai/sd-turbo` に変える（`configs/icon_lora.toml`） |

学習データ側が原因のことも多い。Stage 1 が出す `report.json` の警告を先に見ること。

---

## あとでやること — iPhone に載せる

モデルが固まってから着手する。**いまは触らなくてよい。**

```
 outputs/icon-lora/
      │  make fuse                Stage 4  LoRA を UNet に焼き込んで一体型にする
      ▼
 outputs/fused/
      │  make export              Stage 5  Core AI 形式（3 モデル）へ変換
      ▼
 outputs/coreai/*.aimodel
      │                           Stage 6  ios/IconDiffusionKit で 4 ステップ推論
      ▼
 iOS アプリ内で生成
```

Swift 側（`ios/IconDiffusionKit`）は先に書いてあるが、Core AI の API は**未検証**。
詳しくは [ios/IconDiffusionKit/README.md](ios/IconDiffusionKit/README.md)。

---

## リポジトリ構成

```
training/            Mac 側。ここだけでモデルは完成する
  configs/           全ステージが読む単一の設定ファイル
  scripts/           01〜05 と generate.py
  src/icon_diffusion 各スクリプトが共有するロジック
  prompts.example.txt 生成する被写体のリスト
ios/IconDiffusionKit Swift Package。端末に載せる段になってから
docs/                企画・設計・データセット方針
```

- **検証手順: [docs/verification.md](docs/verification.md)** — 何を実行して何を見るか。まずここ
- 作業時の取り決め: [CLAUDE.md](CLAUDE.md)
- 企画メモ: [docs/plan.md](docs/plan.md)（Notion 原本の内容をリポジトリに固定したもの）
- 設計: [docs/architecture.md](docs/architecture.md)
- データセット方針: [docs/dataset.md](docs/dataset.md)
- **実験ログ: [docs/experiments.md](docs/experiments.md)** — 歩留まりを上げるために何を試し、なぜそれを選んだか
- 学習パイプラインの詳細: [training/README.md](training/README.md)

`make help` で各ステージの入口を一覧できる。

## 動作確認できている範囲

- **Stage 3 / generate / Stage 4 / Stage 5** — 極小モデルで通しで実行済み（`make smoke`）。
  LoRA 2 枚の重ね掛け・重み変更・焼き込み・3 モデル分割まで動く
- **Stage 1** — 合成データで通しで実行・検証済み（前処理・キャプション・スタイル逸脱の検出）
- **背景の透過** — 検証済み。アイコン内側の白抜きを残したまま外周だけ抜けることを確認
- **Stage 2** — `--dry-run` で両モードのコマンド生成を確認
- **Swift 側** — ロジックは Python 参照実装との一致を確認済み。コンパイルは未実施

**実モデル（SD1.5 級）での学習・生成は未実施。** 配管は通っているが、
絵の品質は Mac 側で実際に回して見るしかない。

## Notion

企画の原本と、ここの文書のミラーは Notion のプロジェクトページにある。
[アイコン生成モデル — 少ステップ拡散モデルのオンデバイス化](https://app.notion.com/p/3c32bbb6de9d813783dbc708f4839a74)

`docs/` に文書を足したら、Notion 側にも子ページとして追加する（[CLAUDE.md](CLAUDE.md) 参照）。

## ライセンス・権利まわり

学習データは**クラウド生成で自前調達する**方針。市販アイコンセットをそのまま学習に使わない。
詳細は [docs/dataset.md](docs/dataset.md)。
