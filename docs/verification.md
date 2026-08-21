# 検証手順

上から順に。**各段階に「合格ライン」がある**ので、そこを越えてから次に進む。
前の段階を飛ばして先に進むと、どこが原因で駄目なのかが分からなくなる。

| # | 何を確かめるか | 必要なもの | かかる時間 |
| --- | --- | --- | --- |
| 0 | 環境が整っているか | Python 3.11+ | 5 分 |
| 1 | **コードの配管が通っているか** | なし | 数秒 |
| 2 | 学習データが使えるか | アイコン画像 | 1 分 |
| 3 | 学習が回るか | ベースモデル（初回 DL） | 数十分〜 |
| 4 | **4 ステップでアイコンらしく出るか** | 学習済み LoRA | 数分 |
| 5 | 量産できるか＝完成判定 | 同上 | 数分 |

1 と 4 が関門。1 が通ればコードのせいではないと切り分けられ、
4 が通ればプロジェクトとして成立したことになる。

---

## 0. 環境を用意する

```bash
git clone icon-diffusion-ondevice.bundle icon-diffusion-ondevice
cd icon-diffusion-ondevice/training

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**合格ライン**: エラーなく終わる。

```bash
python scripts/generate.py --help
```

これが動けば OK。重い依存は遅延読み込みにしてあるので、
**`--help` は torch が入っていなくても通る**（＝これが通っても pip が成功したとは限らない）。
確実に見るならこちら:

```bash
python -c "import torch, diffusers; print(torch.__version__, diffusers.__version__)"
```

---

## 1. 配管の確認（モデル不要・数秒）★関門

```bash
python scripts/smoke_test.py
```

ランダムな重みの極小モデルをその場で組み立て、
Stage 3 → generate → Stage 4 → Stage 5 を実際に走らせる。**ネットワークも実モデルも要らない。**

**合格ライン**: 最後にこれが出る。

```
全部通りました。パイプラインの配管は生きています。
```

途中でこういう行が並ぶ。全部に ✓ が付いていること:

```
   ✓ 01_w0.00_a-coffee-cup.png と 02_w1.00_a-coffee-cup.png で出力が変わる
   ✓ manifest: 4 枚, seed=[0, 1, 0, 1]
   ✓ pipeline_config.json: latent=32 x2 -> image=64 (実測 64) timesteps=[999, 759, 519, 279]
```

**ここで確かめていること**:

| 行 | 意味 |
| --- | --- |
| 重み 0.0 と 1.0 で出力が変わる | LoRA の重み設定が本当に効いている |
| seed=[0, 1, 0, 1] | 同じ被写体の 2 枚が別の seed になっている |
| timesteps=[999, 759, 519, 279] | LCM のタイムステップが diffusers と一致 |
| image=64 (実測 64) | 設定が言うサイズと実際の生成サイズが一致 |

**落ちたら**: 環境の問題。ライブラリのバージョン違いがほとんど。
失敗したスクリプトの標準エラーがそのまま出るので、そこを見る。
**この段階で落ちているなら、実モデルを落としても無駄。** 先に直す。

> 絵の中身はランダム重みなのでただのノイズ。ここでは品質を見ていない。

---

## 2. 学習データを確認する

クラウド AI で作ったアイコンを 1 つのフォルダに集めてから:

```bash
python scripts/01_prepare_dataset.py --input ~/icons_raw
```

**合格ライン**: 警告が「枚数が少なめ」だけになること。

```
[stage1] 完了      : 120 枚 -> .../data/processed/images
[stage1] 例        : an icon in ICONST style, a coffee cup, flat, minimal
[stage1] スタイルのばらつきは許容範囲です
```

**警告が出たら**、内容ごとに手当てが違う:

| 警告 | 意味 | どうするか |
| --- | --- | --- |
| フラット塗りでない画像が N 枚あります | 写真・グラデーション・影が混ざっている | 名指しされたファイルを見て、外す |
| 背景色が揃っていない | 単色と透過が混在、または背景色がバラバラ | どちらかに統一して作り直す |
| 彩度のばらつきが大きい | 配色パレットが揃っていない | 生成時のプロンプトで色を固定する |
| 塗りの均一さがばらついています | ディテール量に差がある | 情報量の多い絵を外す |
| N 枚は少なめ | 50 枚未満 | 増やす。ただしこれだけなら進んでよい |

数字で見たいときは `data/processed/report.json`。`flatness` が主指標で、
フラットなアイコンなら 0.99 前後、写真が混ざると一気に下がる。
1 枚ずつの値も `images` に入っているので、犯人を特定できる。

**目視でも 1 回見る**: `data/processed/images/` を Finder のギャラリー表示で眺めて、
浮いている絵が無いか確認する。自動判定は「塗りの均一さ」しか見ていないので、
構図や余白のバラつきは人間が見るしかない。

---

## 3. 学習が回ることを確認する

**いきなり本番を回さない。** まず何が実行されるかを見る:

```bash
./scripts/02_train_lora.sh --dry-run
```

`--dry-run` は依存もネットワークも学習画像も要らない（無ければ警告するだけ）。
Stage 1 の前でも「何が実行されるか」を先に確認できる。

**合格ライン**: `accelerate launch ...` の全文が表示される。
`--instance_data_dir` が Stage 2 で作った `data/processed/images` を指していること、
`--instance_prompt` にトリガーワードが入っていることを目で確認する。

問題なければ本番:

```bash
./scripts/02_train_lora.sh
```

初回はベースモデル（SD1.5 系、数 GB）のダウンロードが走る。

**合格ライン**: `outputs/icon-lora/pytorch_lora_weights.safetensors` ができる。

```bash
ls -la outputs/icon-lora/
```

**時間の目安**: Apple Silicon の MPS で 800 ステップだと**数十分〜数時間**。
機種差が大きいので、最初の数十ステップの進み方を見て見積もること。
遅すぎるなら時間貸し GPU を 1 時間借りるほうが早い。

**落ちたら**:

| 症状 | 原因 | 手当て |
| --- | --- | --- |
| `学習画像がありません` | Stage 1 未実行 | 2 に戻る |
| メモリ不足で落ちる | バッチが大きい | `train_batch_size` は既に 1。`--resolution 384` を試す（下記） |
| MPS のオペ未対応で落ちる | | `PYTORCH_ENABLE_MPS_FALLBACK=1`（スクリプトが自動で設定済み） |
| 学習スクリプトが取得できない | diffusers のバージョンと example のズレ | `DIFFUSERS_REF=v0.31.0 ./scripts/02_train_lora.sh` |

知らないオプションはそのまま `accelerate launch` に渡される。
設定より**後ろに付くので後勝ちになる**（＝一時的な上書きに使える）:

```bash
./scripts/02_train_lora.sh --dry-run --resolution 384 --lr_scheduler cosine
```

恒久的に変えるなら `configs/icon_lora.toml` を編集する。

---

## 4. 4 ステップで出るかを確認する ★関門

**このプロジェクトが成立するかどうかがここで決まる。**

```bash
python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0
```

被写体を 1 つに固定して、スタイル LoRA の効き具合だけを 4 段階振る。

**まず出力の 1 行目を必ず見る**:

```
[stage3] device=mps dtype=torch.float16
```

`device=cpu` になっていたら MPS が使えていない。異常に遅くなるので先に直す。

**次にこの警告が出ていないことを確認する**:

```
[warn] スタイル LoRA が見つかりません: .../outputs/icon-lora
       高速化 LoRA だけで生成します（Stage 2 未実行？）
```

出ていたら**自作 LoRA が効いていない**。何を見ても意味がないので Stage 2 に戻る。

**合格ライン**: `outputs/preview/contact_sheet.png` を開いて、
4 枚のうち少なくとも 1 枚が「アイコンらしい」こと。

コンタクトシートには重みと秒数がラベルで入る。この 4 枚を見比べて、
一番いい重みを `configs/icon_lora.toml` の `preview.style_weight` に書く。

**落ちたら**（＝どれもアイコンらしくない）、症状で切り分ける:

| 症状 | 意味 | 手当て |
| --- | --- | --- |
| 全部ぐちゃぐちゃ・形にならない | 4 ステップが厳しい | `--steps 8` で試す。8 で出るならステップ数の問題 |
| 写真的・立体的 | スタイルが効いていない | 重みを上げる。1.0 でも駄目なら学習データを疑う（2 に戻る） |
| スタイルは出るが被写体が出ない | スタイルが強すぎる／焼き付き | 重みを下げる。それでも駄目なら `max_train_steps` を減らして学習し直す |
| 全体にぼんやり眠い | LCM が強すぎる | `--lcm-weight 0.8` を試す |
| 8 ステップでも駄目 | 学習が足りない or データの問題 | 2 に戻る。それでも駄目ならベースを `stabilityai/sd-turbo` に替える |

**トリガーワードが効いているかの確認**（任意だが有用）:

```bash
echo "a coffee cup, flat, minimal" > /tmp/no-trigger.txt
python scripts/generate.py --raw-prompts /tmp/no-trigger.txt --out outputs/check-trigger
```

トリガーワード無しの絵と、`generate.py` の通常出力を見比べる。
**違いが無ければトリガーワードが機能していない**（学習時と生成時で綴りが違う等）。

---

## 5. 量産して合格を判定する

```bash
python scripts/generate.py --prompts prompts.example.txt --count 4 --transparent
```

12 被写体 × 4 枚 = 48 枚が `outputs/generated/` に出る。

**合格ライン（プロジェクトの完成判定）**:

> `contact_sheet.png` を見て、48 枚のうち **7 割以上**が
> 「そのままアイコンとして使える」なら合格。

数えるだけでよい。落ちたら 4 の表に戻る。

**あわせて確認する**:

```bash
# 透過が効いているか（Finder のプレビューで市松模様が見える）
open outputs/generated/a-coffee-cup_01.png

# 1 枚あたりの秒数（出力の最終行に出る）
# [gen] 48 枚 -> ...  (1 枚あたり 1.8s)
```

透過率は生成中に各行へ出る（`透過=62%`）。
**0% が並ぶなら背景が一様でない**＝アイコンとして扱いにくい絵になっている。
学習データの背景が揃っていない可能性が高いので 2 に戻る。

気に入った 1 枚を再現したいときは `manifest.json` を見る。
被写体・プロンプト・seed が記録されているので、同じ条件で出し直せる。

---

## （後回し）iOS 側

モデルが固まってから。いまは触らなくてよい。

```bash
cd ios/IconDiffusionKit && swift test
```

Core AI もモデルファイルも要らない（スケジューラとトークナイザだけを検証する）。
**この環境ではコンパイルしていないので、最初は失敗する可能性がある。**
詳しくは [../ios/IconDiffusionKit/README.md](../ios/IconDiffusionKit/README.md)。

---

## 困ったときの切り分け

```
smoke_test が落ちる            -> 環境の問題。ライブラリのバージョンを見る
smoke_test は通るが 3 で落ちる  -> 実モデル固有の問題。モデルのダウンロードや MPS を疑う
3 は通るが 4 で絵が駄目         -> コードではなく学習の問題。データか重みか学習量
4 は通るが 5 で歩留まりが悪い   -> データの多様性不足。被写体を増やして学習し直す
```

**smoke_test が通っているなら、コードのせいではない**と切り分けられる。
これが 1 を最初に置いている理由。
