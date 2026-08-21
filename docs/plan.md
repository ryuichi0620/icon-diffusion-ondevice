# アイコン生成モデル — 少ステップ拡散モデルのオンデバイス化

> 出典: Notion「Projects」DB の同名ページ
> <https://app.notion.com/p/3c32bbb6de9d813783dbc708f4839a74>
> ステータス: 提案中 / 種別: 個人開発
> このファイルは企画時点の内容をリポジトリに固定したもの。実装で判断が変わった場合は
> ここではなく [architecture.md](architecture.md) 側を更新する。

キャッチーでシンプルなアイコンを、iPhone の中だけで生成できるようにするプロジェクト。
**「速くする LoRA は既製品を借り、アイコンらしさの LoRA だけ自分で作る」** のが基本方針。

## 背景・つくりたい理由

- 画像生成モデルを使って、キャッチーでシンプルなアイコンを生成できるモデルを作りたい
- WWDC26 で発表された Core AI フレームワークを使って再学習をやってみたい、というのが出発点
- 動かす場所は **iOS アプリ内（オンデバイス）**。サーバーに頼らず端末内で完結させる
- 学習用データセットは **クラウドベースの AI で生成**して用意する
- 品質重視（Flux 中心・Mac/サーバー生成）も検討したが、最終的に **速度重視・オンデバイス** に決定
- 学習まわりはまず **MLX ベースでどこまでいけるか**を試したい

## 前提の整理（重要）

| 役割 | 使うもの | 補足 |
| --- | --- | --- |
| 再学習（ファインチューニング） | MLX / PyTorch・diffusers | Core AI は学習用ではない。SD 系の学習は現状 diffusers のほうが道具が揃っている |
| 端末向け変換・最適化 | coreai-torch / coreai-optimization | PyTorch モデル → Core AI 形式（`.aimodel`） |
| 端末での実行 | Core AI（Swift API） | Core ML の後継。拡散モデル用のランタイムを持つ |

**「Core AI で再学習」ではなく「MLX/PyTorch で再学習 → Core AI で動かす」** という役割分担になる。

## 手法の調査結果

### スタイルを覚えさせる（省メモリなファインチューニング）

- **LoRA** — 拡散モデルのスタイル学習の定番。cross-attention 層に入れるのが一般的
- **QLoRA** — 土台を 4bit 量子化した上に LoRA を乗せる。メモリ節約が大きい
- **DoRA** — LoRA の改良版。MLX は lora / dora / full から選べる
- 補助として DreamBooth / Textual Inversion

### スマホで速く・軽く動かす（オンデバイスの肝）

- **LCM（Latent Consistency Model）** — 2〜4 ステップ、条件次第で 1 ステップ生成
- **SDXL Turbo / Lightning** — 敵対的蒸留（ADD）系で 1〜4 ステップ
- **QAT（量子化を意識した学習）** — 低精度でも精度が落ちにくい
- **小型モデルへの知識蒸留** — EdgeFusion 等、圧縮 UNet ＋ LCM 蒸留の事例あり

### 採用する組み合わせ

少ステップ化は**自分で蒸留せず、配布済みの蒸留モデル・LCM-LoRA を使う**。
自作するのは**アイコンのスタイル LoRA だけ**。これが一番迷いが少ない。

## 具体的な手順

### Stage 1: データセットを用意する

- クラウドで生成したアイコン画像を 512×512 で `icons/` に集める
- スタイルを揃えること（バラつくと学習が濁る）
- **トリガーワード**を決める（例: `ICONST`）。生成時にこの語を入れるとスタイルが呼び出せる
- 枚数は 50〜200 枚で十分

→ 実装: [`training/scripts/01_prepare_dataset.py`](../training/scripts/01_prepare_dataset.py)

### Stage 2: アイコン用スタイル LoRA を学習（Mac）

diffusers の DreamBooth-LoRA スクリプトが確実。ベースは軽さ優先で SD1.5 系。

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="<SD1.5系のベースID>" \
  --instance_data_dir="./icons" \
  --instance_prompt="an icon in ICONST style" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=800 \
  --rank=16 \
  --output_dir="./icon-lora"
```

Mac の MPS でも回るが、遅ければ時間貸し GPU を 1 時間ほど借りると速い。

→ 実装: [`training/scripts/02_train_lora.sh`](../training/scripts/02_train_lora.sh)

### Stage 3: 速さを足して、Mac で質感チェック

既製の高速化 LoRA（LCM-LoRA）と自作スタイル LoRA を重ねて 4 ステップ生成を試す。
**iPhone に載せる前に Mac で仕上がりを確認する**のが大事。

```python
from diffusers import StableDiffusionPipeline, LCMScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "<SD1.5系のベースID>", torch_dtype=torch.float16).to("mps")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")  # 高速化（既製）
pipe.load_lora_weights("./icon-lora", adapter_name="icon")                        # アイコン風（自作）
pipe.set_adapters(["lcm", "icon"], adapter_weights=[1.0, 0.8])

img = pipe("an icon in ICONST style, a coffee cup, flat, minimal",
           num_inference_steps=4, guidance_scale=1.2).images[0]
img.save("out.png")
```

`adapter_weights` の 0.8 がスタイルの効き具合。ここを調整して狙った雰囲気に寄せる。

→ 実装: [`training/scripts/03_preview_lcm.py`](../training/scripts/03_preview_lcm.py)

### Stage 4: LoRA をモデルに焼き込む

```python
pipe.fuse_lora()   # LoRA を UNet に統合。以後は追加パーツ不要の一体型に
```

→ 実装: [`training/scripts/04_fuse_lora.py`](../training/scripts/04_fuse_lora.py)

### Stage 5: Core AI 形式に変換する（Mac）

拡散モデルは中身が 3 つ（テキストエンコーダ / UNet / VAE デコーダ）。毎ステップ回るのは UNet。

```python
import torch
from coreai_torch import TorchConverter, get_decomp_table   # pip install coreai-torch

unet = pipe.unet.eval()
sample = (latent, timestep, encoder_hidden_states)          # UNet の入力例

ep = torch.export.export(unet, args=sample).run_decompositions(get_decomp_table())
program = TorchConverter().add_exported_program(ep).to_coreai()
program.optimize()
program.save("IconUNet.aimodel")
```

圧縮は `coreai_opt.quantization` の Quantizer で INT8 量子化などをかける。
同じ手順をテキストエンコーダと VAE デコーダにも適用し、3 つの `.aimodel` を用意する。

→ 実装: [`training/scripts/05_export_coreai.py`](../training/scripts/05_export_coreai.py)

### Stage 6: iOS アプリに組み込む（Swift）

`import CoreAI` して AIModel を読み込み、入力を NDArray で渡す。
拡散の "数回まわす" 部分は Swift 側で組む。

1. テキストエンコーダを 1 回だけ実行（プロンプト → 埋め込み）
2. UNet を 4 回ループ（各回でノイズ予測 → スケジューラで少し綺麗にする）
3. VAE デコーダで最後に画像へ変換

スケジューラ（LCM）の計算は Swift 側に自前で持たせる。Xcode のモデルビューアで中身を確認でき、
Core AI インストゥルメントで速度のボトルネックを調べられる。

→ 実装: [`ios/IconDiffusionKit`](../ios/IconDiffusionKit)

## 注意点・リスク

- **難所は Stage 5〜6**。単体モデルの変換は明快だが、拡散パイプライン全体
  （3 モデル＋ループ＋トークナイザ）を組むのは手数がかかる
  - 旧 Core ML の `apple/ml-stable-diffusion` が、モデルの分け方と Swift のループの作り方の下敷きになる
  - `apple/coreai-models` はパッケージの組み方の見本になる
- **対応端末が限られる**（Apple Intelligence 対応機が前提）。非対応端末向けの逃げ道を用意しておく
- **速度とメモリ**: モデルを小さく保ち、少ないステップで出せるようにするのが実用化のカギ
- **データセットの権利**: 市販アイコンセットをそのまま学習に使うのはリスクあり。
  クラウド生成で揃える方針はここでも有利
- **MLX の守備範囲**: スタイル LoRA 学習は MLX でも可能だが、SD/SDXL 級は mflux ほど整っておらず
  DIY 寄り。蒸留・QAT・端末向け変換は PyTorch ＋ Core AI ルートが現実的

## 参考

- WWDC26 セッション「Meet Core AI」(324)
- WWDC26 セッション「Dive into Core AI model authoring and optimization」(325)
- `apple/coreai-torch` / `apple/coreai-optimization` / `apple/coreai-models`
- 下敷き: `apple/ml-stable-diffusion`（旧 Core ML 版の分割方針と Swift ループ）

## 次のアクション

- [ ] Stage 1〜3 を通して「4 ステップでアイコンらしく出るか」を確認する（最初の山。半日〜1 日）
- [ ] ベースモデルの選定（SD-Turbo か SD1.5＋LCM-LoRA）
- [ ] クラウド AI でのアイコンデータセット生成（スタイル統一・トリガーワード決め）
- [ ] Mac 側スクリプト一式の整備（データ整形・学習・LCM 検証・変換）
