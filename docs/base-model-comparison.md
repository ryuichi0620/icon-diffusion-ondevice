# ベースモデルの比較 — SD1.5＋LCM / FLUX.2 klein 4B / Sana Sprint 0.6B

調べた日: 2026-10-07

Apple の [apple/coreai-models](https://github.com/apple/coreai-models) に FLUX.2 と Sana Sprint の
Core AI 向けレシピがある。今のベース（SD1.5＋LCM-LoRA）から乗り換える価値があるかを比べる。

## 結論

**今は SD1.5＋LCM-LoRA のままにする。FLUX.2 klein 4B は「質で行き詰まったら試す」候補として残す。
Sana Sprint 0.6B は iPhone 非対応なので外す。**

- SD1.5 は **iPhone 17 で実際に動いていて**（バンドル 1.1〜1.4GB、4 枚 21.4 秒）、
  スタイル LoRA・ControlNet・ANE 回避まで積み上がっている
- FLUX.2 klein は iOS 対応だが、4bit でも **端末に載せる量が SD1.5 の 3〜4 倍**になる見込みで、
  Apple の 2GB の目安を超える（推定。実測していない）
- Sana Sprint はレシピの表で **iOS: No**

## 比較表

| | SD1.5 ＋ LCM-LoRA（今） | FLUX.2 klein 4B | Sana Sprint 0.6B |
|---|---|---|---|
| Apple のレシピ | なし（自前の Stage 5） | あり | あり |
| iOS | **動作確認済み**（iPhone 17） | 対応（既定 512px） | **非対応**（macOS のみ） |
| 本体 | UNet 約 0.86B | DiT 4B | DiT 0.6B |
| テキストエンコーダ | CLIP 約 0.12B | Qwen3（4B 級。36 層中 27 層目まで使う） | Gemma-2-2B |
| 端末に載せる量 | 1.1GB（int8・ControlNet なしの実測） | 本体 4bit で約 2GB ＋ エンコーダ（**合計 3〜4GB と推定**） | 本体 1.1GB ＋ エンコーダ 4.9GB ＋ VAE 0.3GB（bf16。圧縮の既定はなし） |
| ステップ数 | 4（LCM-LoRA を重ねる） | 4（もともと蒸留済み） | 1〜4（もともと蒸留済み） |
| 解像度 | 512 | 512（iOS）/ 1024 | 1024 のみ |
| スタイル LoRA | 学習・焼き込み済み | diffusers で学習できる（BFL も手順を出している） | Sprint 向けの LoRA 学習手順は見つからなかった |
| LoRA を重ねたときの弱め合い | **ある**（LCM と打ち消し合う） | ない（高速化 LoRA が不要） | ない |
| ライセンス | CreativeML OpenRAIL-M | Apache 2.0 | Apache 2.0 ＋ Gemma の利用規約 |
| 今の資産の流用 | ControlNet・データセット・評価がそのまま使える | データセット・評価は流用、**ControlNet は作り直し** | 同左 |

「端末に載せる量」の FLUX.2 は README の「transformer は 1 解像度あたり約 2GB」と、
テキストエンコーダの設定（hidden 2560・36 層）から見積もった値で、実際に書き出してはいない。

## 自作 LoRA は Apple のレシピに載せられるか

**載せられる見込み。ただしレシピに数行の手当てが要る。**

- レシピは配布済みの重みを配るものではなく、**HF の重みを `diffusers` の
  `from_pretrained` で読んで書き出す**仕組み（`python/src/coreai_models/diffusion/pipeline.py`）。
  LoRA を焼き込んだ diffusers 形式のフォルダを渡せば、同じ経路で書き出せるはず
- ただし、どの系統として扱うかを **HF の ID の完全一致**で決めている
  （`diffusion/models.py` の `get_pipeline_type`）。`--experimental` を付けても、
  焼き込み済みのローカルパスは「未知のモデル」として弾かれる。そこを通す改変が要る
- 実際に焼き込み済みの重みを通したことはない（未検証）

## 乗り換えを考え直す条件

- SD1.5 の質（形の崩れ・文字の破綻）が、LoRA や ControlNet を詰めても頭打ちになったとき
- LCM との弱め合いが、`style_weight` の調整で吸収できなくなったとき
- FLUX.2 klein を iOS で書き出し、**実機で載るサイズと生成時間を測れた**とき（まずはこれ）

試すなら、最初の一歩は Mac で `uv run coreai.diffusion.export flux2-klein-4b --platform iOS` を実行し、
書き出されたバンドルの大きさを見ること。2GB を大きく超えるなら、その時点で見送ってよい。

## 出典

- [apple/coreai-models](https://github.com/apple/coreai-models) の `models/flux2/README.md`、
  `models/sana/README.md`、`python/src/coreai_models/model_registry.py`
- [Sana Sprint 追加の PR #289](https://github.com/apple/coreai-models/pull/289)
- [FLUX.2-klein-4B（Hugging Face）](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Sana_Sprint_0.6B_1024px_diffusers（Hugging Face）](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers)
- SD1.5 側の数字: [architecture.md](architecture.md)、[experiments.md](experiments.md)
