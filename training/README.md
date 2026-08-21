# training — Mac 側のパイプライン（Stage 1〜5）

`configs/icon_lora.toml` が全ステージ共有の単一の情報源。スクリプトの CLI 引数は
「その場で試すための上書き」であって、恒久的な設定はすべて TOML に書く。

## セットアップ

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0,'src'); from icon_diffusion.config import main; main()"   # 設定の確認
```

Stage 1 だけなら `pillow` と `numpy` があれば動く。torch は Stage 2 から。

## 各ステージ

### Stage 1 — データ整形

```bash
python scripts/01_prepare_dataset.py --input ~/Downloads/icons_raw
python scripts/01_prepare_dataset.py --input ./raw --captions captions.csv --limit 20
```

透過の合成 → パディングで正方形化 → 512 リサイズ → `metadata.jsonl` 生成まで。
最後に **スタイルの揃い具合を検査して警告を出す**。

`flatness`（上位 8 色が画素の何割を占めるか）が主な指標。フラット塗りのアイコンなら 0.99 前後、
グラデーションや写真が混ざると一気に下がるので、混入した画像をファイル名で名指しできる。
色数そのものは輪郭のアンチエイリアスで形状ごとに大きく変わるため指標にしていない。

出力: `data/processed/{images/, metadata.jsonl, report.json}`

### Stage 2 — スタイル LoRA を学習

```bash
./scripts/02_train_lora.sh --dry-run     # 実行されるコマンドを確認（依存もネットワークも不要）
./scripts/02_train_lora.sh               # DreamBooth-LoRA（既定・企画メモどおり）
./scripts/02_train_lora.sh --captions    # Stage 1 の per-image キャプションを使う
```

diffusers の学習スクリプトを `.vendor/<version>/` に取得して `accelerate launch` する。
**インストール済みの diffusers のバージョンに合わせて**取得するので、
ライブラリと example のズレで落ちることがない。`DIFFUSERS_REF=v0.31.0` で固定もできる。

出力: `outputs/icon-lora/`

### Stage 3 — LCM を重ねて 4 ステップ生成（最初の山）

```bash
python scripts/03_preview_lcm.py
python scripts/03_preview_lcm.py --sweep-style 0.4 0.6 0.8 1.0   # 効き具合の当たりを取る
```

既製の LCM-LoRA と 自作スタイル LoRA を重ねて生成する。
1 枚ずつの PNG に加えて **コンタクトシート**を作る。スタイルの揺れは並べないと気づけない。

`--sweep-style` は被写体を固定して重みだけ振る。`adapter_weights` の当たりを取るのに使う。

出力: `outputs/preview/{*.png, contact_sheet.png, preview.json}`

### Stage 4 — LoRA を焼き込む

```bash
python scripts/04_fuse_lora.py
python scripts/04_fuse_lora.py --device cpu   # fp32 で焼き直す（差が出たときの逃げ道）
```

焼き込みは不可逆で fp16 だと丸め誤差が乗る。既定で**焼く前と後を同じ seed で生成して
差分 (MAE) を出す**ので、絵が変わっていないことを確かめてから Stage 5 に渡せる。

出力: `outputs/fused/`（+ `outputs/fuse_compare/before_fuse.png`, `after_fuse.png`）

### Stage 5 — Core AI 形式へ変換

```bash
python scripts/05_export_coreai.py --check-only   # まず PyTorch のまま形だけ確認
python scripts/05_export_coreai.py
```

`--check-only` は **Core AI をまったく使わずに** 3 つのラッパー（TextEncoder / UNet / VAEDecoder）を
PyTorch で実行し、入出力の形を表示する。Stage 5〜6 が難所なので、変換の前に
モデル分割そのものが正しいことをここで切り分けられる。

同時に Swift 側の資産も書き出す:

| ファイル | 用途 |
| --- | --- |
| `vocab.json` / `merges.txt` | Swift の CLIP トークナイザが読む |
| `pipeline_config.json` | β スケジュール・潜在の形・検証用の既知値 |

出力: `outputs/coreai/Icon{TextEncoder,UNet,VAEDecoder}.aimodel`
＋ `../ios/IconDiffusionKit/Sources/IconDiffusionKit/Resources/`

> **TODO(core-ai)**: `coreai_torch` / `coreai_opt` の API 名は企画メモの記述を写したもの。
> 実物に対して未検証で、量子化（`_apply_quantization`）は未実装のまま警告して素通しする。
> 推測で書いて黙って壊れるより、明示的に空けてある。

## 設定の要点

| キー | 効くステージ | メモ |
| --- | --- | --- |
| `model.base_model` | 2〜5 | **未決**。SD1.5 系ミラー or `stabilityai/sd-turbo` |
| `model.lcm_lora` | 3〜4 | 既製の高速化 LoRA。自作しない |
| `style.trigger_word` | 1〜6 | 一度決めたら変えない。学習時と生成時で不一致だとスタイルが出ない |
| `train.rank` / `max_train_steps` | 2 | 100 枚なら 800〜1200 ステップから |
| `preview.guidance_scale` | 3 | Mac は 1.2、端末は 1.0（CFG なしで UNet 実行が半分） |
| `preview.style_weight` | 3〜4 | スタイルの効き具合。`--sweep-style` で当たりを取る |

## 動作確認できている範囲

- Stage 1 は合成データで通しで実行・検証済み（前処理・キャプション・警告経路）
- Stage 2 は `--dry-run` で両モードのコマンド生成を確認済み
- **Stage 3〜5 は未実行**。torch と実モデルが必要なので Mac 側での確認待ち
