#!/usr/bin/env bash
#
# Stage 2: アイコン用スタイル LoRA を学習する（Mac）。
#
#   ./scripts/02_train_lora.sh                  # DreamBooth-LoRA（企画メモどおり・既定）
#   ./scripts/02_train_lora.sh --captions       # Stage 1 の per-image キャプションを使う
#   ./scripts/02_train_lora.sh --dry-run        # 実行するコマンドを表示するだけ
#
# 学習パラメータは configs/icon_lora.toml が単一の情報源。ここには書かない。
#
# DreamBooth と captions のどちらを選ぶか:
#   DreamBooth … 全画像に同じ instance_prompt を当てる。スタイル LoRA の定番。迷ったらこちら
#   captions   … 被写体名入りのキャプションで学習する。被写体とスタイルが分離しやすく、
#                Stage 1 で --captions を用意できているならこちらのほうが締まる
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR_DIR="$TRAINING_ROOT/.vendor"

MODE="dreambooth"
DRY_RUN=0
CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --captions)   MODE="captions"; shift ;;
    --dreambooth) MODE="dreambooth"; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --config)     CONFIG="$2"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            EXTRA_ARGS+=("$1"); shift ;;   # そのまま accelerate に渡す
  esac
done

# ---- 設定を読み込む（configs/icon_lora.toml が単一の情報源）----
CONFIG_ARGS=(--shell)
[[ -n "$CONFIG" ]] && CONFIG_ARGS+=(--config "$CONFIG")
eval "$(PYTHONPATH="$TRAINING_ROOT/src" python3 -c 'import sys; from icon_diffusion.config import main; sys.exit(main())' "${CONFIG_ARGS[@]}")"

# ---- 前提チェック ----
# --dry-run は「実行前にコマンドを確認する」ためのものなので、学習画像が無くても止めない。
# 代わりに警告を出す（本番実行なら止める）。
IMAGE_COUNT=0
if [[ -d "$CFG_IMAGES_DIR" ]]; then
  IMAGE_COUNT=$(find "$CFG_IMAGES_DIR" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
fi
if [[ "$IMAGE_COUNT" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[stage2] ⚠ 学習画像がありません: $CFG_IMAGES_DIR" >&2
    echo "[stage2]   コマンドの確認だけ行います。本番前に Stage 1 を実行してください" >&2
  else
    echo "エラー: 学習画像がありません: $CFG_IMAGES_DIR" >&2
    echo "先に Stage 1 を実行してください: python scripts/01_prepare_dataset.py" >&2
    exit 1
  fi
fi

# --dry-run は「実行せずコマンドを見る」ためのものなので、学習用の依存は要求しない
if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v accelerate >/dev/null 2>&1 \
    || { echo "エラー: accelerate が見つかりません。pip install -r requirements.txt を実行してください" >&2; exit 1; }
fi

# ---- diffusers の学習スクリプトを取得（インストール済みのバージョンに合わせる）----
DIFFUSERS_REF="${DIFFUSERS_REF:-$(python3 -c 'import diffusers; print("v" + diffusers.__version__)' 2>/dev/null || echo main)}"
if [[ "$MODE" == "captions" ]]; then
  SCRIPT_NAME="train_text_to_image_lora.py"
  SCRIPT_PATH_IN_REPO="examples/text_to_image/$SCRIPT_NAME"
else
  SCRIPT_NAME="train_dreambooth_lora.py"
  SCRIPT_PATH_IN_REPO="examples/dreambooth/$SCRIPT_NAME"
fi
TRAIN_SCRIPT="$VENDOR_DIR/$DIFFUSERS_REF/$SCRIPT_NAME"

if [[ ! -f "$TRAIN_SCRIPT" && "$DRY_RUN" -eq 0 ]]; then
  echo "[stage2] diffusers $DIFFUSERS_REF の $SCRIPT_NAME を取得します"
  mkdir -p "$(dirname "$TRAIN_SCRIPT")"
  curl -fsSL -o "$TRAIN_SCRIPT" \
    "https://raw.githubusercontent.com/huggingface/diffusers/$DIFFUSERS_REF/$SCRIPT_PATH_IN_REPO" \
    || { echo "エラー: 学習スクリプトを取得できませんでした。DIFFUSERS_REF=$DIFFUSERS_REF を確認してください" >&2; rm -f "$TRAIN_SCRIPT"; exit 1; }
fi

# ---- 引数を組み立てる ----
ARGS=(
  --pretrained_model_name_or_path="$CFG_BASE_MODEL"
  --output_dir="$CFG_TRAIN_OUTPUT_DIR"
  --resolution="$CFG_RESOLUTION"
  --train_batch_size="$CFG_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps="$CFG_GRAD_ACCUM"
  --learning_rate="$CFG_LEARNING_RATE"
  --lr_scheduler=constant
  --lr_warmup_steps=0
  --max_train_steps="$CFG_MAX_TRAIN_STEPS"
  --rank="$CFG_RANK"
  --seed="$CFG_SEED"
  --checkpointing_steps="$CFG_CHECKPOINTING_STEPS"
  --mixed_precision="$CFG_MIXED_PRECISION"
)
[[ -n "$CFG_REVISION" ]] && ARGS+=(--revision="$CFG_REVISION")

if [[ "$MODE" == "captions" ]]; then
  if [[ ! -f "$CFG_DATASET_DIR/metadata.jsonl" && "$DRY_RUN" -eq 0 ]]; then
    echo "エラー: $CFG_DATASET_DIR/metadata.jsonl がありません（Stage 1 の出力）" >&2
    exit 1
  fi
  ARGS+=(
    --train_data_dir="$CFG_DATASET_DIR"
    --caption_column=text
    --validation_prompt="$CFG_INSTANCE_PROMPT"
  )
else
  ARGS+=(
    --instance_data_dir="$CFG_IMAGES_DIR"
    --instance_prompt="$CFG_INSTANCE_PROMPT"
  )
fi

ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "[stage2] モード    : $MODE ($SCRIPT_NAME @ $DIFFUSERS_REF)"
echo "[stage2] ベース    : $CFG_BASE_MODEL"
echo "[stage2] 学習画像  : $IMAGE_COUNT 枚 <- $CFG_IMAGES_DIR"
echo "[stage2] プロンプト: $CFG_INSTANCE_PROMPT"
echo "[stage2] 出力      : $CFG_TRAIN_OUTPUT_DIR"
echo "[stage2] steps=$CFG_MAX_TRAIN_STEPS rank=$CFG_RANK lr=$CFG_LEARNING_RATE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '\naccelerate launch %q' "$TRAIN_SCRIPT"
  printf ' \\\n  %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

# MPS で未実装オペに当たったときに CPU へ落とす。無いと学習が始まらないことがある。
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p "$CFG_TRAIN_OUTPUT_DIR"
accelerate launch "$TRAIN_SCRIPT" "${ARGS[@]}"

echo "[stage2] 完了 -> $CFG_TRAIN_OUTPUT_DIR"
echo "[stage2] 次: python scripts/03_preview_lcm.py --steps 4"
