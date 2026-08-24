#!/usr/bin/env bash
#
# Stage 5 の出力をアプリのバンドルに置く。
#
#   ./sync-resources.sh                  # 既定は Stage 5 の出力先（configs の export.output_dir）
#   ./sync-resources.sh coreai-int8      # 別の版を試したいとき
#
# **既定は configs/icon_lora.toml の export.output_dir と揃えてある。** 以前は既定が
# coreai-int8 で設定と食い違っており、Stage 5 を流し直しても古い成果物が端末に入る状態だった。
# 実際、古い fp32 の版を実機の版と取り違えて計測する事故が起きている。
#
# Resources/ の中身は生成物なのでリポジトリに入れない（.gitignore 済み）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VARIANT="${1:-coreai}"
MODELS="$REPO/training/outputs/$VARIANT"
KIT_RES="$REPO/ios/IconDiffusionKit/Sources/IconDiffusionKit/Resources"
DEST="$SCRIPT_DIR/Models"

for name in IconTextEncoder.aimodel IconUNet.aimodel IconVAEDecoder.aimodel; do
  if [[ ! -d "$MODELS/$name" ]]; then
    echo "エラー: $MODELS/$name がありません" >&2
    echo "  先に Stage 5 を通してください:" >&2
    echo "  .venv-coreai/bin/python scripts/05_export_coreai.py" >&2
    exit 1
  fi
done
for name in vocab.json merges.txt pipeline_config.json; do
  if [[ ! -f "$KIT_RES/$name" ]]; then
    echo "エラー: $KIT_RES/$name がありません（Stage 5 が書き出す）" >&2
    exit 1
  fi
done

mkdir -p "$DEST"
for name in IconTextEncoder.aimodel IconUNet.aimodel IconVAEDecoder.aimodel; do
  rm -rf "${DEST:?}/$name"
  cp -R "$MODELS/$name" "$DEST/$name"
  # 日付も出す。outputs/ には過去の版が残っていることがあり、古いものを掴んでいても
  # サイズだけでは気づけない
  echo "  $name ($(du -sh "$DEST/$name" | cut -f1), $(date -r "$MODELS/$name" '+%m/%d %H:%M'))"
done
for name in vocab.json merges.txt pipeline_config.json; do
  cp "$KIT_RES/$name" "$DEST/$name"
  echo "  $name"
done
echo "[sync] $VARIANT -> $DEST"
