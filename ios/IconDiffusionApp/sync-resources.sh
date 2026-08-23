#!/usr/bin/env bash
#
# Stage 5 の出力をアプリのバンドルに置く。
#
#   ./sync-resources.sh            # 既定は int8 版
#   ./sync-resources.sh coreai     # fp32 版を使いたいとき
#
# Resources/ の中身は生成物なのでリポジトリに入れない（.gitignore 済み）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
VARIANT="${1:-coreai-int8}"
MODELS="$REPO/training/outputs/$VARIANT"
KIT_RES="$REPO/ios/IconDiffusionKit/Sources/IconDiffusionKit/Resources"
DEST="$SCRIPT_DIR/Resources"

for name in IconTextEncoder.aimodel IconUNet.aimodel IconVAEDecoder.aimodel; do
  if [[ ! -d "$MODELS/$name" ]]; then
    echo "エラー: $MODELS/$name がありません" >&2
    echo "  先に Stage 5 を通してください:" >&2
    echo "  .venv-coreai/bin/python scripts/05_export_coreai.py --quantize int8" >&2
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
  echo "  $name ($(du -sh "$DEST/$name" | cut -f1))"
done
for name in vocab.json merges.txt pipeline_config.json; do
  cp "$KIT_RES/$name" "$DEST/$name"
  echo "  $name"
done
echo "[sync] $VARIANT -> $DEST"
