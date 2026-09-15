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

# UNet は素の版か残差つき版のどちらかがあればよい。ControlNet を使う構成では
# 残差つきだけを同梱する（残差にゼロを渡すと素の版と完全に一致するため、
# 両方載せると 821MB の重複になる）
REQUIRED=(IconTextEncoder.aimodel IconVAEDecoder.aimodel)
OPTIONAL=(IconControlNet.aimodel IconUNetControlled.aimodel IconUNet.aimodel)
for name in "${REQUIRED[@]}"; do
  if [[ ! -d "$MODELS/$name" ]]; then
    echo "エラー: $MODELS/$name がありません" >&2
    echo "  先に Stage 5 を通してください:" >&2
    echo "  .venv-coreai/bin/python scripts/05_export_coreai.py" >&2
    exit 1
  fi
done
if [[ ! -d "$MODELS/IconUNet.aimodel" && ! -d "$MODELS/IconUNetControlled.aimodel" ]]; then
  echo "エラー: UNet がありません（IconUNet / IconUNetControlled のどちらも無い）" >&2
  exit 1
fi
for name in vocab.json merges.txt pipeline_config.json; do
  if [[ ! -f "$KIT_RES/$name" ]]; then
    echo "エラー: $KIT_RES/$name がありません（Stage 5 が書き出す）" >&2
    exit 1
  fi
done

mkdir -p "$DEST"
# 前回の同期で入った古いモデルを残さない（残差つきへ切り替えたのに素の版が
# 居座ると、バンドルが 821MB 太るだけでなくどちらを使ったか分からなくなる）
for name in "${OPTIONAL[@]}"; do rm -rf "${DEST:?}/$name"; done
for name in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
  [[ -d "$MODELS/$name" ]] || continue
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

# 輪郭の条件づけに使う資産。グリフは Apache-2.0 なので LICENSE も必ず一緒に運ぶ
GLYPHS="$REPO/training/data/glyphs"
if [[ -d "$GLYPHS" ]]; then
  rm -rf "${DEST:?}/Glyphs"
  mkdir -p "$DEST/Glyphs"
  # 端末で gzip を展開する API を書くほどの大きさではない。ここで開いて置く
  gzcat "$GLYPHS/table.tsv.gz" > "$DEST/Glyphs/table.tsv" 2>/dev/null || true
  cp "$GLYPHS/LICENSE" "$GLYPHS/SOURCE.md" "$DEST/Glyphs/" 2>/dev/null || true
  # 条件に渡すのは輪郭なので 256px で足りる（512px のままだと 29MB になる）
  count=0
  for png in "$GLYPHS"/*.png; do
    [[ -f "$png" ]] || continue
    sips -Z 256 "$png" --out "$DEST/Glyphs/$(basename "$png")" >/dev/null 2>&1 && count=$((count+1))
  done
  echo "  Glyphs ($count 件, $(du -sh "$DEST/Glyphs" | cut -f1))"
fi
echo "[sync] ${VARIANT} -> ${DEST}（合計 $(du -sh "$DEST" | cut -f1)）"
