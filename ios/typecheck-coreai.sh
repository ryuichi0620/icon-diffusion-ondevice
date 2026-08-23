#!/usr/bin/env bash
#
# CoreAIBackend.swift を実 API に対して型チェックする。
#
#   ./ios/typecheck-coreai.sh
#
# 通常の `swift test` では CoreAIBackend.swift はビルド対象外（ICON_DIFFUSION_COREAI が
# 未定義）なので、**壊れていても気づけない**。Core AI 側を触ったらこれを通すこと。
#
# Core AI は iOS 27 以降。iOS 26 SDK には CoreAI.framework が無いので、
# Xcode 27 以降が必要になる。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES="$SCRIPT_DIR/IconDiffusionKit/Sources/IconDiffusionKit"

# iOS 27 SDK を持つ Xcode を探す。既定の xcrun が 26 系でも beta を拾う。
find_sdk() {
  for app in "${XCODE_APP:-}" /Applications/Xcode-beta.app /Applications/Xcode*.app; do
    [[ -d "$app" ]] || continue
    local sdk
    sdk=$(ls -d "$app"/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS2[7-9]*.sdk 2>/dev/null | head -1)
    if [[ -n "$sdk" ]]; then
      echo "$app|$sdk"
      return 0
    fi
  done
  return 1
}

if ! found=$(find_sdk); then
  echo "iOS 27 以降の SDK を持つ Xcode が見つかりません。" >&2
  echo "Core AI は iOS 27 からなので、Xcode 27 以降が要ります。" >&2
  echo "別の場所にあるなら XCODE_APP=/path/to/Xcode.app を渡してください。" >&2
  exit 1
fi

APP="${found%%|*}"
SDK="${found##*|}"
SWIFTC="$APP/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc"

echo "[typecheck] Xcode: $APP"
echo "[typecheck] SDK  : $(basename "$SDK")"

"$SWIFTC" -typecheck \
  -sdk "$SDK" \
  -target arm64-apple-ios27.0 \
  -DICON_DIFFUSION_COREAI \
  -swift-version 6 \
  $(find "$SOURCES" -name "*.swift")

echo "[typecheck] 通りました（CoreAIBackend.swift を含む）"
echo "[typecheck] ただし通ったのは型チェックまで。実機で動かしたわけではない"
