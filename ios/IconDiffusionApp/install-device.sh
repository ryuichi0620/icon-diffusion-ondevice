#!/usr/bin/env bash
#
# 接続中の iPhone にビルドしてインストールする。
#
#   ./install-device.sh
#
# 前提:
#   - 端末が **iOS 27 以降**（Core AI が iOS 27 からのため）
#   - 端末を USB で繋ぎ、この Mac を信頼済みにしてある
#   - ./sync-resources.sh を実行済み（Models/ に .aimodel が入っている）
#
# アプリは約 1GB ある。転送に数分かかる。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

find_xcode() {
  for app in "${XCODE_APP:-}" /Applications/Xcode-beta.app /Applications/Xcode*.app; do
    [[ -d "$app" ]] || continue
    if ls -d "$app"/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS2[7-9]*.sdk >/dev/null 2>&1; then
      echo "$app"; return 0
    fi
  done
  return 1
}

if ! XCODE=$(find_xcode); then
  echo "iOS 27 以降の SDK を持つ Xcode が見つかりません。Core AI は iOS 27 からです。" >&2
  exit 1
fi
XCRUN="$XCODE/Contents/Developer/usr/bin/xcrun"
XCODEBUILD="$XCODE/Contents/Developer/usr/bin/xcodebuild"

if [[ ! -d Models ]] || [[ -z "$(ls -A Models 2>/dev/null | grep -v '^\.gitkeep$')" ]]; then
  echo "Models/ が空です。先に ./sync-resources.sh を実行してください。" >&2
  exit 1
fi

DEVICE_ID="${1:-}"
if [[ -z "$DEVICE_ID" ]]; then
  echo "[install] 接続中の端末を探しています..."
  DEVICE_ID=$("$XCRUN" devicectl list devices 2>/dev/null \
    | awk '/connected/ && /iPhone|iPad/ {print $(NF-1); exit}')
  if [[ -z "$DEVICE_ID" ]]; then
    echo "端末が見つかりません。USB で繋ぎ、この Mac を信頼してください。" >&2
    echo "  一覧: $XCRUN devicectl list devices" >&2
    echo "  ID を直接渡すこともできます: ./install-device.sh <device-id>" >&2
    exit 1
  fi
fi
echo "[install] 端末: $DEVICE_ID"

if [[ ! -f IconDiffusionApp.xcodeproj/project.pbxproj ]]; then
  echo "[install] プロジェクトを生成します"
  xcodegen
fi

echo "[install] ビルド中（約 1GB なので数分かかります）"
"$XCODEBUILD" -project IconDiffusionApp.xcodeproj \
  -scheme IconDiffusionApp \
  -configuration Debug \
  -destination "id=$DEVICE_ID" \
  -allowProvisioningUpdates \
  -derivedDataPath build \
  build

APP="build/Build/Products/Debug-iphoneos/IconDiffusionApp.app"
[[ -d "$APP" ]] || { echo "ビルド成果物が見つかりません: $APP" >&2; exit 1; }

echo "[install] 転送中 ($(du -sh "$APP" | cut -f1))"
"$XCRUN" devicectl device install app --device "$DEVICE_ID" "$APP"
echo "[install] 完了。端末のホーム画面に Icon Diffusion があります"
