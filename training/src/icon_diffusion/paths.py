"""パス解決。設定ファイル内の相対パスはすべて training/ を基準にする。"""

from __future__ import annotations

from pathlib import Path

# src/icon_diffusion/paths.py -> src/icon_diffusion -> src -> training
TRAINING_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = TRAINING_ROOT / "configs" / "icon_lora.toml"


def resolve(path: str | Path) -> Path:
    """相対パスを training/ 基準の絶対パスにする。絶対パスはそのまま返す。"""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (TRAINING_ROOT / p).resolve()
