"""Stage 1〜5 のスクリプトが共有するロジック。

各スクリプトは `configs/icon_lora.toml` を単一の情報源として読む。
CLI 引数を渡した場合だけ設定を上書きする。
"""

from .config import Config, load_config
from .paths import TRAINING_ROOT, resolve

__all__ = ["Config", "load_config", "TRAINING_ROOT", "resolve"]
