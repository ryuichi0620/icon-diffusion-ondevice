"""Stage 1〜5 のスクリプトが共有するロジック。

各スクリプトは `configs/icon_lora.toml` を単一の情報源として読む。
CLI 引数を渡した場合だけ設定を上書きする。
"""

from .config import Config, load_config
from .paths import TRAINING_ROOT, resolve

__all__ = ["Config", "load_config", "TRAINING_ROOT", "resolve", "load_torch_stack"]


def load_torch_stack():
    """torch / diffusers に依存するモジュールを遅延インポートする。

    モジュール先頭で import すると `--help` や引数チェックまで torch を要求してしまう。
    数 GB の依存を入れる前にオプションを確認できたほうがよいので、実際に使う直前まで遅らせる。

    戻り値は `(torch, icon_diffusion.pipeline)`。
    """
    try:
        import torch

        from . import pipeline
    except ImportError as exc:
        raise SystemExit(
            "torch / diffusers が見つかりません。Stage 2 以降にはこれらが必要です。\n"
            "  cd training && pip install -r requirements.txt\n"
            f"元のエラー: {exc}"
        ) from exc
    return torch, pipeline
