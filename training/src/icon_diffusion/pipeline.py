"""Stage 3〜5 が共有する、パイプラインの組み立てとデバイス選択。

torch / diffusers に依存するので、Stage 1 しか使わない環境では import しないこと。
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import Config


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    # CPU の fp16 は遅いうえに未対応オペが多いので fp32 にする
    return torch.float32 if device.type == "cpu" else torch.float16


def load_base_pipeline(
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
    *,
    model_path: str | Path | None = None,
    lcm_scheduler: bool = True,
):
    """ベース（または fuse 済み）のパイプラインを読み込み、LCM スケジューラを差す。"""
    from diffusers import LCMScheduler, StableDiffusionPipeline

    source = str(model_path) if model_path else cfg.model.base_model
    kwargs: dict = {"torch_dtype": dtype}
    if cfg.model.revision and model_path is None:
        kwargs["revision"] = cfg.model.revision

    pipe = StableDiffusionPipeline.from_pretrained(
        source,
        safety_checker=None,          # ローカル検証用。配布物には載せない
        requires_safety_checker=False,
        **kwargs,
    )
    if lcm_scheduler:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def attach_loras(
    pipe,
    cfg: Config,
    *,
    style_lora_dir: str | Path | None = None,
    style_weight: float = 0.8,
    lcm_weight: float = 1.0,
) -> list[str]:
    """高速化 LoRA（既製）と スタイル LoRA（自作）を重ねる。

    企画メモの Stage 3 と同じ組み立て。戻り値は有効化したアダプタ名。
    """
    adapters: list[str] = []
    weights: list[float] = []

    pipe.load_lora_weights(cfg.model.lcm_lora, adapter_name="lcm")
    adapters.append("lcm")
    weights.append(lcm_weight)

    style_dir = Path(style_lora_dir) if style_lora_dir else cfg.train.output_dir
    if style_dir.is_dir():
        pipe.load_lora_weights(str(style_dir), adapter_name="icon")
        adapters.append("icon")
        weights.append(style_weight)
    else:
        print(
            f"[warn] スタイル LoRA が見つかりません: {style_dir}\n"
            "       高速化 LoRA だけで生成します（Stage 2 未実行？）"
        )

    pipe.set_adapters(adapters, adapter_weights=weights)
    return adapters


def make_generator(seed: int) -> torch.Generator:
    """CPU 上の Generator を使う。MPS/CUDA を跨いでも同じ乱数列になり再現しやすい。"""
    return torch.Generator(device="cpu").manual_seed(seed)


def make_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    *,
    columns: int = 3,
    padding: int = 12,
    label_height: int = 22,
    background: tuple[int, int, int] = (250, 250, 250),
) -> Image.Image:
    """生成結果を 1 枚に並べる。並べて見ないとスタイルの揺れに気づけない。"""
    if not images:
        raise ValueError("画像が空です")

    cell_w, cell_h = images[0].size
    columns = max(1, min(columns, len(images)))
    rows = (len(images) + columns - 1) // columns

    sheet_w = columns * cell_w + (columns + 1) * padding
    sheet_h = rows * (cell_h + label_height) + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_w, sheet_h), background)
    draw = ImageDraw.Draw(sheet)

    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + label_height + padding)
        sheet.paste(image, (x, y))
        if index < len(labels):
            draw.text((x + 2, y + cell_h + 4), labels[index][:60], fill=(60, 60, 60))

    return sheet
