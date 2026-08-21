"""`configs/icon_lora.toml` の読み込み。

Python 3.11 の tomllib を使うので追加依存は無い。Stage 1（データ整形）だけなら
torch を入れなくてもこのモジュールは動く。

シェルスクリプト（02_train_lora.sh）から使えるよう、`--shell` で
`KEY=value` 形式を吐くエントリポイントも持たせてある。設定の二重管理を避けるため。
"""

from __future__ import annotations

import argparse
import shlex
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import DEFAULT_CONFIG_PATH, resolve


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    lcm_lora: str
    revision: str | None = None


@dataclass(frozen=True)
class StyleConfig:
    trigger_word: str
    prompt_template: str
    fallback_subject: str
    negative_prompt: str = ""

    def prompt(self, subject: str | None = None) -> str:
        """被写体を埋めた生成／学習プロンプトを組み立てる。"""
        return self.prompt_template.format(
            trigger=self.trigger_word,
            subject=(subject or self.fallback_subject).strip(),
        )

    @property
    def instance_prompt(self) -> str:
        """DreamBooth-LoRA の --instance_prompt。被写体を含まないスタイルのみの文。"""
        return f"an icon in {self.trigger_word} style"


@dataclass(frozen=True)
class DatasetConfig:
    input_dir: Path
    output_dir: Path
    resolution: int = 512
    background: str = "#FFFFFF"

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.jsonl"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "report.json"


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    rank: int = 16
    learning_rate: float = 1e-4
    max_train_steps: int = 800
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    mixed_precision: str = "no"
    checkpointing_steps: int = 200


@dataclass(frozen=True)
class PreviewConfig:
    output_dir: Path
    steps: int = 4
    guidance_scale: float = 1.2
    style_weight: float = 0.8
    lcm_weight: float = 1.0
    seed: int = 42
    subjects: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FuseConfig:
    output_dir: Path


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path
    swift_resources_dir: Path
    quantize: str = "int8"


@dataclass(frozen=True)
class Config:
    path: Path
    model: ModelConfig
    style: StyleConfig
    dataset: DatasetConfig
    train: TrainConfig
    preview: PreviewConfig
    fuse: FuseConfig
    export: ExportConfig


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    model_raw = raw.get("model", {})
    style_raw = raw.get("style", {})
    dataset_raw = raw.get("dataset", {})
    train_raw = raw.get("train", {})
    preview_raw = raw.get("preview", {})
    fuse_raw = raw.get("fuse", {})
    export_raw = raw.get("export", {})

    revision = model_raw.get("revision") or None

    return Config(
        path=config_path.resolve(),
        model=ModelConfig(
            base_model=model_raw["base_model"],
            lcm_lora=model_raw["lcm_lora"],
            revision=revision,
        ),
        style=StyleConfig(
            trigger_word=style_raw["trigger_word"],
            prompt_template=style_raw["prompt_template"],
            fallback_subject=style_raw.get("fallback_subject", "a simple object"),
            negative_prompt=style_raw.get("negative_prompt", ""),
        ),
        dataset=DatasetConfig(
            input_dir=resolve(dataset_raw.get("input_dir", "data/raw")),
            output_dir=resolve(dataset_raw.get("output_dir", "data/processed")),
            resolution=int(dataset_raw.get("resolution", 512)),
            background=dataset_raw.get("background", "#FFFFFF"),
        ),
        train=TrainConfig(
            output_dir=resolve(train_raw.get("output_dir", "outputs/icon-lora")),
            rank=int(train_raw.get("rank", 16)),
            learning_rate=float(train_raw.get("learning_rate", 1e-4)),
            max_train_steps=int(train_raw.get("max_train_steps", 800)),
            train_batch_size=int(train_raw.get("train_batch_size", 1)),
            gradient_accumulation_steps=int(train_raw.get("gradient_accumulation_steps", 1)),
            seed=int(train_raw.get("seed", 42)),
            mixed_precision=str(train_raw.get("mixed_precision", "no")),
            checkpointing_steps=int(train_raw.get("checkpointing_steps", 200)),
        ),
        preview=PreviewConfig(
            output_dir=resolve(preview_raw.get("output_dir", "outputs/preview")),
            steps=int(preview_raw.get("steps", 4)),
            guidance_scale=float(preview_raw.get("guidance_scale", 1.2)),
            style_weight=float(preview_raw.get("style_weight", 0.8)),
            lcm_weight=float(preview_raw.get("lcm_weight", 1.0)),
            seed=int(preview_raw.get("seed", 42)),
            subjects=list(preview_raw.get("subjects", [])),
        ),
        fuse=FuseConfig(output_dir=resolve(fuse_raw.get("output_dir", "outputs/fused"))),
        export=ExportConfig(
            output_dir=resolve(export_raw.get("output_dir", "outputs/coreai")),
            swift_resources_dir=resolve(
                export_raw.get(
                    "swift_resources_dir",
                    "../ios/IconDiffusionKit/Sources/IconDiffusionKit/Resources",
                )
            ),
            quantize=str(export_raw.get("quantize", "int8")),
        ),
    )


def _shell_exports(cfg: Config) -> str:
    """02_train_lora.sh が eval する KEY=value 行を組み立てる。"""
    values = {
        "CFG_BASE_MODEL": cfg.model.base_model,
        "CFG_REVISION": cfg.model.revision or "",
        "CFG_INSTANCE_PROMPT": cfg.style.instance_prompt,
        "CFG_TRIGGER_WORD": cfg.style.trigger_word,
        "CFG_DATASET_DIR": str(cfg.dataset.output_dir),
        "CFG_IMAGES_DIR": str(cfg.dataset.images_dir),
        "CFG_RESOLUTION": str(cfg.dataset.resolution),
        "CFG_TRAIN_OUTPUT_DIR": str(cfg.train.output_dir),
        "CFG_RANK": str(cfg.train.rank),
        "CFG_LEARNING_RATE": repr(cfg.train.learning_rate),
        "CFG_MAX_TRAIN_STEPS": str(cfg.train.max_train_steps),
        "CFG_TRAIN_BATCH_SIZE": str(cfg.train.train_batch_size),
        "CFG_GRAD_ACCUM": str(cfg.train.gradient_accumulation_steps),
        "CFG_SEED": str(cfg.train.seed),
        "CFG_MIXED_PRECISION": cfg.train.mixed_precision,
        "CFG_CHECKPOINTING_STEPS": str(cfg.train.checkpointing_steps),
    }
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="設定ファイルの内容を表示する")
    parser.add_argument("--config", default=None, help="設定ファイルのパス")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="シェルから eval できる KEY=value 形式で出力する",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.shell:
        print(_shell_exports(cfg))
        return 0

    print(f"config       : {cfg.path}")
    print(f"base_model   : {cfg.model.base_model}")
    print(f"lcm_lora     : {cfg.model.lcm_lora}")
    print(f"trigger_word : {cfg.style.trigger_word}")
    print(f"instance     : {cfg.style.instance_prompt}")
    print(f"example      : {cfg.style.prompt('a coffee cup')}")
    print(f"dataset in   : {cfg.dataset.input_dir}")
    print(f"dataset out  : {cfg.dataset.output_dir}")
    print(f"lora out     : {cfg.train.output_dir}")
    print(f"fused out    : {cfg.fuse.output_dir}")
    print(f"coreai out   : {cfg.export.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
