"""Stage 5 で Core AI に渡す 3 つのモジュール。

拡散パイプラインをそのまま変換することはできないので、推論の形が違う 3 つに割る。
毎ステップ回るのは UNet だけなので、最適化も計測もそこに集中させたい。

torch を import するので、`load_torch_stack()` 経由で遅延読み込みすること。
"""

from __future__ import annotations

import torch


class TextEncoderWrapper(torch.nn.Module):
    """input_ids -> last_hidden_state。int32 で受けて内部で long に上げる。"""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids=input_ids.to(torch.long))[0]


class UNetWrapper(torch.nn.Module):
    """毎ステップ回る本体。ここだけが実行回数 × なので最適化の対象。"""

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


class VAEDecoderWrapper(torch.nn.Module):
    """スケジューラが吐いた潜在をそのまま受け取り、画像にする。

    scaling_factor の割り戻しは**モデルの中で**やる。Swift 側で掛け忘れる事故を防ぐため。
    pipeline_config.json の vae_scaling_applied_in_model が true なのはこの理由。
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(vae.config.scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.scaling_factor).sample


class ControlNetWrapper(torch.nn.Module):
    """輪郭を条件として受け取り、UNet に足す残差を返す。

    **出力はタプル 13 本**（down 12 + mid 1）。Core AI は名前付きの出力を並べるだけなので、
    ここで平らにして 1 本ずつ名前を付けられる形にする。名前を明示しないと
    `convolution_97` のような自動生成名になり、モデルが少し変わると Swift 側が静かに壊れる
    （docs/architecture.md 6.5 節と同じ轍）。
    """

    def __init__(self, controlnet: torch.nn.Module) -> None:
        super().__init__()
        self.controlnet = controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        down, mid = self.controlnet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        return (*down, mid)


class UNetWithResidualsWrapper(torch.nn.Module):
    """ControlNet の残差も受け取る UNet。入力は 3 本 + 13 本 = 16 本。

    残差なしの `UNetWrapper` と**別モデルとして書き出す**。同じモデルに任意個の入力を
    持たせられないうえ、条件を使わない生成では残差なし版のほうが速いので、両方を載せる。
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *residuals: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=list(residuals[:-1]),
            mid_block_additional_residual=residuals[-1],
            return_dict=False,
        )[0]
