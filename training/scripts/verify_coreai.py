#!/usr/bin/env python3
"""書き出した `.aimodel` を Mac で実行し、PyTorch の参照と突き合わせる。

**端末を使わずに「変換が正しいか」だけを切り分けるための道具。**

実機で絵が壊れたとき、原因は「変換が壊れている」「実行環境が壊れている」のどちらかで、
両方を同時に見ていると切り分けられない。ここは前者だけを見る。
**ここが通っても実機で動く保証にはならない。逆にここが落ちれば変換の問題と確定する。**

macOS 27 では `--unit ane` も動く（26.5 では動かなかった）。ただし
**実機の壊れ方は Mac の ANE でも再現しない。** iPhone では VAE の値域が
参照 [-0.78, 1.09] に対し [-13.27, 26.59] と 24 倍に飛んだが、Mac の ANE は
値域が正常なまま誤差が増えるだけで終わる。2026-09-24 の実測（同一 seed）:

    ユニット   TextEncoder  UNetControlled  VAEDecoder
    GPU          0.0075       0.0054         0.0036
    ANE          0.0056       0.0068         0.0481
    CPU          0.0075       0.0054         0.0905

**VAEDecoder だけが fp16 の精度に敏感で、GPU が最も正確。** ANE 固有の問題ではない
（CPU のほうが悪い）。「VAEDecoder だけ GPU で動かす」という既定の判断
（docs/architecture.md 6.7 節）は、実機の症状とは独立にここでも裏付けられる。

    .venv-coreai/bin/python scripts/verify_coreai.py
    .venv-coreai/bin/python scripts/verify_coreai.py --unit ane
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# 判定のしきい値（平均相対誤差）。fp16 で書き出しているので完全一致はしない。
#
# **過去に「正常」と判断した実績（UNet 0.34% / VAE 1.45%）より厳しい値を置いてある。**
# 出荷する経路（GPU）が 0.0036〜0.0075 で余裕を持って通るので、それより緩めると
# 変化に気づけなくなるため。ANE と CPU はこの値では VAE が落ちるが、それは
# 上の表のとおり既知の差であって不具合ではない。理論値ではなく実測から置いた値。
TOLERANCE = {
    "TextEncoder": 0.01,
    "UNetControlled": 0.01,
    "VAEDecoder": 0.01,
}

# 値域がこの倍率を超えてずれていたら、丸め誤差ではなく破綻とみなす。
# ANE で VAE が発散したときは参照の 24 倍に飛んでいた
RANGE_FACTOR = 3.0


def compare(a, b) -> dict:
    """参照と実行結果を突き合わせる。

    **最大誤差だけで判定しない。** 1 画素の外れ値で決まってしまい、fp16 の丸めと
    本物の破綻を区別できない。実際に過去の不具合（VAEDecoder が ANE で発散）を
    捕まえたのは**値域**だった（参照 [-0.78, 1.09] に対し [-13.27, 26.59]）。
    だから平均・最大・値域の 3 つを出して、判定は平均で行う。
    """
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"形が違います: {a.shape} vs {b.shape}")
    scale = np.maximum(np.abs(a), 1.0)
    err = np.abs(a - b) / scale
    return {
        "mean": float(err.mean()),
        "max": float(err.max()),
        "ref_range": (float(a.min()), float(a.max())),
        "got_range": (float(b.min()), float(b.max())),
    }


async def run_model(path: Path, inputs: dict, unit: str = "gpu"):
    """`.aimodel` を 1 回実行する。入力は名前付きの辞書で渡す。

    `NDArray.from_descriptor()` は空の器を作るだけで、`.numpy()` が返すのは
    **ビューではなくコピー**。書き込んでも入力に反映されない。ここを誤ると
    ゼロ入力で推論して「変換が壊れている」と誤判断する（実際にやった）。

    **既定を GPU にしてあるのは、ANE だと Mac では読み込みに失敗するため**
    （macOS 27 の coreai.runtime は既定で ANE を試し、メモリ不足で落ちる:
    "Program load failed — no memory"）。ここで見たいのは変換の正しさなので、
    確実に動くユニットで回す。**ANE 固有の壊れ方はこの道具では見られない。**
    """
    import numpy as np
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    # ComputeUnitKind のメンバーは**プロパティではなくメソッド**。呼ばずに渡すと
    # "Invalid ComputeUnitKind" になる
    kind = {"gpu": ComputeUnitKind.gpu, "cpu": ComputeUnitKind.cpu,
            "ane": ComputeUnitKind.neural_engine}[unit]()
    options = SpecializationOptions.from_preferred_compute_unit_kind(kind)
    model = await AIModel.load(str(path), options)
    # load は async だが load_function は同期。await すると TypeError になる
    fn = model.load_function(model.function_names[0])
    named = {k: NDArray(np.ascontiguousarray(v)) for k, v in inputs.items()}
    result = await fn(named)
    if isinstance(result, dict):
        return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
    return {"output": result.numpy() if hasattr(result, "numpy") else result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="outputs/coreai-cn",
                        help="検証する .aimodel の置き場（既定 outputs/coreai-cn）")
    parser.add_argument("--fused", default="outputs/fused",
                        help="参照にする PyTorch パイプライン（既定 outputs/fused）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unit", default="gpu", choices=["gpu", "cpu", "ane"],
                        help="実行する計算ユニット（既定 gpu。ANE は Mac では読めない）")
    args = parser.parse_args(argv)

    models = (REPO / args.models) if not Path(args.models).is_absolute() else Path(args.models)
    fused = (REPO / args.fused) if not Path(args.fused).is_absolute() else Path(args.fused)
    if not models.is_dir():
        print(f"[verify] 変換結果がありません: {models}", file=sys.stderr)
        return 1
    if not fused.is_dir():
        print(f"[verify] 参照がありません: {fused}", file=sys.stderr)
        print("[verify] 先に Stage 4 を実行してください", file=sys.stderr)
        return 1

    # 重い依存はここまで遅らせる（--help は torch 無しで通る）
    import numpy as np
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"[verify] 参照を読み込み中: {fused}")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(fused), torch_dtype=torch.float32, safety_checker=None,
    )
    pipe.to("cpu")
    generator = torch.Generator("cpu").manual_seed(args.seed)

    results: list[tuple[str, dict, float]] = []

    # --- TextEncoder ---
    ids = pipe.tokenizer("an icon in ICONST style, a coffee cup, flat, minimal",
                         padding="max_length", max_length=77, truncation=True,
                         return_tensors="pt").input_ids
    with torch.no_grad():
        ref_embed = pipe.text_encoder(ids)[0].numpy()
    got = asyncio.run(run_model(models / "IconTextEncoder.aimodel",
                                {"input_ids": ids.numpy().astype(np.int32)}, args.unit))
    results.append(("TextEncoder", compare(ref_embed, next(iter(got.values()))),
                    TOLERANCE["TextEncoder"]))

    # --- UNetControlled（残差ゼロ = 素の UNet と一致するはず）---
    latent = torch.randn(1, 4, 64, 64, generator=generator)
    timestep = torch.tensor([999.0])
    embed = torch.from_numpy(ref_embed)
    with torch.no_grad():
        ref_noise = pipe.unet(latent, timestep, encoder_hidden_states=embed).sample.numpy()

    unet_path = models / "IconUNetControlled.aimodel"
    if unet_path.is_dir():
        import json
        cfg = json.loads((REPO.parent / "ios/IconDiffusionKit/Sources/IconDiffusionKit"
                          "/Resources/pipeline_config.json").read_text())
        names = [o["name"] for o in cfg["control"]["ControlNet"]["outputs"]]
        shapes = {o["name"]: o["shape"] for o in cfg["control"]["ControlNet"]["outputs"]}
        zeros = {n: np.zeros(shapes[n], dtype=np.float16) for n in names}
        inputs = {"sample": latent.numpy().astype(np.float16),
                  "timestep": timestep.numpy().astype(np.float16),
                  "encoder_hidden_states": ref_embed.astype(np.float16), **zeros}
        label = "UNetControlled"
    else:
        unet_path = models / "IconUNet.aimodel"
        inputs = {"sample": latent.numpy().astype(np.float16),
                  "timestep": timestep.numpy().astype(np.float16),
                  "encoder_hidden_states": ref_embed.astype(np.float16)}
        label = "UNet"
    got = asyncio.run(run_model(unet_path, inputs, args.unit))
    results.append((label, compare(ref_noise, next(iter(got.values()))),
                    TOLERANCE["UNetControlled"]))

    # --- VAEDecoder ---
    with torch.no_grad():
        ref_image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample.numpy()
    got = asyncio.run(run_model(models / "IconVAEDecoder.aimodel",
                                {"latent": latent.numpy().astype(np.float16)}, args.unit))
    results.append(("VAEDecoder", compare(ref_image, next(iter(got.values()))),
                    TOLERANCE["VAEDecoder"]))

    print()
    print(f"{'モデル':<16}{'平均誤差':>9}{'最大':>9}{'許容':>7}  {'参照の値域':<22}{'結果の値域':<22}判定")
    ok = True
    for name, c, tol in results:
        spread = max(abs(c["got_range"][0]), abs(c["got_range"][1]))
        ref_spread = max(abs(c["ref_range"][0]), abs(c["ref_range"][1]), 1e-6)
        sane = spread <= ref_spread * RANGE_FACTOR
        passed = c["mean"] <= tol and sane
        ok = ok and passed
        verdict = "OK" if passed else ("値域が破綻" if not sane else "誤差が大きい")
        print(f"{name:<16}{c['mean']:>9.5f}{c['max']:>9.5f}{tol:>7.3f}  "
              f"[{c['ref_range'][0]:7.2f},{c['ref_range'][1]:7.2f}]      "
              f"[{c['got_range'][0]:7.2f},{c['got_range'][1]:7.2f}]      {verdict}")
    print()
    if ok:
        print("[verify] 変換は参照と一致しています。")
        print("[verify] **ただし ANE 固有の壊れ方はここでは再現しません。**")
        print("[verify] 実機で動くかは端末で確かめること（docs/architecture.md 6.7 節）。")
    else:
        print("[verify] 一致しません。変換側に問題があります。", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
