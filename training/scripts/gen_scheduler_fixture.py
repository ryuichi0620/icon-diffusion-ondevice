#!/usr/bin/env python3
"""Swift の LCM スケジューラテストが突き合わせる期待値を diffusers から作り直す。

    python scripts/gen_scheduler_fixture.py

**なぜこのスクリプトが要るか。** 以前この期待値は手作業で作られていて、
リポジトリに再生成手段が無かった。その結果、diffusers 側がタイムステップの
選び方を変えた（`[::-skipping_step]` → 等間隔インデックス）ことに気づけず、
Swift だけが古い挙動のまま `[999, 759, 519, 279]` を返していた。
Python 側の検査も定数同士の比較になっていて空振りしていた。

期待値は**必ず diffusers の実装を実際に呼んで**作る。ここを手で書かない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "ios/IconDiffusionKit/Tests/IconDiffusionKitTests/Fixtures/expected_scheduler.json"
)

# Swift 側と揃える必要がある固定入力。値そのものに意味は無いが、変えると期待値も変わる。
STEP_CASE_SAMPLE = [0.5, -0.25, 1.0, 0.125]
STEP_CASE_MODEL_OUTPUT = [0.1, 0.2, -0.3, 0.05]
STEP_CASE_NOISE = [0.3, -0.7, 0.15, -0.2]
STEP_COUNTS = [1, 2, 4, 8]
VERIFICATION_STEPS = 4


def build(config: dict) -> dict:
    import torch
    from diffusers import LCMScheduler
    from diffusers.schedulers import scheduling_lcm

    def make() -> "LCMScheduler":
        return LCMScheduler(
            num_train_timesteps=config["num_train_timesteps"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            beta_schedule="scaled_linear",
            original_inference_steps=config["original_inference_steps"],
            timestep_scaling=config["timestep_scaling"],
        )

    timesteps = {}
    for n in STEP_COUNTS:
        s = make()
        s.set_timesteps(n)
        timesteps[str(n)] = [int(t) for t in s.timesteps]

    base = make()
    base.set_timesteps(VERIFICATION_STEPS)
    ts = [int(t) for t in base.timesteps]

    # diffusers は alphas_cumprod を float32 で持つが、Swift 側は Double で計算する。
    # ここで float32 の値を書くと 8 桁目でずれてテストが落ちる。
    # 05_export_coreai.py と同じく float64 で組み直す。
    betas64 = torch.linspace(
        config["beta_start"] ** 0.5, config["beta_end"] ** 0.5,
        config["num_train_timesteps"], dtype=torch.float64,
    ) ** 2
    alphas_cumprod64 = torch.cumprod(1.0 - betas64, dim=0)
    alphas = {str(t): float(alphas_cumprod64[t]) for t in ts}
    scalings = {}
    for t in ts:
        c_skip, c_out = base.get_scalings_for_boundary_condition_discrete(t)
        scalings[str(t)] = {"c_skip": float(c_skip), "c_out": float(c_out)}

    sample = torch.tensor(STEP_CASE_SAMPLE, dtype=torch.float64)
    model_output = torch.tensor(STEP_CASE_MODEL_OUTPUT, dtype=torch.float64)
    noise = torch.tensor(STEP_CASE_NOISE, dtype=torch.float64)

    # diffusers の step() は内部で randn_tensor を呼ぶ。固定ノイズで再現したいので
    # そこだけ差し替える。step() の計算そのものは diffusers のものを使う。
    original_randn = scheduling_lcm.randn_tensor
    scheduling_lcm.randn_tensor = lambda shape, **kw: noise.to(kw.get("dtype", torch.float64))
    try:
        steps = []
        for index in range(VERIFICATION_STEPS):
            s = make()
            s.set_timesteps(VERIFICATION_STEPS)
            s._step_index = index
            out = s.step(model_output, ts[index], sample, return_dict=True)
            steps.append({
                "index": index,
                "prev_sample": [float(v) for v in out.prev_sample],
                "denoised": [float(v) for v in out.denoised],
            })

        loop = make()
        loop.set_timesteps(VERIFICATION_STEPS)
        latent = sample.clone()
        for t in loop.timesteps:
            latent = loop.step(model_output, t, latent, return_dict=True).prev_sample
        full_loop_final = [float(v) for v in latent]
    finally:
        scheduling_lcm.randn_tensor = original_randn

    return {
        **config,
        "timesteps": timesteps,
        "alphas_cumprod_at": alphas,
        "scalings_at": scalings,
        "step_case": {
            "sample": STEP_CASE_SAMPLE,
            "model_output": STEP_CASE_MODEL_OUTPUT,
            "noise": STEP_CASE_NOISE,
            "timesteps": ts,
            "steps": steps,
        },
        "full_loop_final": full_loop_final,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="書き換えず、いまのフィクスチャと一致するかだけ見る")
    args = parser.parse_args(argv)

    if not FIXTURE.is_file():
        print(f"フィクスチャが見つかりません: {FIXTURE}", file=sys.stderr)
        return 1
    current = json.loads(FIXTURE.read_text(encoding="utf-8"))
    config = {k: current[k] for k in
              ["beta_start", "beta_end", "num_train_timesteps",
               "original_inference_steps", "sigma_data", "timestep_scaling"]}

    built = build(config)
    if built == current:
        print("[fixture] 一致しています")
        return 0
    if args.check:
        print("[fixture] ⚠ ずれています。scripts/gen_scheduler_fixture.py で再生成してください",
              file=sys.stderr)
        old = current.get("timesteps", {}).get("4")
        print(f"          4 ステップ: 現在 {old} -> diffusers {built['timesteps']['4']}", file=sys.stderr)
        return 1

    FIXTURE.write_text(json.dumps(built, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixture] 更新しました: {FIXTURE}")
    print(f"[fixture] 4 ステップ: {built['timesteps']['4']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
