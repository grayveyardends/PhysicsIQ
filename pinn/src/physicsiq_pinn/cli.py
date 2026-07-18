# Copyright (c) 2026 Gabrial Alex. MIT license, see LICENSE.
"""cli.py — the entry point FreeCAD invokes (via tools/pinn_runner.py):

    ~/.venvs/usage/bin/python -m physicsiq_pinn.cli \
        --step runs/job1/part.step --bcs runs/job1/bcs.json \
        --out runs/job1 [--epochs 3000]

Pipeline: sample geometry (freecadcmd subprocess) -> assign BCs -> build
the 3D elasticity problem -> train -> von Mises -> weak-point report ->
result.json + field.npz (+ plots). Every stage prints progress lines;
the workbench streams them into the chat panel live.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description="break-point elasticity PINN")
    ap.add_argument("--step", required=True, help="STEP file of the part")
    ap.add_argument("--bcs", required=True, help="bcs.json (see bc_extract)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--n-interior", type=int, default=2000)
    ap.add_argument("--n-boundary", type=int, default=1500)
    ap.add_argument("--snapshot-every", type=int, default=500,
                    help="stream a live stress field every N epochs (0=off)")
    ap.add_argument("--patience", type=int, default=3000,
                    help="early-stop window in epochs (0=off)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Import here, not at module top: `--help` should not need JAX loaded.
    from physicsiq_pinn.geometry import bc_extract, sampler
    from physicsiq_pinn.postproc import breakpoint
    from physicsiq_pinn.problems.elasticity3d import Elasticity3D
    from physicsiq_pinn.training import trainer
    from physicsiq_pinn import report
    from physicsiq_pinn.viz import plot

    print("stage 1/4: sampling geometry from STEP…", flush=True)
    npz_path = os.path.join(args.out, "points.npz")
    data, faces = sampler.sample_step(
        args.step, npz_path, args.n_interior, args.n_boundary)

    print("stage 2/4: assigning boundary conditions…", flush=True)
    with open(args.bcs, encoding="utf-8") as fh:
        spec = json.load(fh)
    groups, material = bc_extract.build_groups(data, faces, spec)
    beam = bc_extract.beam_prior(data["interior"], faces, groups)

    print("stage 3/4: training PINN (first step compiles, be patient)…",
          flush=True)
    problem = Elasticity3D(data["interior"], groups, material,
                           sigma_ref_mpa=beam["sigma_est"], beam=beam)

    pts = problem.eval_points()
    live_path = os.path.join(args.out, "field_live.npz")

    def snapshot(epoch, p):
        # cheap: one forward pass over the eval points. Written atomically
        # (tmp + rename) so the FreeCAD side never reads a half file.
        import numpy as np
        vm_now = np.asarray(problem.von_mises(p, pts))
        tmp = live_path + ".tmp.npz"
        np.savez(tmp, points=np.asarray(pts, dtype=np.float32),
                 von_mises=vm_now.astype(np.float32))
        os.replace(tmp, live_path)
        print(f"LIVE_FIELD {live_path} epoch={epoch}", flush=True)

    config = trainer.TrainConfig(epochs=args.epochs,
                                 patience=args.patience,
                                 snapshot_every=args.snapshot_every,
                                 on_snapshot=snapshot)
    params, history = trainer.train(problem, config)
    trainer.save_params(params, os.path.join(args.out, "params.pkl"))

    print("stage 4/4: stress field and weak points…", flush=True)
    vm = problem.von_mises(params, pts)
    bp = breakpoint.rank_weak_points(pts, vm, material.yield_mpa)

    plot.scatter_field(pts, vm, os.path.join(args.out, "von_mises.png"))
    plot.loss_curves(history, os.path.join(args.out, "loss.png"))
    report.write(args.out, pts, vm, bp,
                 meta={"epochs": args.epochs, "step_file": args.step})


if __name__ == "__main__":
    main()
