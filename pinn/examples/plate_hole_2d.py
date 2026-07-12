"""plate_hole_2d.py — THE correctness check for the whole PINN package.

Trains the Kirsch plate-with-hole problem and prints the stress
concentration factor at the hole edge. Theory says Kt ≈ 3.0 for an
infinite plate (a bit above 3 for our finite one). If this script prints
something in the 2.6–3.4 range, the physics machinery is trustworthy.

Run:  ~/.venvs/usage/bin/python examples/plate_hole_2d.py [epochs]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physicsiq_pinn.problems.elasticity2d import KirschPlate  # noqa: E402
from physicsiq_pinn.training import trainer  # noqa: E402
from physicsiq_pinn.viz import plot  # noqa: E402

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
outdir = os.path.join(os.path.dirname(__file__), "..", "..", "runs",
                      "plate2d")
os.makedirs(outdir, exist_ok=True)

problem = KirschPlate()
config = trainer.TrainConfig(epochs=epochs)
params, history = trainer.train(problem, config)

kt = problem.stress_concentration(params)
print()
print(f"stress concentration factor Kt = {kt:.2f}   (theory: ≈ 3.0)")
verdict = "PASS ✓" if 2.6 <= kt <= 3.4 else "FAIL ✗ (train longer?)"
print(f"verdict: {verdict}")

pts = problem.eval_points()
vm = problem.von_mises(params, pts, applied_stress_mpa=100.0)
plot.scatter_field(pts, vm, os.path.join(outdir, "von_mises.png"),
                   title="von Mises (MPa), S=100 MPa applied")
plot.loss_curves(history, os.path.join(outdir, "loss.png"))
trainer.save_params(params, os.path.join(outdir, "params.pkl"))
