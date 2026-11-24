# Workers — Lyrel Forge

Research into compact neural artists that operate deterministic drawing/modeling tools. CPU-first; no external model or dataset required for V0.

## V0: executable experiment
Python 3.10+ with PyTorch. Tested with PyTorch 2.10.0 CPU on Python 3.13.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m unittest -q
.venv\Scripts\python workers.py --supervised 20 --rl 500 --out runs/hybrid
```

Linux: use .venv/bin/python instead. The CLI creates supervised.pt, reinforcement.pt, examples.json (target, canvas and action IDs), and report.json. Use unique output directories to preserve runs.

The policy sees a binary reference and canvas, selects from 34 full-length horizontal/vertical/diagonal strokes and STOP at size 16, and receives the change in pixel error. The teacher greedily scores strokes, imitation uses cross entropy, then actor-critic learns from actual sampled actions. Each batch contains independent vectorized environments. Camera projection is deterministic, not a learned perspective model.

## Scope and roadmap
See [experiment evidence](docs/V0.md) and [roadmap](docs/ROADMAP.md). No professional assets, trained 3D sculptor or integration with LowPoly Builder yet. V0 measures mechanics only. No third-party assets or private user data are included.
