# Workers — Lyrel Artist V0

Experimental tool-based 2D artist, CPU-first. The network selects discrete painting operations rather than generating pixels. This release is a reference-conditioned geometric drawing proof of concept, not a professional illustrator.

## Run (Python 3.11+, Windows PowerShell)

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python -m workers.train supervise --updates 200 --batch 32 --out runs/v0
    python -m workers.train reinforce --steps 4096 --chunk 2048 --out runs/v0

To resume the included model in a fresh workspace:

    New-Item -ItemType Directory -Force runs/v0
    Copy-Item artifacts/v0/best.pt runs/v0/latest.pt
    Copy-Item artifacts/v0/best.pt runs/v0/best.pt
    python -m workers.train evaluate --out runs/v0
    python -m workers.train draw --out runs/v0 --input reference.png
    python -m workers.benchmark --checkpoint artifacts/v0/best.pt --samples 512

Use a black-background reference close to the supported palette and primitives. Drawing is 32x32, nearest-neighbor enlarged for preview, with a replayable JSON stroke sequence. Examples show targets on top and model drawings below. No text-to-image, perspective, freeform brushes, human aesthetic judge, or trained Fantasy Lyrel style yet.

## Experience accounting

One equivalent year = 250 practice days x 8 hours x 3600 decisions/hour = 7,200,000 actual environment decisions. A 5,000-year block is 36,000,000,000 decisions. This is a convention for workload, NOT human expertise. Supervised samples do not count as reinforcement actions. Every action including stop counts once per environment. Parallel environments share one policy. A block finishes only when that many transitions really occur.

    python -m workers.train reinforce --blocks 1 --out runs/v0
    python -m workers.train reinforce --blocks 1 --steps 100000 --out runs/v0

The first command is extremely long and should not be run casually; the second caps this invocation to 100,000 actual transitions. Resume from latest.pt; --blocks 1 completes the next boundary, while --years explicitly requests additional equivalent years. Checkpoints contain model, optimizer, RNG and actual counters. Each completed block is archived. Best checkpoint is selected by fixed validation score; latest preserves continuation. Use independent benchmark seed for final comparisons.

## Architecture

- 32x32 RGB canvas; 8x8 action grid, five brush shapes, three inks and erase, plus stop (1,281 discrete actions).
- 86,726 learned parameters, shared convolutional policy and critic.
- Procedural demonstrations, cross-entropy imitation, then on-policy actor-critic with discounted return, entropy bonus and gradient clipping.
- Reward: normalized reduction in squared RGB error minus a per-stroke cost. It measures reconstruction, not artistry.
- A deterministic greedy brush search is a mandatory baseline. V0 is slower and less accurate than greedy on its own simple brush dictionary.

## Evidence from the first Windows CPU run

200 supervised updates, 6,400 examples; 36,864 reinforcement transitions. No 5,000-year block completed. Fixed validation score rose from 0.90214 after imitation to 0.95157 after RL; exact reconstructions from 46.09% to 63.28%. Independent holdout (512 examples each): up to four strokes neural IoU 97.88%, exact 63.67%; greedy IoU 99.92%, exact 95.12%. Up to eight strokes neural IoU 96.04%, exact 36.91%; greedy IoU 99.70%, exact 79.49%. See artifacts/v0/holdout.json and history.jsonl. Do not extrapolate those scores to complex art.

## Next milestones

1. Expand tools to arbitrary coordinates, continuous curves, widths, pressure, palette, compositing and layers. Add replay/render at arbitrary resolutions.
2. Add structured geometry, icons and unseen procedural families; compare imitation, greedy search, RL and hybrid planning under equal budgets.
3. Add orthographic/perspective exercises with camera parameters and shared 3D scene truth.
4. Train artistic preference evaluator on owned approved references, with held-out categories and human ratings. Only then scale long RL runs; stop or revise if quality plateaus.
5. Integrate multiview and mesh refinement as separate workers, not in this initial model.
