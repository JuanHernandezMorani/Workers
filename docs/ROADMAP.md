# Workers research roadmap

## V0 — mechanism (implemented)
Discrete tool vocabulary, deterministic renderer, procedural teacher, shared batched policy, imitation then actor-critic, action history and checkpoints, deterministic projection.

## V0.1 — rigorous learning benchmark
Disjoint shape/combination splits; several seeds; matched-compute imitation-only, RL-only and hybrid comparisons; monitor early stopping, entropy and retention. Establish benchmark on target CPU before scaling. Add checkpoint resumption and replay renderer. Gate: reproducible metrics and no data leakage.

## V1 — Artist
Parameterized finite segments, curves, brush width, color, erase, layers, undo and STOP; render from an action log. Curriculum from primitive construction to icons; expert trajectories generated procedurally. Separate style evaluation and fidelity, and human approval. Gate: improve on deterministic baseline for held-out shapes under a fixed action budget.

## V2 — Perspective
Shared 3D representation; calibrated yaw, pitch, roll, focal and orthographic views; visibility, depth, normals, correspondence and occlusion. Distinguish observed from inferred surfaces. Use synthetic 3D shapes and approved reference assets only. Gate: held-out objects and camera views, cross-view reprojection and silhouette metrics. Never treat inferred images as measured geometry.

## V3 — Sculptor
Adapter for LowPoly Builder RAW mesh + calibrated images. Local operations smooth, flatten, move, edge collapse/split/flip, transactional undo. Deterministic guards for topology and feature preservation; reward multiview fidelity, geometric error, triangle budget and later artistic preference. Train on clean/degraded pairs, including actual reconstruction failures. Gate: beat existing deterministic refinement on held-out assets at matched triangle budget; no invalid exports. Do not touch the original repository before explicit integration.

## Engineering decisions
CPU-first, vectorized environments and cheap intermediate rewards. Separate training from inference; evaluate on disjoint assets; export small policies and action logs. Later choose native acceleration based on profiling, not assumed speed. GPU support, GUI, professional quality and shared artist/sculptor representation remain unvalidated.
