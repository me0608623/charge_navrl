# SA4-D4 Geometry-Feasible 19x19 Selector

Status: `READY_NOT_RUN` (2026-08-01). This is a fixed-checkpoint diagnostic
upper bound. It is not a training change, SA4 graduation result, SA5 parent, or
deployment shield.

## Question

D3 showed that a maximum turn can reduce lateral pedestrian collisions while
moving failures to static obstacles and walls. D4 asks a narrower question:

> Does selecting only actions that are jointly feasible against predicted
> pedestrian motion and policy-visible static/wall geometry reduce lateral
> obstacle collisions without increasing wall collisions or timeouts?

## Insertion Point

```text
policy action indices
-> optional D4 selector indices
-> process_actions / decode
-> fixed d1 decoded-command queue
-> applied command
-> simulator
```

The selector never writes post-delay commands. For each candidate path, the
first 0.2 s is the decoded command already pending in the d1 queue. The newly
selected command affects only later samples.

## Candidate Geometry

All 19x19 actions are decoded with the frozen action contract. Paths use a
2.4 s horizon sampled every 0.2 s.

For dynamic obstacle j:

```text
p_j(t) = p_j(0) + v_j t
c_dynamic(a) = minimum OBB-to-circle surface clearance over j and t
```

For walls and static obstacles, the current raw 72-bin LiDAR scan is converted
to body-frame surface points. Returns attributable to known dynamic circles
are removed before evaluating swept robot-OBB clearance:

```text
c_static(a) = minimum point-to-OBB surface clearance over LiDAR points and t
feasible(a) = [c_dynamic(a) >= 0.10 m] AND [c_static(a) >= 0.10 m]
```

This is trajectory-intersection geometry rather than the scalar
future-occupancy reward. It checks where the robot and pedestrian will be over
the same horizon and separately vetoes candidates that intersect visible
static geometry.

## Deterministic Selection

1. Preserve the policy action if it is jointly feasible.
2. Otherwise minimize L1 distance from the policy's two action indices.
3. Tie-break by the largest minimum of dynamic/static clearance margins.
4. Then prefer greater forward speed.
5. Resolve any remaining tie by the smallest flattened action index.
6. If no action is jointly feasible, preserve the policy action and mark the
   frame unresolved. Such a frame has no safety guarantee.

The two-frame radial-TTC trigger and two-frame release contract remain the
frozen D3 trigger so the diagnostic changes the action selector, not the event
detector.

## Frozen Identity

- Protocol schema: `sa4_d4_geometry_selector/v1`
- Protocol SHA-256:
  `1729fe3a878b1a1c64f172bb7b3bb0a23f6553385c8b8ca9e982e0cf97170e9d`
- Machine-readable freeze:
  `docs/freeze/sa4_d4_geometry_selector_v1.json`
- Action grid: 19x19
- `v_max=1.0 m/s`, reverse scale `0.2`, `a_max=0.5 m/s^2`
- `omega_max=1.2 rad/s`, `alpha_max=3.0 rad/s^2`
- Robot OBB half extents: `(0.35, 0.30) m`, x offset `-0.128 m`
- Dynamic and static surface clearance: `0.10 m`
- LiDAR: 72 bins, 20 m maximum, returns below 0.40 m ignored

## Fixed Comparison

The ready-not-run runner creates a fresh identity baseline and D4 arm:

```bash
/home/aa/miniconda3/envs/env_isaaclab/bin/python \
  scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa4_d4_geometry_suite.py \
  logs/rnn_car/sa4_r2_futureocc015_from_sa3r1_c100_ne1024_s42_p50_r1/checkpoint_6400.pt \
  --expect-checkpoint-sha256 c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197 \
  --output-dir logs/gates/sa4_d4/geometry_suite_20260801
```

Fixed cell: stage 4 lateral, seed 818, d1=200 ms, 64 environments,
2,500 steps per arm. The runner writes `PROTOCOL.json` before the first GPU
rollout, refuses overwrite, fingerprints sources before/after each arm, and
fails closed on recorder or delay reconciliation errors.

Report SR, obstacle CR, wall CR, total CR, TO, trigger/override counts, and
no-feasible frames. A lower obstacle CR is not an improvement if wall CR or TO
replaces it.

## Limitations

- Dynamic positions, velocities, radii, and dynamic-return attribution are
  privileged simulator information.
- Static geometry consists only of currently visible LiDAR surface samples;
  geometry occluded behind a removed dynamic return cannot be reconstructed.
- Pedestrian motion is constant-velocity over the horizon.
- The experiment is one checkpoint and one evaluator seed.
- Even a positive result would establish a mechanism upper bound, not a
  deployable safety guarantee.

## Verification

- Targeted D3/D4 contracts: `51 passed`.
- Full `rnn_car_wdclean` regression: `817 passed, 27 subtests passed`.
- 64-environment synthetic selector batch: output shapes `[64,2]` and
  `[64,19,19]`; CPU peak RSS approximately 678 MB.
- No Isaac Sim or GPU comparison has been launched.
