# N1 -> Near-field LiDAR -> Corridor Side-gap -> Dynamic Human-gap -> Actuator Bridge

Date: 2026-07-27

Status: future implementation direction approved by the user. N1 is the only
training arm allowed to run first. Later phases remain design specifications
until the preceding gate passes.

## 1. Authoritative decisions

1. Finish N1 first:
   - fixed rigid-wall gaps;
   - width 1.2--1.4 m;
   - per-episode goal distance `U[2.0, 4.0]` m and lateral offset
     `U[-1.5, 1.5]` m;
   - scripted direct-crossing teacher;
   - prove that the policy first traverses the opening and only then turns
     toward an off-axis goal.
2. In parallel, calibrate and then fix the 0.3--0.5 m near-field LiDAR model.
3. Add controlled 4 m corridor side-gap scenes:
   - free surface-to-surface gap 1.0--1.6 m;
   - left/right mirrored;
   - goals behind, front-left, and front-right of the obstacle;
   - rigid/static and human scenarios evaluated separately.
4. Add dynamic-human side-gap scenes:
   - pass only when predicted free space is sufficiently wide and stable;
   - slow/yield when it is shrinking or uncertain;
   - a 1.0 m moving-human gap is not a mandatory-pass task.
5. Add actuator-delay adaptation only after the clean policy passes all gates:
   - `enable_actuator_dr=True`;
   - `obs_delay_steps=(0, 0)` remains fixed;
   - use the existing 4D history of the previous two applied actions.

Do not merge these phases into one run. Each phase changes one causal factor and
must retain the accepted checkpoint from the previous phase as a fallback.

## 2. Ground truth and current mismatch

### 2.1 Robot geometry

Measured robot OBB:

- physical length: 0.70 m;
- physical width including wheels: 0.60 m;
- half length / half width: 0.35 / 0.30 m;
- collision buffer: 0.10 m per side;
- inflated collision footprint: 0.90 x 0.80 m;
- OBB center offset: -0.128 m along the robot forward axis.

The aligned rigid-gap geometric floor is:

`G_geom = 0.60 + 2 * 0.10 = 0.80 m`

At a 1.0 m rigid gap, an aligned robot has 0.10 m modeled clearance per side.
This is physically feasible but not a robust moving-human safety margin.

### 2.2 Near-field LiDAR fact

User-confirmed real-vehicle observation on 2026-07-27:

- a human at approximately 0.30 m from the LiDAR still produces a sparse point
  cloud;
- return density decreases at very close range.

The active simulator instead applies a hard `r_min=0.5`: every ray below 0.5 m
is invalidated and replaced with `r_max`. That makes a close obstacle appear
absent. This is not an acceptable deployment model.

At a centered 1.0 m gap, both surfaces lie at about 0.50 m from the LiDAR. A
5 cm lateral error moves the near surface to 0.45 m, where the current model
removes it completely. K=8 history cannot recover information that the
simulator hard-deletes on every close frame.

## 3. Phase N1: fixed-wall direct crossing

N1 keeps LiDAR, reward, network, PPO, and replay proportions unchanged while
the LiDAR calibration is performed offline. Before launch, its narrow-goal
distribution is corrected from the historical pinned `3.0 m / 0 m` target to
the explicit ranges below. D0/W1/W2 retain the pinned target for reproducibility.

### 3.1 Frozen training values

| Item | Value |
|---|---:|
| Parent | D0 `checkpoint_3840.pt`, model + Adam |
| Environments | 1024 |
| Rollout length | 128 |
| Budget | 10 iterations |
| Checkpoint interval | 2 iterations |
| LR | inherited constant `2e-4` |
| Scene mix | 68% native SA6 / 10% SA5 replay / 12% narrow / 10% corridor |
| Narrow width | `U[1.2, 1.4]` m |
| Initial yaw | `U[-4 deg, +4 deg]` |
| Goal distance behind barrier | `U[2.0, 4.0]` m |
| Goal lateral offset from gap | `U[-1.5, 1.5]` m |
| Old narrow-policy KL | disabled |
| Scripted CE | narrow frames only |
| CE lambda | `0.067` (no-update shadow; pooled target near 0.5 x PPO actor gradient) |
| Rollout override | disabled |
| LiDAR change | none |
| Actuator DR | disabled |
| Observation delay | `(0, 0)` |

### 3.2 N1 hard gates

- Gate5 crossing >= 95%;
- Gate5 direct crossing >= 95%;
- Gate5 CR <= 5%;
- pre-cross lateral excursion p95 <= 1.0 m;
- first-cross time p95 <= 10 s;
- Gate2, lateral corridor, and longitudinal corridor remain PASS;
- random_2d and mixed_iid do not regress by more than 2 percentage points from
  the official D0 baseline.

Every candidate checkpoint faces Gate5. A passing seed-42 candidate must be
replicated with a second effective training seed before the method is promoted.
Run both the historical aligned-goal Gate5 and the randomized-goal Gate5.
Before lambda calibration or N1 training, the scripted teacher itself must pass
the randomized-goal three-seed check with direct crossing >= 99% and CR <= 1%.
Also run the untrained D0 parent on that same gate. Because N1 now combines a
necessary scene-distribution correction with a teacher-loss change, it is a
capability arm rather than a pure one-variable attribution experiment. A causal
claim about imitation requires a separate randomized-goal/no-CE control arm.

Randomized-goal teacher pre-gate completed on 2026-07-27:

- seeds 404/505/606;
- 6,706 / 6,706 episodes reached the goal and crossed directly;
- collisions: 0;
- first-cross-time p95: 4.2 s for all seeds;
- pre-cross lateral excursion p95: 0.158 / 0.160 / 0.160 m.

Shadow calibration then completed on this same distribution. The two valid
no-update iterations measured:

- iter 1: PPO grad `1.05597`, CE grad `15.84770`, lambda50 `0.03332`;
- iter 2: PPO grad `3.16143`, CE grad `15.76935`, lambda50 `0.10024`;
- pooled lambda: `0.5 * sum(PPO grad) / sum(CE grad) = 0.06669`, frozen as
  `0.067`.

Both valid rows had post-update KL about `2.6e-7` and clip fraction `0`. A stale
pre-fix row with KL `0.00873` had performed a PPO update and was excluded.
The shadow path now skips PPO/RNN/aux optimizer steps, logs update counts and
parameter deltas, and archives an existing JSONL before reruns so stale rows
cannot be averaged silently.

The remaining pre-training action is the D0 zero-shot randomized-goal baseline.
N1 training has not begun.

## 4. Phase NF0: real near-field LiDAR calibration

This phase is data collection and sim-free fitting, not RL training.

### 4.1 Measurement grid

Collect at least 200 scans per cell:

- surface range: 0.30, 0.40, 0.50, 0.60, and 0.80 m;
- incidence angle: 0, 30, and 60 degrees where physically meaningful;
- target class: rigid wall and human;
- human material: at least one light and one dark/low-reflectance garment.

Record:

- valid ROI rays / expected ROI rays;
- occupied 72-bin count / expected occupied bins;
- consecutive all-miss run length, including p50/p90/p95;
- range error of retained points;
- reference definition of distance: LiDAR optical center to target surface.

The fitted quantity is empirical keep probability, not an invented Gaussian:

`q_keep(d, angle, material) = valid ROI returns / expected ROI rays`

### 4.2 Proposed implementation

Add a near-field visibility model before the 72-bin `amin` reduction:

1. hard cutoff: provisional 0.25 m, because 0.30 m is observably non-empty;
2. 0.25--0.60 m: per-ray Bernoulli retention from the measured monotone lookup
   table;
3. >= 0.60 m: retain the existing material model;
4. no-return rays become `r_max`, as today;
5. preserve the 72D output and K=8 input shape.

The 0.25 m hard cutoff is provisional. It may be frozen only after the distance
reference and the 0.30 m ROI measurement are audited. Do not guess the dropout
probability from the word "sparse".

For a material-aware run, combine near-field and material visibility without
double counting. The preferred implementation is one empirical table per
material in the near field, transitioning continuously to the existing
far-field model.

### 4.3 Required code controls and tests

Proposed config fields:

- `lidar_near_field_enabled: bool = False`;
- `lidar_hard_min_range_m: float = 0.25`;
- `lidar_near_field_end_m: float = 0.60`;
- measured keep-probability table identifier/version.

Required tests:

- feature disabled is bit-identical to the current path;
- `< hard_min` is always no-return;
- 0.30 m is not deterministically deleted;
- `>= near_field_end` is unchanged;
- dropout occurs per ray before 72-bin reduction;
- policy observation shape remains 83D and K=8 remains compatible;
- runtime logs hard minimum, table version, and measured retention fractions.

Before training, evaluate the accepted N1 checkpoint under:

- current hard-0.5 observation;
- calibrated near-field observation.

This is an observation-distribution audit, not a graduation comparison.

## 5. Phase N2: static side-gap curriculum in a 4 m corridor

Implement this as a new injector. Do not mutate the proven N1 wall-gap
generator or teacher.

### 5.1 Geometry

- corridor free width: 4.0 m;
- corridor length: 10.0 m or longer;
- desired gap is measured surface-to-surface, never center-to-center;
- side mirror: 50% left / 50% right;
- free gap `G`: 1.0--1.6 m;
- every scene passes the OBB constructive solvability check;
- start offset is limited to 50% of the remaining inflated-OBB clearance.

For a circular obstacle of physical radius `r` beside a wall whose inner face
is at `y_wall`, place its center using:

`y_obs = y_wall - side_sign * (r + G)`

The implementation must recompute the actual surface gap and assert it matches
the sampled value.

### 5.2 Scene families

1. `rigid_forced_gap`: only the controlled opening is a valid route. This tests
   alignment and direct passage.
2. `rigid_choice_gap`: a wall-side gap and a wider alternative both exist. This
   tests safe path choice rather than forcing the narrow side.
3. Goal placement, mirrored with the scene:
   - 40% behind the controlled opening;
   - 30% front-left;
   - 30% front-right.

Width distribution within side-gap episodes:

- 25%: `U[1.0, 1.2]` m stress;
- 50%: `U[1.2, 1.4]` m deployment core;
- 25%: `U[1.4, 1.6]` m easy retention.

Width-conditioned initial yaw:

- 1.0--1.1 m: +/-2 degrees;
- 1.1--1.3 m: +/-4 degrees;
- 1.3--1.6 m: +/-8 degrees;
- the constructive OBB validator remains authoritative over these caps.

### 5.3 Training values

Preserve the total 68/10/12/10 mix. Repartition only the existing 12% narrow
quota:

- 8% original fixed-wall narrow replay;
- 4% static corridor side-gap replay.

Initial arm:

- parent: accepted N1 checkpoint + Adam;
- 1024 envs;
- 10 iterations maximum;
- save every 2 iterations;
- LR and PPO unchanged;
- no new reward;
- no actuator DR;
- calibrated near-field observation enabled.

Do not blindly extend after 10 iterations. Sweep gates first.

### 5.4 Static side-gap gates

Separate metrics by width band and scene family:

- completion/SR, wall CR, obstacle CR, and TO;
- direct crossing;
- minimum OBB clearance;
- first-cross time;
- path-length ratio;
- pre-cross lateral excursion and backtrack;
- goal-side choice accuracy for mirrored scenes.

The 1.2--1.6 m deployment bands must pass before the 1.0--1.2 m stress band can
be promoted. A rigid 1.0 m gap is a pass task; it is not evidence that a moving
human 1.0 m gap is safe to enter.

## 6. Phase N3: dynamic wall-human gap

### 6.1 Observability gate before RL

From the policy's actual K=8 LiDAR input, train a frozen offline probe to
classify:

- gap opening / stable / closing;
- left/right wall side;
- privileged teacher decision: pass / approach slowly / yield.

Report accuracy by range and point-retention band. Require held-out decision
accuracy >= 90% before imitation training. If this fails, reward or teacher
labels cannot solve the information deficit.

### 6.2 Scene and motion values

Preserve the total 12% narrow quota:

- 8% original fixed-wall replay;
- 2% static corridor side-gap;
- 2% dynamic wall-human side-gap.

Dynamic-human parameters:

- initial surface gap: `U[1.0, 1.6]` m;
- speed: `U[0.2, 0.6]` m/s;
- 50% closing, 25% approximately stable, 25% opening;
- left/right mirror: 50/50;
- goal positions use the same 40/30/30 distribution as N2;
- future horizon: 1.5 s with 8 samples, matching the existing future-occupancy
  configuration.

### 6.3 Teacher policy

Use a separate privileged, testable state machine:

1. `APPROACH`: approach a waiting/alignment point at <= 0.15 m/s.
2. `YIELD`: if predicted minimum gap over 1.5 s is <= 1.0 m or is rapidly
   shrinking, command stop and hold a stable heading.
3. `CAUTION`: predicted minimum gap in (1.0, 1.2) m; do not commit to the gap.
4. `ALIGN`: when predicted gap is >= 1.2 m and stable/opening, align with the
   throat.
5. `PASS`: cross at <= 0.35 m/s.
6. `EXIT`: after clearing the person, return control toward the goal.

Use hysteresis around the 1.0/1.2 m thresholds so labels do not flip every
frame. Validate the teacher itself across three seeds before distillation.

The goal never overrides `YIELD`. A goal behind or front-right of a person is
not permission to squeeze through a closing 1.0 m gap.

### 6.4 Training and gates

- parent: accepted N2 checkpoint + Adam;
- first arm: 10 iterations, save every 2;
- PPO/reward unchanged; future occupancy remains weight 0.10 / horizon 1.5 s;
- teacher CE applies only to the 2% dynamic side-gap frames;
- no rollout override;
- no actuator DR yet.

Dynamic hard metrics:

- human and wall collision rates reported separately;
- unsafe-gap entry rate when predicted minimum gap <= 1.0 m;
- correct-yield rate for closing cases;
- eventual pass rate for stable/opening gaps >= 1.2 m;
- freeze/TO, path recovery, first-turn direction, and minimum clearance;
- all prior Gate2/Gate5/corridor gates remain mandatory.

For dynamic 1.0 m cases, safe yielding is a success outcome. Do not use raw
crossing rate as the sole metric.

## 7. Final actuator-delay bridge

Run only from a clean checkpoint that passes N1--N3 and all general gates.

### 7.1 Frozen bridge semantics

- `enable_actuator_dr=True`;
- action delay `U{0,1,2}` steps = 0/200/400 ms at dt=0.2 s;
- motor lag alpha = 0.3;
- velocity scale `U[0.9,1.1]`;
- `obs_delay_steps=(0,0)`;
- retain the 4D history of the previous two applied actions; no observation
  dimension change is required.

### 7.2 Initial bounded arm

- same scene mix as accepted N3;
- model + Adam warm start;
- 1024 envs;
- first budget 10 iterations;
- checkpoint every 2 iterations;
- keep the inherited LR for the first single-variable DR arm;
- evaluate fixed delay 0, 1, and 2 steps separately in addition to randomized
  delay.

Do not restore the old 400-iteration blind budget. Extension is allowed only
when deterministic fixed-delay gates show continued improvement without clean
zero-delay regression.

## 8. Global stop rules

- Do not alter N1's LiDAR model while measuring whether direct imitation works.
- Do not silently revert N1 goals to the historical pinned `3.0 m / 0 m`
  target; D0/W1/W2 remain pinned, N1 explicitly uses the randomized ranges.
- Do not use the static direct teacher in a dynamic-human scene.
- Do not define a moving-human 1.0 m gap as mandatory pass.
- Do not enable observation delay.
- Do not add actuator DR before clean narrow/side-gap/dynamic gates pass.
- Do not infer narrow or side-gap ability from aggregate training SR.
- Every skill claim requires deterministic, per-checkpoint, fixed-seed gates.
- Keep the accepted parent checkpoint for every phase; a later arm never
  destroys the last verified policy.

## 9. Proposed implementation map

Build later phases behind default-off flags and permanent no-op tests. The N1
goal ranges are explicit config-only overrides; all pre-N1 configs must retain
`None` and therefore reproduce the historical fixed goal.

### 9.1 Near-field observation

- `rnn_car_modular/experiment_config.py`
  - add default-off near-field fields and the calibration-table version.
- `utils/charge_env_overrides.py`
  - propagate the selected near-field profile into both policy and critic
    `lidar_static` terms;
  - treat near-field visibility as an explicit sensor transfer function, so it
    may be enabled while unrelated far-field noise remains off;
  - log the resolved composition and never silently let `lidar_no_noise` erase
    the requested near-field profile.
- `mdp/observations/obs_functions.py`
  - replace the single hard-min operation with hard-min plus empirical
    near-field retention;
  - apply retention per ray before 72-bin reduction;
  - expose raw audit counters without changing observations.
- Add a sim-free pure helper and tests for table interpolation, range
  boundaries, material transition, deterministic seeded sampling, and the
  disabled-path no-op.

### 9.2 Static corridor side-gap

- Add `corridor_side_gap_geometry.py`
  - pure scene sampler;
  - surface-gap calculation;
  - OBB/yaw solvability;
  - mirror and goal quota allocation.
- Add `corridor_side_gap.py`
  - install wall/obstacle/goal state on the selected disjoint env mask;
  - keep native, previous-stage, original narrow, side-gap, and long-corridor
    masks mutually exclusive;
  - log requested and actual fractions plus unsolvable count.
- Add a separate static side-gap teacher. Do not extend
  `scripted_narrow_teacher.py` with dynamic-human branches.
- Extend `play_rnn_car.py` or a dedicated runner with width-stratified and
  scene-family metrics.

### 9.3 Dynamic wall-human side-gap

- Add `dynamic_side_gap_teacher.py`
  - privileged phase state machine;
  - 1.5 s predicted-gap computation;
  - threshold hysteresis;
  - action-grid snapping;
  - shadow and override-for-teacher-verification modes.
- Reuse `narrow_imitation_loss.py` only for the final labels/loss; keep the
  dynamic teacher and its mask independent from N1's static mask.
- Add an offline K=8 observability-probe dataset/runner before enabling the
  dynamic imitation weight.
- Add gate JSON fields for pass/yield opportunity, unsafe entry, collision
  source, and predicted minimum gap.

### 9.4 Actuator bridge

- Add a dedicated experiment config inheriting the accepted clean config.
- Lock its allowed diff to actuator DR fields and metadata.
- Runtime assert:
  - input remains 83D;
  - action history is enabled;
  - actuator delay is active;
  - observation delay is exactly zero.
- Add deterministic delay-matrix evaluation for 0, 1, and 2 steps.
