# SA1 Sim-to-Real V1 Training Plan

## Decision

The failed real-robot deployment of `w1c10_k8_e2e_1280.ts` is not a reason to
continue W1 or to warm-start another short bridge. The new training lineage
starts from random SA1 model and optimizer state.

The known 200 ms command dead time is modeled from SA1. Unmeasured actuator
scale and motor lag are not guessed:

```text
d_e ~ DiscreteUniform{0, 1, 2}
u_delay[e,t] = u[e,t-d_e]

velocity_scale = (1, 1)
alpha_v = alpha_omega = 1
```

At `control_dt=0.2 s`, the delay support is exactly 0/200/400 ms. Delay is
sampled per environment at reset and remains fixed within the episode.

After vehicle step-response system identification, channel-specific lag is:

```text
v_t     = alpha_v     * u_v[t-d] + (1-alpha_v)     * v_{t-1}
omega_t = alpha_omega * u_w[t-d] + (1-alpha_omega) * omega_{t-1}
```

`alpha_v` and `alpha_omega` must be measured independently. The legacy scalar
`alpha=0.3` corresponds to a roughly 1.29 s 90% rise time at 5 Hz and must not
be inferred from the 200 ms transport dead time.

## Config Matrix

| Arm | Config | Scenes | Delay | LiDAR | Scale / lag | Purpose |
|---|---|---|---|---|---|---|
| Scene control | `e2e_sa1_k8_obb_scene_mix` | 78/12/10 | off | clean | off | Scene-distribution control |
| Delay control | `e2e_sa1_k8_obb_actuator_delay_only` | 78/12/10 | U{0,1,2} | clean | neutral | Isolate command delay |
| Mainline | `e2e_sa1_k8_obb_sim2real_v1` | 78/12/10 | U{0,1,2} | measured `full` | neutral | Deployment-oriented bundle |

All three new-lineage arms use disjoint reset classes:

- 78% native SA1 scenes
- 12% fixed 1.2-1.4 m narrow passages, with 10.0 m barrier segments
- 10% 4 m x 10 m deployment corridors

SA1 uses a 20 m x 20 m room. The historical 9.0 m narrow-passage barrier
from the smaller later-stage rooms leaves a bypass at the extreme gap
centers. The SA1 config therefore locks 10.0 m segments; the geometry
contract checks both gap-center endpoints and both width endpoints.

The corridor arm uses deployment `wander` kinematics, env-stratified motion
families and the audited 25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D /
5% 5S5D density mix. `long_corridor_gate_aligned_share` stays zero because
the paired SA7.1 recipe failed. No SA7 policy, optimizer or teacher is reused.

SA1 native scenes activate their original two obstacle slots. The scheduler
reserves ten tensor/asset slots so corridor resets can install up to 5S5D, but
reserved slots remain inactive on native resets. Capacity and active count are
separate invariants; a stage that exceeds its reserved capacity fails closed.

The measured VLP-16 `full` preset is fixed, not an arbitrary wide DR range:

- Gaussian range sigma: 0.008672 m
- measured per-ring bias: enabled
- hole/dropout rate: 0.194859
- mixed-pixel distractor rate: 0.002515
- provisional human-material noise: disabled

Physics DR, external pushes and observation delay remain disabled. This keeps
the new factors narrow enough to diagnose.

## Training-Side Contract

The actuator pipeline is:

```text
policy indices
  -> 19x19 decoder
  -> reverse bound + linear/angular slew
  -> issued-command history (83D observation)
  -> per-episode command delay
  -> velocity scale
  -> channel-specific motor lag
  -> simulator
```

The 83D state carries the two previous post-decode/post-slew policy controls
`[linear_accel, omega]`. For `d<=2`, these expose the recent control intent
that produced the pending velocity commands. This reduces delay-induced
partial observability, but it does not expose the per-episode random delay
`d` itself and therefore is not claimed to be an exact Markov-state proof.

For this fresh lineage only, each history pair is encoded as:

```text
h_t = [a_issued / 0.5, omega_issued / 1.2]
history_t = [h_(t-1), h_(t-2)]
```

The legacy function defaults (`0.2`, `pi/15`) saturated much of the 1.2 rad/s
angular range at the final `[-2,2]` clamp. Existing checkpoints retain those
legacy divisors; the new checkpoint metadata explicitly carries `0.5` and
`1.2`, and play/eval inherits them before creating the environment.

The current mainline is not recurrent despite the historical trainer filename
and `rnn_type` metadata. With `end_to_end_frame_stack=true` and
`aux_profile=none`, policy/value consume the current 83D observation plus a
K=8 LiDAR CNN feature. The RNN hidden state is bypassed and all RNN/aux
learning rates are zero.

Training decoder constants currently observed by the wiring smoke:

```text
v_max_forward         = 1.0 m/s
reverse_velocity_scale= 0.2
v_min_reverse         = -0.2 m/s
omega_max             = 1.2 rad/s
linear_accel_max      = 0.5 m/s^2
angular_accel_max     = 3.0 rad/s^2
control_dt            = 0.2 s
action table          = MultiDiscrete([19, 19])
```

## Verification Completed

- New per-channel lag supports `(alpha_v, alpha_omega)`.
- Legacy scalar lag remains behavior-compatible.
- Config to args to action-term wiring is tested.
- Frozen train/car decoder fixture:
  `docs/freeze/sa1_action_contract_v1.json` (361 zero-state pairs plus
  reverse/slew/sign-flip sequences and source hashes).
- Delay-only differs from clean SA1 only by the intended actuator fields.
- Mainline differs from delay-only only by measured LiDAR noise fields.
- Unit/config suites: 347 passed + 27 subtests.
- Corridor event suites: 75 passed + 995 subtests.
- 256-env full scene-mix wiring smoke: 2/2 iterations, 256 steps,
  checkpoint produced and marked non-deployable.
- Smoke runtime markers:
  - `delay=(0,2)`
  - `vel_scale=(1.0,1.0)`
  - `motor_lag alpha=1.0`
  - VLP-16 `full`
  - policy and critic observation shape 83D
  - issued-action history divisors `0.5` and `1.2`
  - scheduler `active=2 capacity=10`
  - corridor 19/19 constructively solvable, including 14/14 random walks
  - narrow passage 28/28 constructively solvable

The smoke is wiring evidence only. Its learning score is meaningless because
the policy started from random weights and ran for two iterations.

Isaac emitted an existing headless viewport-extension startup error and a
PhysX filter-pattern message (`expected 256, found 25600`). Neither stopped
the run. They remain recorded instead of being mislabeled as a completely
error-free log; the training path itself completed without a Python traceback,
NaN or OOM.

## Blockers Before Long Training

1. Vehicle decoder, export metadata and training decoder must pass an exhaustive
   19x19 action parity test.
2. The vehicle reverse bound must be fixed in the decoder, not clamped only
   after history has already recorded an invalid target.
3. The vehicle must define the same issued-command history layer as training.
4. TF/pose jumps must fail closed before a policy receives the corrupted goal.
5. Initial vehicle validation must force reverse off and VO/shield on.

The delay-only arm can be trained without motor sysid. Channel-specific lag and
non-neutral velocity scale remain blocked until `alpha_v`, `alpha_omega` and
tracking scale are measured.

## Gates

For every candidate checkpoint:

1. True-clean actuator-off regression.
2. Fixed `d=0`, `d=1` and `d=2` evaluations, reported separately.
3. Stage gates for SR/CR/TO; do not average delay conditions together.
4. Stability audit: turn sign-flip rate, full-steer fraction, omega RMS,
   dominant frequency, same-sign turn duration and reverse fraction.
5. Python checkpoint versus TorchScript parity on the same observation
   sequence.
6. Vehicle rollout order: motor-disabled replay, lifted wheels, low-speed open
   space with reverse off and VO on, then bounded obstacle tests.

All new-lineage fixed-delay gate commands must include:

```text
--actuator-profile sa1_delay_only
```

Without that flag, the gate intentionally defaults to the historical `bridge`
profile and would reintroduce `U(0.9,1.1)` velocity scaling plus scalar
`alpha=0.3`, invalidating the new-lineage evaluation.

No checkpoint becomes deployable solely because training completed.
