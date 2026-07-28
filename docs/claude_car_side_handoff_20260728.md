# Claude Code Handoff: Car-Side Contract and Safety Repair

Use this as the implementation prompt for the vehicle session.

---

You own only the vehicle/deployment repository. Do not modify IsaacLab training
files and do not deploy or restart a policy until all offline tests pass.

`w1c10_k8_e2e_1280.ts` is permanently classified as REAL-ROBOT FAIL. The W1
bridge c10/c15 and SA7/SA7.1 are also not deployment candidates.

## Confirmed Failure Evidence

- W1 diagnostic: 57.34 s / 1,145 rows.
- `|rl_w| >= 0.83`: 33.7%.
- 13 clear left/right sign flips.
- `map_yaw` jumped about 97.4 degrees in about 0.05 s.
- W1 ran with VO off and reverse enabled.
- Training reverse bound is `-0.2 m/s`; vehicle decoder produced roughly
  `rl_v=-0.557 m/s`, then a later filter clamped `sent_v=-0.2 m/s`.
- Action history recorded a different layer from the command actually sent.

## Required Changes

1. **Make the action decoder exactly match training.**
   - Action table is 19x19.
   - `v_max_forward=1.0 m/s`.
   - `reverse_velocity_scale=0.2`; decoder lower bound is `-0.2 m/s`, not
     `-1.0 m/s`.
   - `omega_max=1.2 rad/s`.
   - `linear_accel_max=0.5 m/s^2`.
   - `angular_accel_max=3.0 rad/s^2`.
   - `control_dt=0.2 s`.
   - Do not leave reverse scaling as a later output-only clamp.

2. **Define one issued-command layer.**
   - The 83D history must contain the two previous post-decode/post-slew
     policy controls `[linear_accel, omega]` used to construct the velocity
     commands entering the actuator transport-delay queue.
   - For the fresh SA1 sim-to-real lineage, encode each
     `[linear_accel, omega]` pair as `[linear_accel/0.5, omega/1.2]`,
     newest to oldest, with final clipping to `[-2,2]`. The old
     `0.2` / `pi/15` divisors belong only to legacy checkpoints.
   - Decoder, history and diagnostics must refer to the same decode invocation;
     diagnostics must separately record the resulting `[cmd_v, cmd_w]`.
   - Never log/store a pre-filter target while publishing a different command.
   - If VO/shield changes the command, record both policy-issued and
     safety-issued values explicitly; do not silently substitute one for the
     other.

3. **Add a fail-closed pose/TF jump guard.**
   - Reject translation/yaw changes that cannot be explained by elapsed time
     and configured physical speed limits.
   - On rejection: publish zero, clear RNN state and action history, record the
     reason and require consecutive stable samples before resuming.
   - Apply the guard on the exact `map -> base` pose path used by policy goal
     construction. An unused tracker elsewhere is not sufficient.

4. **Restore safety defaults.**
   - Reverse disabled for all initial tests.
   - VO/shield enabled.
   - Policy command must not bypass the safety topic chain.
   - Start with low outer limits; do not use full speed for first validation.

5. **Add a strict model-bundle manifest.**
   Fail startup on any mismatch:
   - raw observation dimension/order and normalization;
   - 79D versus 83D action-history mode;
   - K=8 frame history;
   - action table and decoder constants;
   - reverse scale, slew constants and `speed_rate`;
   - control period;
   - issued-history semantics;
   - issued-history normalization divisors;
   - model SHA-256.

6. **Do not add heuristic 200 ms prediction compensation.**
   The new training lineage models command delay as U{0,1,2} at 5 Hz. Vehicle
   observation delay remains separate and must not be used to double-count
   latency.

7. **Collect actuator system-identification data.**
   With the vehicle restrained or at low speed, independently measure:
   - command-to-motion dead time;
   - linear step response `alpha_v`;
   - angular step response `alpha_omega`;
   - steady-state tracking scale for both channels.
   Return raw timestamped command and odometry data plus the fitting script.
   Do not reuse the uncalibrated scalar `alpha=0.3`.

8. **Expand diagnostics.**
   Log with synchronized timestamps:
   - policy logits and two action indices;
   - decoded target;
   - post-slew issued command;
   - post-VO/shield published command;
   - measured odometry;
   - TF pose delta and jump-guard state;
   - the exact four history values fed to the 83D policy.

## Mandatory Tests

1. Exhaustively compare all 361 action pairs against a frozen training-side
   decoder fixture:
   `/home/aa/IsaacLab/docs/freeze/sa1_action_contract_v1.json`.
   The fixture also contains reverse saturation, positive angular slew and
   full sign-flip stateful sequences. Verify its recorded source hashes before
   consuming it.
2. Assert forward/reverse boundaries and angular/linear slew for multi-step
   sequences.
3. Feed a fixed observation sequence through Python/TorchScript/vehicle
   inference and compare logits, indices and decoded commands.
4. Inject a 97-degree TF jump and assert zero output, state reset and guarded
   recovery.
5. Assert 83D history equals the commands entering the actuator delay queue.
6. Assert startup rejects wrong observation dimensions, constants or checksum.

## Acceptance Sequence

Do not jump directly to free driving:

1. Offline recorded-observation replay with motors disabled.
2. Lifted-wheel or restrained test.
3. Open-space test with reverse off, VO on and low `v/omega` limits.
4. Only after stable command/history/TF logs, proceed to bounded obstacle tests.

Stop immediately on an unexplained reverse command, TF jump passthrough,
repeated full-steer sign flips, manifest mismatch or safety bypass.

Deliver changed file list, exact tests and outputs, the 361-action parity
artifact, TF-jump test evidence, and system-identification data. Do not report
"fixed" based only on a GUI drive.
