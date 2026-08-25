---
name: project_sim2real_speed_density_v3_20260821
description: Frozen rebuilt SA3-SA4 lineage with joint vehicle-speed and corridor-density profiles
type: project
---

# Sim2real speed-density v3 lineage (2026-08-21)

Frozen authority:
`docs/freeze/sim2real_speed_density_curriculum_v3_20260821.json`
SHA-256 `a2d40f443fb151fca1b0e1f178e5adad600fddb9f90eacd128d6a8b4d363a24a`.

Status: `SA4_V3_FORMALLY_LAUNCHED_ACTIVE_GREEN`.

Core decision:

- Do not resume old SA5 speed0p7 adaptation.
- Rebuild from exact SA2 R1 c100 parent.
- Set vehicle `speed_rate=0.7` and `speed_rate_obs=ego` from SA3 onward.
- Couple pedestrian speed and corridor density in one joint profile; never sample
  them independently.
- Exclude P100 from SA3/SA4 training and graduation gates.

SA3 parent:
`logs/rnn_car/sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1/checkpoint_12800.pt`
SHA-256 `a5ea893446c69257caf7e1d5e55dd3dfc6bfaeeb384429552f727f8ef524188c`.

SA3 profiles:

- 0S1D/P035 15%, 1S1D/P035 15%
- 0S1D/P060 25%, 1S1D/P060 25%
- 2S1D/P035 10%, 3S1D/P035 10%

SA4 profiles after SA3 gate only:

- 0S1D/P060 15%, 1S1D/P060 20%
- 0S1D/P080 10%, 1S1D/P080 15%
- 2S1D/P035 15%, 2S1D/P060 10%, 3S1D/P035 15%

Invariants: K8/83D, full VLP-16 noise, `valid_return_only`, actuator delay U{1,2},
unity actuator scale, motor lag 1.0, rollout 128, lateral/longitudinal 50/50 and no
random-2D in the joint curriculum.

Validation:

- 74 targeted contract/legacy/sampler tests passed.
- GPU smoke `sa3_speed_density_v3_smoke_ne64_s42_p1_20260821_r4` completed 1/1.
- Runtime profile/range mismatch = 0.
- Runtime motion reset share = lateral 0.5, longitudinal 0.5, random-2D 0.

Formal SA3 config:
`e2e_sa3_k8_obb_speed_density_v3_from_sa2_r1_c100` (300 iterations, save every 50,
optimizer reset).

Formal run launched 2026-08-21 14:06:13 CST:
`sa3_speed_density_v3_from_sa2r1_c100_ne1024_s42_p300_r1`
under `sa3-speed-density-v3-from-sa2r1-c100-p300-r1.service`, systemd invocation
`ccf12bcd91c84751910918f99146cc1f`. Runtime banners confirmed speed_rate 0.7,
full VLP-16 noise, valid-return-only, delay U{1,2}, and the exact six joint
profiles. The first metrics had reconciliation_ok=1, profile mismatch=0, no
nonfinite actions, and no Traceback/CUDA/OOM/NaN.

At iteration 3, the detail reporter labeled a CR rebound RED by extrapolating
only a two-iteration startup span to 50 iterations. It is report-only, does not
stop the run, and is not regression evidence. Fixed checkpoint gates remain
authoritative.

Formal SA3 completed normally at 2026-08-21 17:12:21 CST: 300/300 iterations,
38,400 steps, 11,062 s, six checkpoints, no strict runtime errors, no accounting
failures, and no nonfinite actions. Final checkpoint SHA-256:
`6888d4c4759413ddc81f1e1eb13f6fb02244823ea1977a97705f4ee2e5c79122`.

Full-run episode-weighted CR: native 5.04%, narrow 0.003%, corridor 20.19%,
lateral 28.12%, longitudinal 12.26%. Final-50 CR: native 4.64%, narrow 0.008%,
corridor 14.17%, lateral 20.29%, longitudinal 8.07%. Training-window hints put
it250 slightly ahead of it200/it300, but this is not a fixed-gate ranking.

Post-training P060 Phase A was narrowed to c100/c150/c200/c250/c300. c50 is
preserved but omitted because its pre-screen worst-direction CR was about 36 pp
behind the leading cluster. Frozen protocol:
`docs/freeze/sa3_v3_p060_checkpoint_screen_v1.json`, SHA-256
`b127a7cd704001c01a5a50f9e0ce3e6ef7e6f8c049c8c3b2aacb905ab5095067`.
The ten-cell queue (five checkpoints x P060 lateral 0S1D/1S1D) started under
`sa3-v3-p060-checkpoint-screen-r1.service`, invocation
`a144757f50f243e897ef4cae7e9f25f7`. It cannot auto-run retention or SA4.

The queue completed normally at 2026-08-21 18:38:57 CST: 10/10 cells valid,
zero invalid, source fingerprint stable, and service exit 0. Ranking by the
pre-registered worst fixed-cell CR was:

- c300: 0S1D 4.39%, 1S1D 7.73%, worst 7.73%, hard-gate PASS.
- c200: 0S1D 4.13%, 1S1D 9.61%, worst 9.61%, hard-gate PASS.
- c250: 0S1D 4.66%, 1S1D 9.71%, worst 9.71%, hard-gate PASS.
- c150: 0S1D 4.40%, 1S1D 10.23%, worst 10.23%, hard-gate FAIL.
- c100: 0S1D 7.82%, 1S1D 17.56%, worst 17.56%, hard-gate FAIL.

c300 is 1.88 percentage points better than c200 and is outside the 0.5 pp
descriptive tie band. c200 and c250 differ by only 0.10 pp, so their point
estimates are practically tied even though the pre-registered top-two output is
c300/c200. These two are promoted for native, narrow and P035 retention only.
No parent was accepted, no retention cell auto-started, the SA4 sentinel was not
replaced, and SA4 was not started. Authoritative result:
`logs/gates/sa3_v3_p060_checkpoint_screen/screen_20260821_r1/SUMMARY.json`.

SA4 config remains fail-closed with a pending parent sentinel. Only an SA3 checkpoint
passing fixed P060 0S1D/1S1D plus native/narrow/P035 retention may replace it.

The frozen c300/c200 retention protocol is
`docs/freeze/sa3_v3_retention_screen_v1.json`, SHA-256
`5945e50dc634ddc8138b7cdaaeaacc9470aad650606bcb1a47a43da888008720`.
It fixed evaluator seed 818, 64 env, 2500 steps, at least 1000 completed
episodes, speed_rate 0.7 ego observation, full VLP-16 noise,
valid-return-only eligibility, d1=200 ms delay-only actuation, and four cells
per candidate: native, narrow, P035 0S1D lateral, and P035 1S1D lateral.

The retention queue completed 8/8 valid cells with zero invalid and a stable
source fingerprint (`bba45095a69c0e514462a24d5ed0a7e71e9932dda4995a539153e8ccec283b46`):

- c300 native: n=3249, SR 96.24%, CR 3.72%, TO 0.03%, PASS.
- c300 narrow: n=3200, SR 100%, CR/TO 0%, crossing/direct 100%, PASS.
- c300 P035 0S1D: n=1823, SR 98.68%, CR 1.32%, PASS.
- c300 P035 1S1D: n=1790, SR 95.92%, CR 4.08%, PASS.
- c200 native: n=3098, SR 96.22%, CR 3.42%, TO 0.36%, PASS.
- c200 narrow: n=3200, SR 100%, CR/TO 0%, crossing/direct 100%, PASS.
- c200 P035 0S1D: n=1851, SR 94.87%, CR 5.13%, PASS.
- c200 P035 1S1D: n=1820, SR 91.48%, CR 8.52%, PASS.

Both candidates pass all retention cells. The preregistered Phase A order and
retention result recommend c300. This is still single-training-seed,
single-evaluator-seed developmental evidence: accepted_parent remains null,
the SA4 sentinel has not been replaced, and SA4 has not started. Authoritative
result:
`logs/gates/sa3_v3_retention_screen/screen_20260821_r1/SUMMARY.json`.

User formally accepted c300 as the sole SA4-v3 parent on 2026-08-21. The
acceptance record is
`docs/freeze/sa3_v3_c300_parent_acceptance_20260821.json`, SHA-256
`b941ef990651eb2102be418ec2cbb7a32e9b76a2b9ec3acf258ceb2890ae7272`.
Accepted checkpoint SHA-256 remains
`6888d4c4759413ddc81f1e1eb13f6fb02244823ea1977a97705f4ee2e5c79122`.

`e2e_sa4_k8_obb_speed_density_v3_from_sa3` uses the exact c300 path. Import
verifies five hashes: checkpoint, acceptance record,
P060 summary, retention summary, and curriculum freeze. It also compares every
non-allowlisted dataclass field against the Stage-4 factory. Budget remains 300
iterations / 38,400 steps, save every 50, optimizer reset at the stage boundary,
speed_rate 0.7 ego, full VLP-16 valid-return-only, and actuator delay U{1,2}.
Import and registry resolution passed; 36 focused tests passed.

The 64-env, one-iteration GPU smoke
`sa4_speed_density_v3_smoke_ne64_s42_p1_20260821_r1` completed successfully.
It confirmed Stage 4, the exact seven frozen profiles, `speed_rate=0.7`, full
VLP-16 `valid_return_only`, delay U{1,2}, `reconciliation_ok=1`, and zero
nonfinite actions. Smoke checkpoint SHA-256 is
`5a9c18af95f5f0cc4e70c533df25a81e50a4ac7022ef9ee3a9052f99ab9d8570`.

Formal SA4 launched at 2026-08-21 23:06:31 CST:
`sa4_speed_density_v3_from_sa3r1_c300_ne1024_s42_p300_r1` under
`sa4-speed-density-v3-from-sa3r1-c300-p300-r1.service`, invocation
`f11ee855bb204ad0bb6c61bd86032d91`. The exact accepted c300 checkpoint was
loaded and the optimizer was reset as required by the Stage boundary.

At iteration 6 the unit remained active/running. Startup metrics were SR
88.64%, CR 10.94%, TO 0%, native CR 11.41%, narrow CR 0%, corridor CR 28.30%,
lateral CR 43.75%, longitudinal CR 12.66%, entropy 4.125, KL 0.00415, and 4660
fps. Accounting reconciled, all numeric values were finite, and the strict
Traceback/CUDA/OOM scan was clear. These startup values are not checkpoint
ranking evidence; frozen post-training screens remain authoritative.

Two limitations remain explicit. Retention used fixed d1 while SA4 trains with
U{1,2}, so task difficulty and delay randomization are confounded if the run
underperforms. The curriculum reaches P080, not deployment-bound P100. SA5
auto-launch remains disabled.
