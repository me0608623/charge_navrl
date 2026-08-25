# Real-robot angular tracking aggregate was not a steady-state gain

Date: 2026-08-18
Status: RAW CSV REANALYZED; LOW-GAIN CLAIM REJECTED; DELAY MISMATCH RETAINED

Two 2026-08-17 robot diagnostic sessions initially produced whole-session ratios
`mean(abs(act_w)) / mean(abs(sent_w))` near 0.28 and 0.44. Those ratios must not
be encoded as actuator velocity scale.

Why the aggregate is invalid as gain:

- Session 1 includes roughly t=2--24 s where policy angular commands exist while
  measured linear and angular motion remain zero.
- Session 2 after roughly t=31 s is consistent with manual reverse or mux
  takeover. The source note explicitly says the policy-driver field cannot see
  the final post-mux command.
- The aggregate does not align command and response in time despite a measured
  300--400 ms lag.

Clean moving windows, excluding those intervals and aligning lag:

| Window | Static angular gain | Best lag |
|---|---:|---:|
| segment 1a | 1.042 | 400 ms |
| segment 1b | 0.983 | 400 ms |
| segment 1c | 1.024 | 400 ms |
| segment 2a | 1.013 | 400 ms |
| segment 2b | 0.997 | 400 ms |

Auxiliary ARX estimates give DC gain about 0.965--1.046 and delay about
350--400 ms. The defensible conclusion is:

1. Angular steady-state scale is approximately unity in the clean evidence.
2. Do not train with scale 0.28--0.44 from these aggregates.
3. The delay center is still mismatched: original U{0,1,2} at 200 ms steps has
   mean 200 ms, while reported p50 is 300 ms and clean-window lag is 350--400 ms.
4. SA5 replacement uses U{1,2}, mean 300 ms, and keeps scale and motor-lag alpha
   at 1.0. This is a bounded median-delay approximation, not a full empirical
   delay distribution.

Sources:

- `docs/car_side_reply_Q1_velocity_base_20260818.md`
- `/home/aa/Documents/Obsidian Vault/isaaclab_v4/部署/2026-08-17_車端實測/2026-08-17_速度變化量與位姿guard_給PC訓練端.md`
- `diag_20260817_202713.csv`
- `diag_20260817_202836.csv`

Do not restore the low-gain interpretation without a new controlled experiment
that records the final post-mux command and measured body motion on the same
clock under confirmed autonomous control.
