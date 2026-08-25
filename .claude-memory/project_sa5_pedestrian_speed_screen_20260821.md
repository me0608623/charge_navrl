# SA5 c250 pedestrian-speed screen

## Status

- COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC on 2026-08-21.
- Output: `logs/gates/sa5_pedestrian_speed_screen/screen_20260821_r1/`.
- Protocol SHA-256: `c1b3ab6950483e8779296bafb4ff9720fd297c0f04dff7d07723475da2d41339`.
- Source fingerprint stable before/after; all four cells valid; no training or SA6.
- Fixed c250 (`df14c45d...37a4a`), sealed 4S2D lateral, evaluator seed 818, 64 env, 3000 steps, speed_rate=1.0, d1=200ms, full valid-return-only VLP16 noise.
- Only independent variable: pedestrian speed range P035=0.25-0.45, P060=0.50-0.70, P080=0.70-0.90, P100=0.90-1.10 m/s.

## Results

- P035: n=2058, SR/CR/TO=61.18/38.82/0%; obstacle/wall CR=37.80/1.02%.
- P060: n=2947, SR/CR/TO=21.28/78.72/0%; delta CR +39.90pp (19.83 conservative SE).
- P080: n=3801, SR/CR/TO=6.00/94.00/0%; delta CR +55.18pp (32.19 SE).
- P100: n=4633, SR/CR/TO=1.53/98.47/0%; delta CR +59.64pp (37.53 SE).
- Impact radial closing p50/p90 rises monotonically: 0.216/0.376, 0.386/0.719, 0.496/0.904, 0.714/1.057 m/s. This is collision-normal radial closing, not full 2D relative speed.
- Wall CR falls 1.02%->0.15%; essentially all added failures are dynamic-obstacle collisions, not failure transfer to walls.
- Actual body stop frame share remains 14.7-18.0%. A stop appears somewhere in the 5s collision window for 39.7-69.0% of events, but this is not sustained yielding; those episodes still collide.
- D5 dynamic-feasible action exists within the final 1s for only 22.56/27.86/28.01/28.65% of collision events. Persistent no-dynamic-feasible onset p50 is 1.2/1.2/1.2/1.0s.
- Diagnostic label: `D5_MODEL_FEASIBILITY_LIMITED_MAJORITY_WITHOUT_FINAL_1S_DYNAMIC_SOLUTION`.

## Evidence boundary and decision

- D5 is a frozen 19x19 finite-horizon model. Its no-feasible result is not physical inevitability and does not exclude earlier sustained slowing/waiting or continuous-space actions.
- The screen proves the frozen policy degrades severely as pedestrians speed up in the fixed high-density lateral exam. It does not prove a newly trained policy cannot learn fast pedestrians.
- Do not continue the late SA5 speed0p7 adaptation. Do not set every 4S2D corridor to 0.9-1.1m/s.
- New SA3->SA4 lineage should use a joint speed-density curriculum: fast pedestrians first in 0S1D/1S1D, intermediate speed in medium density, P035-dominant high-density with a small late P060 share.
- Before freezing the new training config, run a low-density speed_rate 1.0 vs 0.7 cross-screen. The original high-density Phase 2 is superseded because Phase 1 shows it is already D5-model feasibility-limited.
- Obsidian: `Gate結果/42_SA5_c250行人速度四臂固定Screen_20260821.md`.
