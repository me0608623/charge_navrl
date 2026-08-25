# SA5-v3 B3 continuation and interrupted 24-cell screen (2026-08-24)

## Training

- Run: `sa5_v3_c50_stage3_b3_cont25_from_it25_ne1024_s42_p25_r1`.
- Exact resume from B3 conceptual it25 `checkpoint_3200.pt`, including PPO optimizer (38 state entries).
- Added 25 iterations to conceptual it50; checkpoints at it30/35/40/45/50.
- Completed successfully in 1,375 s with 0 nonfinite values, reconciliation failures, tracebacks, OOMs, or source drift during training.
- Completion record: `docs/freeze/sa5_v3_c50_stage3_b3_cont25_completion_20260824.json`.

## Screen integrity

- Frozen protocol: 6 checkpoints x P060/P080 0S1D/1S1D = 24 cells, protocol SHA `db02b096...02836`.
- r1 invalidated after 1/24 because `play_rnn_car.py` changed; no cell was reused.
- r2 restarted from zero and reached 22/24, then failed closed after `it50/p060_1s1d` because two corridor-generation files changed at 20:15:59:
  - `long_corridor_replay.py`: `9e6e1f6f...` -> `df1deb74...`
  - `long_corridor_replay_geometry.py`: `f57f4351...` -> `72101d85...`
- r2 source fingerprint: `e9e95a5f...da510` -> `130ca9a9...a77dc`.
- Invalidation: `logs/gates/sa5_v3_c50_stage3_b3_cont25_screen/screen_20260824_r2/SOURCE_DRIFT_INVALIDATION.json`.
- Do not pool r2, append the final two cells under the new source, select a parent, start Phase B, or start SA6.

## Diagnostic pattern from the 22 completed r2 cells

- P060 0S1D passed at all six checkpoints.
- P060 1S1D failed at all six checkpoints. CR trajectory:
  - it25 14.10%
  - it30 15.03%
  - it35 13.59% (best point estimate, still 3.59 pp over gate)
  - it40 13.67%
  - it45 14.47%
  - it50 14.72%
- There is no credible iteration-stacking improvement signal.
- it45 P080 1S1D CR was 32.80% versus baseline 25.61%, a 7.19 pp retention failure.
- The last two it50 P080 cells cannot rescue any candidate because every candidate already fails the required P060 1S1D absolute gate.

## Decision

- Engineering decision: stop stacking iterations and move toward the preregistered 0S1D -> 1S1D density curriculum / separate 2S2D launch-timing diagnosis.
- Formal status remains `INCOMPLETE_FAIL_CLOSED_DO_NOT_POOL`; no formal 24-cell `sa5_candidate=None` verdict may be claimed.
- Do not immediately run r3 while parallel sessions are still editing frozen evaluator sources. If a publication-quality complete table is later required, freeze the new fingerprint and rerun all 24 cells from zero.

Obsidian: `/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/58_B3續訓25輪與24格Screen中止_20260824.md`
