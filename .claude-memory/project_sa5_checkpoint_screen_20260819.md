# SA5 fixed multi-checkpoint screen

Date: 2026-08-19
Status: HISTORICAL; SEALED_R3 STOPPED EARLY, SA5-R2 DENSITY-MIX TRAINING ACTIVE

Current authority: see
`project_sa5_r2_sealed_densitymix_launch_20260819.md`. The sealed r3 queue was
stopped after 9 valid cells with sufficient diagnostic evidence. The
geometry-only SA5-R2 candidate below was superseded before formal training by
the user-requested low-density mix; its files and smoke evidence are retained.

Critical geometry erratum:

- r2 is wholly `INVALID_GEOMETRY_BYPASS`; retain but never rank or pool it.
- Stage-5 boundary inner faces were at y=+/-7.0m while the old 10m side walls
  ended at y=+/-5.0m, leaving a traversable 2.0m opening at each end.
- The conservative robot diameter is about 1.204m, so this was a real bypass,
  not a visual seam.
- v2 keeps the 10m interaction zone but uses 15m physical side walls, yielding
  0.5m overlap with each north/south boundary wall.
- Runtime JSON must report wall_span_m=15, overlap=0.5,
  sealed_to_boundary_pass=true, and geometry_pass=true or the cell fails closed.

Purpose:

- Phase A compares SA5 c150/c200/c250/c300 on fixed lateral,
  longitudinal, random2d, and mixed corridor cells.
- Phase B runs native and narrow only for the Phase-A top two.
- The queue starts neither training nor SA6 and accepts no parent.

Frozen conditions:

- protocol: `docs/freeze/sa5_checkpoint_screen_v1.json`
- protocol SHA: `b17ba048430a514dc56310b78e26bfe0f789ad116333ce459209c54d96933d94`
- evaluator seed818, 64 env, fixed d1=200ms, actuator profile
  `sa1_delay_only`
- LiDAR `full + valid_return_only`
- Stage5 room 15m, corridor 4.2m x 10m, fixed hard-side 4S2D,
  speed 0.25--0.45m/s, random2d patrol
- corridor cells: 2500 steps, n>=1000
- native/narrow cells: 1200 steps, n>=1000

Ranking is frozen before rollout:

1. minimize max family CR across all four corridor cells;
2. minimize unweighted mean family CR;
3. prefer later checkpoint only on exact score ties;
4. differences below 0.005 from the leader are reported as a tie group, but
   exactly two candidates continue to Phase B.

Hard threshold failures remain visible but do not erase structurally valid
ranking data. Missing JSON, n<1000, runtime mismatch, checkpoint mismatch, or
source drift produces `INCOMPLETE_NO_VERDICT` and blocks Phase B.

Validation:

- focused sealed-geometry and screen tests: 74 passed
- complete wdclean CPU suite in env_isaaclab: 981 passed, 27 subtests passed
- the first broad invocation used system Python without torch and stopped at
  collection; it was rerun correctly in env_isaaclab

Execution:

- r1 exclusive-mode unit was stopped before any Isaac cell and retained at
  `logs/gates/sa5_checkpoint_screen/screen_20260819_r1/`; see
  `SUPERSEDED_WAIT_POLICY.md`.
- active unit: `sa5-fixed-checkpoint-screen-r2.service`
- startup queue PID: 3812308
- output: `logs/gates/sa5_checkpoint_screen/screen_20260819_r2/`
- console: `logs/gates/sa5_checkpoint_screen/screen_20260819_r2.console.log`
- shared mode rechecks GPU free >=12000MiB and available RAM >=12GiB before
  every cell. GPU utilization and coexisting compute apps are recorded rather
  than treated as blockers.
- two launch polls both had 16423MiB free. After the first evaluator loaded,
  it used about 2.7GiB and the GPU retained about 12.8GiB free; there was no
  OOM or initialization error.
- Phase A cell 1/16 (`c150 / corridor_lateral`) started at 12:48 alongside cm
  YOLO PID3214308 (14612MiB). Shared utilization may extend wall-clock time but
  does not alter the fixed step count, seed, or protocol.

The preceding r2 execution above is retained history only. It was stopped
during c200/lateral after c150's four cells completed, then marked invalid in
`screen_20260819_r2/INVALID_GEOMETRY_BYPASS.{md,json}`.

Current execution:

- protocol: `docs/freeze/sa5_checkpoint_screen_v2.json`
- protocol SHA: `7fc8ce95d829b92fb6502979bbbb824dd4ce643dd0ef0e1278367afaf7413e48`
- unit: `sa5-fixed-checkpoint-screen-r3-sealed.service`
- startup queue PID: 3912930
- output: `logs/gates/sa5_checkpoint_screen/screen_20260819_r3_sealed/`
- console: `logs/gates/sa5_checkpoint_screen/screen_20260819_r3_sealed.console.log`
- started at 14:50 with c150/lateral; first formal log records wall_span=15.00m
  and boundary_overlap=0.50m
- first sealed result: c150/lateral n=1878, SR=0.2891, CR=0.7103,
  TO=0.0005; wall CR=0.2455 and obstacle CR=0.4649. The invalid unsealed r2
  counterpart had CR=0.1271, confirming that the bypass materially inflated
  apparent capability. Do not rank until all 16 sealed cells are valid.

Evidence boundary: single training seed, single evaluator seed, fixed d1. This
is checkpoint selection evidence, not the formal multi-seed/multi-delay SA5
graduation matrix. No SA6 authorization is implied.

## SA5-R2 sealed corrective lineage

- New config:
  `e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12.py`.
- New run name:
  `sa5_r2_sealed_from_sa4r3_it125_actd12_ne1024_s42_p300_r1`.
- Restart from the same original SA4-R3 conceptual-it125 checkpoint, SHA
  `57f43d07...95ab71d`; do not inherit a legacy SA5 checkpoint or optimizer.
- ExperimentConfig behavior fields are identical to the completed legacy SA5;
  the frozen environment geometry is the sole behavior difference.
- 64-env/1-iteration GPU smoke passed, loaded the correct parent, observed
  interaction length 10m, physical wall span 15m, boundary overlap 0.5m,
  full+valid_return_only LiDAR, actuator delay U{1,2}, reconciliation=1, and
  wrote checkpoint_128.pt (SHA `ffce25fc...54488`). This is runtime evidence,
  not capability evidence.
- Full CPU suite after adding the lineage and parent-baseline requirement:
  990 passed + 27 subtests.
- Formal 300-iteration training is not started and must not overlap the active
  sealed fixed screen. SA6 remains forbidden.

New qualitative observation: in the sealed GUI, a legacy policy initially
steers toward the former corridor end gap and hits the new wall corner. The
same c150/lateral/s818 fixed cell changes wall CR from 0.00843 unsealed to
0.24547 sealed, directionally supporting learned bypass use. This does not
prove the exact SA4 parent has the same bias. Before training, run that exact
parent on all four sealed corridor families and retain SR/CR/TO plus wall and
obstacle CR as the correction baseline.
