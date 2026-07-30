# SA1 Nav20 Acceptance — Result Record

> **Status: BLOCKED (closed 2026-07-29T19:45+08:00).**
> The acceptance itself is complete and passing. Three rules of the governing
> directive — R1 must keep training, no SA2 auto-launch, completion guard
> preserved for c1900/c2109 — refer to events that occurred at 18:37–18:44 and
> are irreversible preconditions, not open work. They cannot be remedied by any
> later action, so this record is closed rather than left pending.
> Standing constraints while blocked: do not restart R1, do not stop SA2, do not
> modify freeze-list files.

- Recorded at: `2026-07-29T19:35+08:00`
- Matrix: `docs/freeze/sa1_r1_nav20_acceptance_matrix_20260729.md`
- Schema: `sa1_r1_acceptance/v3`
- Blocking scene: `nav20_clean` only. `nav20_native` is advisory and was not re-run.

## Verdict

Two checkpoints from `sa1_sim2real_v1_ne1024_s42_r1` each passed all nine
blocking cells independently. No score is pooled across seeds or delays.

| checkpoint | iteration | md5 | blocking cells | verdict |
|---|---|---|---|---|
| `checkpoint_217600.pt` | it1700 | `6b04e292b48ca9ef596b48bc58027739` | 9/9 PASS | `sa1_pass=true` |
| `checkpoint_230400.pt` | it1800 | `364327db40fcc572f2c17ba1b861be0d` | 9/9 PASS | `sa1_pass=true` |

Thresholds: `SR >= 0.98`, `CR <= 0.015`, `TO <= 0.005`.

## Per-cell results

```text
        cell |  c1700 SR      CR     ep  gate |  c1800 SR      CR     ep  gate
  d0 seed515 |    1.0000  0.0000   2581  PASS |    1.0000  0.0000   2544  PASS
  d0 seed616 |    1.0000  0.0000   2554  PASS |    1.0000  0.0000   2549  PASS
  d0 seed717 |    1.0000  0.0000   2545  PASS |    1.0000  0.0000   2554  PASS
  d1 seed515 |    1.0000  0.0000   2219  PASS |    1.0000  0.0000   2252  PASS
  d1 seed616 |    1.0000  0.0000   2213  PASS |    1.0000  0.0000   2228  PASS
  d1 seed717 |    1.0000  0.0000   2184  PASS |    1.0000  0.0000   2237  PASS
  d2 seed515 |    1.0000  0.0000   1923  PASS |    1.0000  0.0000   1952  PASS
  d2 seed616 |    1.0000  0.0000   1931  PASS |    1.0000  0.0000   1929  PASS
  d2 seed717 |    0.9990  0.0010   1945  PASS |    1.0000  0.0000   1938  PASS
```

Seventeen of eighteen cells are identical at `SR=1.0000, CR=0`. The single
difference is two collision episodes out of 1945 in `c1700 d2 seed717`, against
an allowance of 29. With SR pinned at its 1.0 ceiling this difference is not
separable from measurement noise, so **c1800 must not be called better than
c1700**. What the pair does establish is that c1700's result is not a fluke: a
checkpoint 100 iterations later reproduces it.

## Structural verification (both runs)

```text
cells                                9/9
schema                               all sa1_r1_acceptance/v3
threshold dictionary                 unique across all cells
(delay, seed) coverage               complete, no gaps, no duplicates
cell_id                              unique  (assert_no_cross_delay_average passed)
actuator profile/scale/alpha         9/9 correct (sa1_delay_only, [1.0,1.0], 1.0)
[SIM2REAL] actuator_dr marker        9/9 present
  pipeline=decode->delay->scale->lag 9/9 match
  history=issued_command_queue       9/9 match
[SIM2REAL] VLP16 mode=full           9/9 present
cell.invalid.json                    0
```

The validator re-derives each `gate_pass` from the frozen thresholds and
cross-checks it against the runner's own `pass`; it does not simply relay the
runner's verdict.

## Delay dose-response

Episodes fall monotonically with dead time in both runs
(c1700 2560 → 2205 → 1933, c1800 2549 → 2239 → 1940), confirming that 200 ms and
400 ms genuinely alter the control loop. SR/CR/TO do not degrade under either.

## Why c1900 and c2109 are absent

R1 was stopped externally at `18:37:12` by `systemctl --user stop` and SIGKILLed
at `18:38:42`, halting at iteration 1808/2109 with no completion marker. This was
not a crash, OOM, or stall: the kernel log has no OOM record, the console has
zero strict error matches, and peak memory was 7.1 GB of 62 GB.

The frozen comparison points can never be produced afterwards, for an
implementation reason rather than a scheduling one:

```text
train_rnn_car_wdclip.py:5435  for iteration in range(num_iterations)   # always from 0
train_rnn_car_wdclip.py:8236  total_steps = (iteration + 1) * RL
train_rnn_car_wdclip.py:8946  ckpt_path = f"checkpoint_{total_steps}.pt"
grep for restoring the checkpoint's saved `iteration` on load -> 0 hits
```

Any resume restarts the counter, writes `checkpoint_6400.pt` onward, and — if the
original `run_name` is reused — overwrites R1's existing 36 checkpoints.
`checkpoint_269952.pt` can therefore never exist. c1800 is the surviving late
checkpoint and was evaluated as the substitute comparison point.

## Deployment status

`deployable=false` in both verdicts, unchanged by these passes:

- `reverse_fraction` is still not emitted by the frozen `play_rnn_car.py`
- 83D/K8 Python-to-TorchScript parity has no vehicle-side JSON; the historical
  `e2e_parity_oracle.py` is 79D/K4 and invalid for this checkpoint

Passing the SA1 hard gate does not make a checkpoint deployable.

## Artefacts

```text
logs/gates/sa1_r1_nav20_acceptance/early_c1700_clean/   9 x cell.json + stage_summary.json + sa1_verdict.json
logs/gates/sa1_r1_nav20_acceptance/late_c1800_clean/    9 x cell.json + stage_summary.json + sa1_verdict.json
```

`stage_summary.json` reports `matrix_complete=false` and
`advisory_cells_complete=false` in both runs. That is the correct reflection of
running the blocking scene only, and must not be presented as a complete matrix.
