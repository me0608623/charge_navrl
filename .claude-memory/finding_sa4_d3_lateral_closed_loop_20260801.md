---
name: SA4-D3 lateral closed-loop mechanism diagnosis
description: SA4-R2 c50 fixed-screen result and D3 baseline/brake/turn/combined causal diagnostic; all arms fail total Gate
type: finding
date: 2026-08-01
updated: 2026-08-01
status: authoritative-hold
---

# SA4-D3 lateral closed-loop mechanism diagnosis

## Authority and decision

- Current state: `SA4 HOLD`.
- `accepted_parent=null`; do not start SA5.
- Reject sustained-brake, best-turn, and combined as deployment shields, SA4 completion,
  or training parents.
- Preserve the mechanism finding: lateral dynamic collisions are steerable, but action
  selection must include wall/static geometry feasibility.

Obsidian source of truth:

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/20_SA4_R2與D3_Lateral閉環介入診斷_20260801.md`

Paper-ready card:

`/home/aa/Documents/Obsidian Vault/isaaclab_v4/Gate結果/21_SA4_D3論文數據卡_20260801.md`

## Fixed identity

- Checkpoint:
  `logs/rnn_car/sa4_r2_futureocc015_from_sa3r1_c100_ne1024_s42_p50_r1/checkpoint_6400.pt`
- Checkpoint SHA-256:
  `c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197`
- Cell: stage4 `corridor_lateral`, evaluator seed818, d1=200 ms, 64 env,
  2,500 steps/arm.
- Protocol SHA-256:
  `6625c6997afc9f7d51d061f5ed46052a19ca9f66a56839218f519ed9f259065a`
- Ordering:
  `policy index -> optional shield index -> decode -> d1 queue -> applied command`.
- Trigger: distance <=3.0 m, closing >0, `max(distance-0.70,0)/closing <=2.0 s`
  for two consecutive frames.

## SA4-R2 checkpoint screen

| checkpoint | lateral CR | longitudinal CR | worst CR | status |
|---|---:|---:|---:|---|
| c3200 / it25 | 23.78% | 31.92% | 31.92% | FAIL |
| c6400 / it50 | 16.95% | 1.53% | 16.95% | FAIL |

c6400 is a diagnostic checkpoint, not an accepted parent. The unrun 0.10 control means
the c50 learning curve cannot be causally attributed to future-occupancy weight 0.15.

## D3 closed-loop results

| arm | episodes | SR | total CR | dynamic CR | static CR | wall CR | TO |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1976 | 83.05% | 16.95% | 15.79% | 0.66% | 0.51% | 0.00% |
| sustained brake | 1835 | 46.92% | 53.08% | 52.21% | 0.76% | 0.11% | 0.00% |
| best turn | 2237 | 14.80% | 85.20% | 2.28% | 57.53% | 25.39% | 0.00% |
| combined | 1977 | 22.56% | 77.39% | 8.40% | 62.62% | 6.37% | 0.05% |

All arms fail SR>=90%, CR<=10%, TO<=5%.

## Mechanism interpretation

1. Sustained brake did execute: median dynamic-contact speed 0.102 m/s and 99.69% of
   collision windows had an override. The exact rule often stopped inside the crossing
   path and increased dynamic CR. This does not reject all earlier braking.
2. Best turn reduced dynamic CR by 85.6% relative, proving a steerable component, but
   shifted failures to static obstacles and walls.
3. Combined reduced dynamic CR below 10% but static CR remained 62.62%; dynamic-only
   success is not total Gate success.
4. Do not increase only future-occupancy reward, stop in the conflict path, or blindly use
   maximum turn.

## Validity

- Four arms each recorded exactly 160,000 transitions.
- Baseline policy/effective actions were bitwise identical.
- d1 alignment errors were zero in all arms.
- Record, episode, and dynamic-collision ledgers reconciled.
- Source fingerprints were stable before/after every rollout.
- Full CPU regression: `800 passed, 27 subtests passed`.
- Scope is one checkpoint and one evaluator seed; do not claim general statistical
  significance or universal braking/turning behavior.

## Next authorized design work

Design, freeze, and CPU-test a conflict-point-aware, geometry-feasible 19x19 selector that
jointly evaluates dynamic crossing risk and LiDAR/OBB wall-static clearance. Do not run a
new GPU comparison or SA5 until that intervention contract is explicit and frozen.

Raw evidence:

- `logs/gates/sa4_d3/intervention_suite_20260801/MECHANISM_ANALYSIS.md`
- `logs/gates/sa4_d3/intervention_suite_20260801/PROTOCOL.json`
- `logs/gates/sa4_d3/intervention_suite_20260801/comparison.json`
- `logs/gates/sa4_d3/intervention_suite_20260801/suite_manifest.json`

