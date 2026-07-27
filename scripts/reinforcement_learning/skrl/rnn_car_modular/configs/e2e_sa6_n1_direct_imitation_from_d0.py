"""N1: direct-crossing imitation — swap the narrow supervision, nothing else.

Three results forced this arm (07-27):

1. The SA5 narrow teacher that W1/W2 distilled with KL 0.30 is itself a
   detourer: 3 seeds, n=2094, crossing 95.9% but **direct 0.000** with
   pre-cross |y| p95 3.86 m. The KL term had been stabilising the detour.
2. Start-distance curriculum is ruled out: W2-c10 scores direct 0.000 from
   1.5 m, 2.0 m and 3.0 m alike — from 1.5 m the robot still swings to
   |y|~3.8 m before crossing. The detour is not a distance artefact.
3. A scripted privileged controller crosses the same gap straight every
   time: 3 seeds, n=7104, direct 1.000, pre-cross |y| p95 0.0096 m,
   4.2 s to cross, zero backtrack. The gap is physically passable; the
   detour is learned behaviour.

W2 (LR decay) is archived as a diagnostic: its Gate5 plateau did not
survive a seed change (s1 oscillated 37 -> 94 -> 62 while s42 held 99-100),
so N1 returns to the plain W1 recipe and constant LR.

N1 also fixes a deployment-distribution defect discovered before launch:
historical narrow replay pinned every goal exactly 3.0 m behind the opening
with zero lateral offset. N1 samples goal distance in [2.0, 4.0] m and lateral
offset in [-1.5, 1.5] m so the teacher's two phases are exercised: aim through
the opening first, then turn toward the off-axis goal. This makes N1 a
capability arm, not a pure one-variable attribution experiment; D0 must be
measured on the same randomized-goal gate before training.

The policy-learning change remains the narrow-frame supervision:

    total = PPO loss + lambda * [CE(linear head, teacher linear bin)
                                 + CE(angular head, teacher angular bin)]

applied **only** on narrow-replay frames. The teacher labels; it never
drives — no rollout override, so PPO keeps collecting its own clean
on-policy data. lambda is calibrated from a shadow rollout so the CE
gradient starts at roughly half the PPO actor gradient rather than being
guessed.

Deferred by verdict until direct crossing is solved: the D0 KL anchor
(it would pull the direct=0 behaviour back), actuator delay, and any
random_2d / mixed_iid work.

`test_sa6_sa8_replay_configs.py` locks this allowed diff permanently.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_w1_wander_from_d0 import CONFIG as _W1


# ===========================================================================
# BLOCKED — do not launch this arm as written (2026-07-27 verdict).
# ===========================================================================
# The premise ("the policy cannot cross a narrow gap straight") was an
# artefact of the evaluation scene, not a property of the policy.
#
# `NarrowGapSpec.arena_half_extent` was hard-coded to 5.0 and `--arena_size`
# was never passed in, so the central wall always ended at y=+/-4.5 m. Only
# at a 10 m arena does that seal against the outer wall; every other size
# leaves a side opening. Measured on D0 (gap 1.2 m):
#
#   arena  outer inner face  side opening  direct crossing
#   10 m       +/-4.5 m          0 m        0.000   (89.4% wall crashes)
#   12 m       +/-5.5 m        1.0 m        0.0042  (detours through the
#                                                    side opening; lateral
#                                                    median 5.02 m)
#   14 m       +/-6.5 m        2.0 m        1.000
#
# With the barrier actually sealed (new `--narrow_gap_mode sealed`, Gate5a)
# at the training arena size, D0 scores **direct 1.000 over 2304 episodes,
# 0% collisions, 0.052 m median pre-cross lateral**. A second, independently
# sealed scene (the production narrow replay) agrees: 0.9993 over n=6738.
#
# So N1 would teach, at SA6/14 m, a behaviour the policy already performs
# perfectly. No training value.
#
# The real finding is different and worth its own arm: at 12 m the policy
# abandons the 1.2 m central gap (~0.4 m OBB clearance) for a 1.0 m side
# opening (~0.2 m clearance) — it takes the *more* dangerous route. That is
# not path optimisation; it is a room-scale shortcut read implicitly off the
# 360-degree LiDAR.
#
# Per the verdict, direct-crossing imitation is deferred to the SA8 / 12 m
# setting and only if Gate5a still fails there. `_LAMBDA` below is the
# SA6/14 m measurement and is **void for that setting** — re-run
# `run_n1_shadow_calibration.sh` against the 12 m recipe before any launch.
_ARM_IS_BLOCKED = True
_BLOCKED_REASON = (
    "premise refuted: D0 already scores direct 1.000 on the sealed Gate5a at "
    "the training arena size; deferred to SA8/12m per the 2026-07-27 verdict"
)

# Measured 2026-07-27 on the SA6 / 14 m recipe (now void — see above):
#   grad(CE)  = 15.84770 / 15.76935
#   grad(PPO) =  1.05597 /  3.16143
#   pooled lambda = 0.5 * sum(grad(PPO)) / sum(grad(CE)) = 0.06669 -> 0.067
_LAMBDA = 0.067
_LAMBDA_IS_CALIBRATED = False   # void for the SA8/12m setting; must re-derive


CONFIG = replace(
    _W1,
    name="e2e_sa6_n1_direct_imitation_from_d0",
    description=(
        "Ten-iteration N1 arm from the D0 checkpoint on the W1 wander "
        "recipe. The detouring SA5 narrow teacher KL is removed and "
        "replaced by a cross-entropy against the scripted direct-crossing "
        "teacher on narrow-replay frames only. Checkpoints every 2 "
        "iterations for per-checkpoint Gate5."
    ),
    timesteps=10 * _W1.rollout_length,
    save_interval=2,
    teacher_retention_weight=0.0,
    narrow_imitation_weight=_LAMBDA,
    narrow_passage_goal_distance_range=(2.0, 4.0),
    narrow_passage_goal_lateral_offset_range=(-1.5, 1.5),
    tags=_W1.tags + (
        "direct_crossing_imitation",
        "randomized_narrow_goal",
        "n1_arm",
    ),
    notes=(
        "Resume the D0 model and Adam state (same checkpoint and seed 42 as "
        "W1). Reward, future-occupancy weight 0.10 and horizon 1.5 s, "
        "network, PPO settings, constant LR, the 68/10/12/10 replay mix and "
        "the 10% corridor share are all unchanged. Narrow goals now sample "
        "distance 2.0-4.0 m and lateral offset -1.5..+1.5 m; D0/W1/W2 retain "
        "their historical fixed 3.0 m / 0 m goal. Two coupled supervision "
        "changes make one policy-loss variable: teacher_retention_weight "
        "0.30 -> 0 (that teacher "
        "detours) and narrow_imitation_weight 0 -> lambda (scripted "
        "direct-crossing CE on the same 12% narrow frames). The teacher "
        "never drives — no rollout override — so PPO data stays clean. "
        "Verify the schedule via wandb narrow_imitation/* (CE, per-head "
        "action agreement, and the CE-vs-PPO gradient ratio). Save every "
        "two iterations; every checkpoint faces both aligned and randomized-"
        "goal Gate5. Record D0 zero-shot on randomized-goal Gate5 first. "
        "Pass conditions: "
        "Gate5 crossing >=95% and direct >=95%, CR <=5%, pre_y p95 <=1 m, "
        "first_cross_time p95 <=10 s, Gate2/lateral/longitudinal still PASS, "
        "and random_2d/mixed_iid no more than 2 pp worse than D0. If direct "
        "crossing succeeds but general navigation regresses, the next arm "
        "adds a D0 anchor scoped to non-narrow envs only — not this one."
    ),
)
