"""W2: LR-decay arm — same W1 recipe, decaying optimizer schedule.

Both D1 and W1 peaked around iteration 10 and then drifted: they share the
D0 warm start, the resumed Adam state, LR=2e-4, seed 42 and the PPO/replay
structure, so the common ~iter-10 ceiling is attributed to continued
full-rate updates on a mature policy, not to either arm's sampling change.
Plain continuation is ruled out (c10 + same Adam + same LR + 10 more iters
IS the original c20); the ruled next step is a genuinely different
optimizer schedule.

W2 therefore changes exactly one training variable versus W1: a linear LR
decay of 0.05 per iteration (2.0e-4 -> 1.5e-4 @ iter5 -> 1.0e-4 @ iter10 ->
5.0e-5 @ iter15 -> ~0 @ iter20, trainer floor 1%). The decay is applied by
the trainer directly to the optimizer param_groups from `initial_lr`, which
is re-stamped after the Adam state is loaded — so the checkpoint's stored
2e-4 cannot silently undo the schedule (the failure mode of editing
CONFIG.lr alone). The run is 20 iterations with a checkpoint every 2, and
every checkpoint faces Gate5 (1.2 m narrow gap, direct crossing) as a hard
per-checkpoint gate.

Everything else — wander corridor kinematics, balanced 1:1:1 env-stratified
sampling, the 10% corridor share, the 68/10/12/10 replay mix incl. the 12%
narrow-gap replay and its teacher KL 0.30, reward, network, PPO — is
inherited from W1 untouched. `test_sa6_sa8_replay_configs.py` locks this
allowed diff permanently.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_w1_wander_from_d0 import CONFIG as _W1


CONFIG = replace(
    _W1,
    name="e2e_sa6_w2_wander_lrdecay_from_d0",
    description=(
        "Twenty-iteration W2 arm from the D0 checkpoint with the W1 wander "
        "recipe unchanged; the only new training variable is a linear LR "
        "decay of 0.05 per iteration on the resumed Adam optimizer. "
        "Checkpoints every 2 iterations for per-checkpoint Gate5."
    ),
    timesteps=20 * _W1.rollout_length,
    save_interval=2,
    lr_decay=0.05,
    tags=_W1.tags + ("lr_decay", "w2_arm"),
    notes=(
        "Resume the D0 model and Adam state (same checkpoint and seed 42 as "
        "W1). Reward, future-occupancy weight 0.10 and horizon 1.5 s, "
        "network, PPO settings, the 68/10/12/10 replay mix (12% narrow-gap "
        "replay + teacher KL 0.30 included) and the 10% corridor share are "
        "all unchanged; the only training variable versus W1 is lr_decay "
        "0.05/iter. Save every two iterations; every checkpoint must pass "
        "Gate5 direct crossing as a hard gate, c10/c20 additionally run the "
        "full corridor four-mode suite + Gate2. Verify the schedule via the "
        "logged actual param_group lr, not CONFIG.lr. If W2 still collapses "
        "around c10, the next arm is a fixed D0 KL anchor — no plain "
        "continuation, no reweighting, no reward changes."
    ),
)
