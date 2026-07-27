"""W1: deployment-kinematics corridor arm — patrol → wander, nothing else.

The D1 weighted-sampling arm is permanently archived: doubling random_2d
exposure moved nothing (33.79% → 32.90% → 35.18% CR over 20 iters) while the
down-weighted longitudinal family degraded monotonically. The mechanism
audits that followed (pause A/B, phase attribution, counterfactual, teacher
shadow/override) converged on the patrol kinematics themselves: a ~1.5 m
ping-pong leg reverses every ~3.3 s, inside the 1.5 s future-occupancy
horizon, so a large share of constant-velocity predictions straddle a
turnaround the reward cannot anticipate. The user has ruled that deployment
pedestrians do not do this; the deployment kinematics are a continuous
bounded random walk ("wander").

Zero-training evidence for the switch: the same D0 policy scores 33.79% CR
under patrol but 21.94% under wander (3 seeds, 4,203 episodes, TO 0%).

W1 therefore changes exactly one training variable: the corridor random_2d
family runs "wander" instead of "patrol". Sampling stays balanced 1:1:1
env-stratified (the D1 lesson), the corridor share stays 10%, and reward,
future-occupancy horizon, network, PPO, LR and every other replay are
untouched. `test_sa6_sa8_replay_configs.py` locks this allowed diff
permanently.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_env_stratified_probe_from_c100 import (
    CONFIG as _D0,
)


_D0_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
)


CONFIG = replace(
    _D0,
    name="e2e_sa6_w1_wander_from_d0",
    description=(
        "Thirty-iteration W1 arm from the D0 checkpoint. The corridor "
        "random_2d family switches from the patrol ping-pong to the bounded "
        "random walk (wander); sampling stays balanced 1:1:1 env-stratified "
        "and the corridor replay fraction stays at 10%."
    ),
    checkpoint=_D0_CHECKPOINT,
    no_resume_optimizer=False,
    long_corridor_random_2d_kinematics="wander",
    tags=_D0.tags + ("wander_corridor_kinematics", "w1_arm"),
    notes=(
        "Resume the D0 model and Adam state. Reward, future-occupancy weight "
        "0.10 and horizon 1.5 s, network, PPO settings, LR, the 68/10/12/10 "
        "replay mix and the 10% corridor share are all unchanged; the only "
        "training variable is random_2d kinematics patrol -> wander. Save "
        "every five iterations. Deployment gates (wander profile): every "
        "corridor mode SR>=90 / CR<=10 / TO<=5 and Gate2 SR>=90 / CR<=9 / "
        "TO<=4; the first full JOINT PASS freezes immediately. If 30 iters "
        "pass without a JOINT PASS, report the curve only - no reweighting, "
        "no extension, no reward changes."
    ),
)
