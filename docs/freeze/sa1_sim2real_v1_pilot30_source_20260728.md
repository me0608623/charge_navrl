# SA1 Sim-to-Real V1 Pilot-30 Source Freeze

- Frozen at: `2026-07-28T19:55:42+08:00`
- Repository: `/home/aa/IsaacLab`
- Branch: `wdclean-repro-20260429-pcB`
- HEAD: `5a7acb2e600c4b8c70c37d6eb78f2b5327cb2790`
- Worktree: dirty by design; the hashes below, rather than HEAD alone, define
  the training source.
- Vehicle-side 361-action parity, 83D history, TF guard and safety chain:
  reported complete by the user; not independently rerun in this repository.

## Run Contract

```text
run_name=sa1_sim2real_v1_ne1024_s42_pilot30_20260728
experiment_config=e2e_sa1_k8_obb_sim2real_v1
num_envs=1024
seed=42
timesteps=3840
rollout_length=128
iterations=30
save_interval=5
checkpoint=None
resume_optimizer=False
```

The pilot uses the exact mainline behavior recipe. The shorter timestep budget
is the only run-length override. A pilot checkpoint is not a deployment
candidate.

## Effective Behavior

```text
native=0.78
narrow=0.12, width=1.2..1.4 m, segment_length=10.0 m
long_corridor=0.10, width=4.0 m, length=10.0 m
corridor_density=25% 3S1D / 35% 4S2D / 20% 4S3D / 15% 5S3D / 5% 5S5D
corridor_motion=env_stratified
random_2d_kinematics=wander
gate_aligned_share=0.0
lidar_stack=8
raw_observation=83D
issued_history_normalizers=(0.5, 1.2)
vlp16_noise=full
actuator_delay_steps=U{0,1,2}
velocity_scale=(1.0,1.0)
motor_lag_alpha=1.0
observation_delay=(0,0)
RNN=unused
```

## SHA-256

```text
5fbb373e9d0d7340b5c3ff491e4418652563377cef4b170eefa6838f31a4866f  scripts/reinforcement_learning/skrl/rnn_car_modular/experiment_config.py
03ea3cae76d07f4620d1e70a9277ca23092a39780379e3b3f365ae21687bae99  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_k8_future_frozen_base.py
a03262eb3e8b8be1076346cd1fbc4fac74ad4cb0ee37beb46bca9cd8f1a9ae85  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa1_k8_obb.py
68a0d02c9fb1fd3f2b91fb6f368440daf31516d5390f2b38d4f4f247b243d3fc  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa1_k8_obb_scene_mix.py
a167602da2a4d74f83b52d874a93797016d2ef4d5a6b26ce63cca98ec83f6639  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa1_k8_obb_actuator_delay_only.py
590a706378dd8ab91c4ba40d8f30502c0389637f29683ab985f8623b34034236  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa1_k8_obb_sim2real_v1.py
17706a410ce4fd4482f926a7e2e64fdca963870437bb80f48af97fbe32b0a446  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/e2e_sa7_wander_from_w1c10.py
da84ae83ee05b2cff0cb8b421bac97b7b75aabbbf35dbc4cc80c8ac1c8356b40  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py
bdc4099245645f5c829c155da30b5698b780b74480ed0f05a8fae603eb96890c  scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py
d0bffc19d81b40642e505811badbf40a80533cdfae278a920c59220e0cf6a823  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/cfg/charge_env_cfg_vlp16.py
650c9b204b5b71b0c988b6997077aedb6f0e9535c62d63399b70b6a23d3a3cc8  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/goal_obstacle_curriculum.py
f50c8562917d9d1dc5618a1965d9136943eebeb4a5d27f9d0df3a88edc155aed  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/domain_randomization/__init__.py
a9bc1ef46117a769290fa6037f91b3e6244b44fecfa7b9b8409523082c6e165d  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/domain_randomization/actuator_dr.py
54ff36caaa18da6e36eb45bc7830e5ae60fc2b25055f93644b8cc88171d6f2bd  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/actions/discrete_differential_drive.py
3e7092a6576393a8976548359494c9989172c73982582bc1b607f3cfdc43ee8d  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/behavior_scheduler.py
84447669f926c34c671ea8e39b82016f80a2ba4189608f05d638f70a75188d86  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/corridor_density.py
1048bfb3f19ec02773b4fc72f1e8d21fa8af91901b7c7ead899af7d5ea68e312  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/long_corridor_replay.py
887ee815dc7cf187a816b60417c1e94683b061681b844aaaba0c175ba44e14a2  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/long_corridor_replay_geometry.py
5592795759b6c499f2f1e9e55e1aa0194e8179f82a49083678dc147e4f313d31  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/narrow_passage_bridge.py
eff1ed2ba47ef630cf3212bef01957e6bb0e3932739dfe6b44c4c3e97a997891  source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/observations/obs_functions.py
```
