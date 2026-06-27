# NaN Debugging Notes (VLP16 Training)

## Root Cause Chain
1. Scene entities (obstacle_0..4) occasionally have NaN in `root_pos_w` or `root_lin_vel_w`
2. `topk_obstacles_body_frame()` reads these NaN values into position/velocity features
3. Invalid slot zeroing (`px_norm * valid_flag`) fails because `NaN * 0 = NaN` (IEEE 754)
4. `shared_states` (347D) gets NaN in obstacle dims (216-247)
5. `RunningStandardScaler` processes NaN input -> running mean/var become NaN permanently
6. ALL subsequent preprocessor outputs become NaN -> critic outputs NaN -> values stored as NaN
7. GAE produces NaN returns/advantages -> NaN loss -> (without guard) NaN gradients corrupt ALL parameters

## Key Diagnostic Pattern
- `states (policy obs)` from time t is CLEAN
- `shared_states` from `infos` is from time t+1 (temporal mismatch!) and has NaN
- NaN dims: [216,217,218,219, 223,224,225,226, ...] = first 4 of each 7D obstacle block (px,py,vx,vy)
- dims 4-6 (radius, static/dynamic, valid_bit) are clean because they don't depend on scene entity positions

## Fixes Applied
### File: `charge_skrl/mdp/observations/obs_functions.py`
- Added `nan_to_num` when reading `root_pos_w` and `root_lin_vel_w` from scene entities
- Mark obstacles with NaN positions as invalid in `all_valid` mask
- Replaced `px_norm * v` (NaN*0=NaN) with `torch.where(valid_mask, px_norm, zero)`
- Added final `nan_to_num` safety net before return

### File: `charge_skrl/mdp/observations/functions.py`
- Added `nan_to_num` to `obs_pos_w` in `dynamic_obstacles_state()`

### File: `scripts/reinforcement_learning/skrl/train_charge.py`
- Fixed temporal mismatch: cache shared_states from previous step, use cached version in record_transition
- Added `nan_to_num` before RunningStandardScaler in record_transition, GAE bootstrap, and optimization loop
- Added NaN guard: skip optimizer step if loss is NaN/Inf
- Added NaN gradient check: skip step if gradients contain NaN

### File: `scripts/reinforcement_learning/skrl/vlp16_models.py`
- Added NaN detection logging in VLP16DiscretePolicy.compute() and VLP16Value.compute()

## Verification
After fixes: training runs ~10k steps with no NaN GUARD triggers, real PPO learning happening:
- Value loss: 64.6 -> 8.0 (decreasing)
- Entropy: 0.0609 -> 0.053 (policy differentiating from uniform)
- Policy loss: varying (normal PPO behavior)
