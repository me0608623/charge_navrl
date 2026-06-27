---
name: Warp Drive to Isaac Lab Migration
description: WarpDrive → Isaac Lab 遷移差異分析與三套融合方案
type: project
---

## 關鍵差異
| 面向 | Warp Drive | Isaac Lab | 融合難度 |
|------|-----------|-----------|--------|
| Recurrent policy | 自建 RNN in module_connected | 無 (FC only) | 中 |
| Auxiliary loss | PreProcess_Module_Loss | 無 | 中-高 |
| Multi-Agent | 3 policies 交替 | 單 policy | 高 |
| Gradient detach | preprocess ↔ RL 分離 | 無 | 中 |
| Algorithm | A2C | PPO | 低（PPO 更好） |

## 三套方案
### 保守: RNN only
- vlp16_models.py 加 GRU layer (128→64 hidden)
- SKRL 原生支援 RNN, 風險低

### 平衡: Modular + Auxiliary Loss
- 新增 ModularPolicy: preprocess→GRU→detach→RL
- 新增 preprocess_loss.py
- 修改 wandb_trainer.py 加 auxiliary loss hook

### 積極: Multi-Agent Adversarial
- 加 learnable Goal/Obstacle agents
- 工程量大，僅在方案二成功後考慮

## 建議路徑
保守方案先驗證 RNN 效果 → 成功則推進平衡方案
