---
name: 3F 場景行人 IRA 設定流程
description: Isaac Sim 5.1 在 3F 走廊場景設置行人的完整流程和已知問題
type: project
---

## 已完成

1. **IRA extensions**: `isaacsim.replicator.agent.core` + `isaacsim.replicator.agent.ui`
2. **角色來源**: NVIDIA CDN `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/People/Characters/`
3. **角色走動**: 使用 `omni.anim.people` behavior script + GoTo 指令 + `navmesh=False`
4. **走路動畫**: NVIDIA 官方角色 + Biped_Setup animation graph
5. **完整設定腳本**: `/home/aa/charge_rl/setup_pedestrians_complete.py`

## 已知問題

- **NavMesh 碎片化**: 3F 走廊的門/隔牆把 NavMesh 切成不連通區域 → `navmesh=True` 跨區域路徑失敗
- **NavMesh random point 查詢**: IRA 的 Generate Random Commands 失敗 → 手動寫命令
- **RTX LiDAR 不偵測 skeletal mesh**: `type=Lidar` 的 RTX LiDAR 不對 animated character mesh 做 raytracing
- **OmniLidar**: 待測試，`type=OmniLidar` 在 Isaac Sim 5.1 可建立
- **Capsule collider 不是人體輪廓**: 用戶要求 LiDAR 照出人體外形，不接受 capsule 近似
- **MCP execute_script**: 迴圈超過 5 次的 USD 操作容易觸發 validation error → 改用 Script Editor
- **Referenced prim 子節點**: `stage.DefinePrim` 在 CDN reference 子路徑失敗 → 用 `CreatePrimWithDefaultXform` command

## 設定流程

1. Actor SDG → Config → `/home/aa/charge_rl/ira_3floor_config.yaml` → Set Up Simulation
2. Script Editor → `exec(open("/home/aa/charge_rl/setup_pedestrians_complete.py").read())`
3. Play

**Why:** 用戶需要在 3F 走廊場景中加入可被 LiDAR 偵測的移動行人，用於導航 RL 訓練。
**How to apply:** 每次重啟 Isaac Sim 後需重新執行此流程。
