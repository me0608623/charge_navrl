---
name: reward_analysis_rigor
description: Reward analysis must consider full return structure, not just weight ratios — user corrected imprecise reasoning
type: feedback
---

分析 reward weight 比例時不能直接從 weight ratio 推出行為結論。

**Why:** 用戶指出 `goal_vel:static_safety = 5:1` 不能直接推出「理性選擇是接受 26% 碰撞率」：
1. PPO 最大化的是折扣總回報 E[G]，不是單步 reward 比較
2. 函數輸出範圍不同：goal_vel ∈ [-1,1] vs static_safety ∈ [-2.5, +2.0]
3. 前進側不只 goal_vel，還有 goal_progress（w=5.0）一起推
4. 碰撞終端 -50 的隱性成本（失去所有未來 reward）也是安全側的一部分
5. 碰撞率數字需明確口徑（滑動窗口 vs 全局 vs 某 stage）

**How to apply:**
- 分析時要算 **實際 per-step reward 貢獻** (f × w × dt)，不只看 weight
- 前進側 = goal_vel + goal_progress + reaching_goal 全部加總
- 安全側 = static_safety + dynamic_safety + collision_terminal 全部加總
- 結論用「reward 結構偏向 aggressive progress」而非「理性選擇是接受 N% 碰撞率」
- 要證明某策略是「理性最優」需要 E[G|aggressive] > E[G|detour] 的比較
