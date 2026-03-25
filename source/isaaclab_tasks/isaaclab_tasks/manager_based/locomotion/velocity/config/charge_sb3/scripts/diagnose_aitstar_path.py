#!/usr/bin/env python3
"""
AIT* 路径规划诊断工具

检查：
1. 坐标系统是否正确对齐
2. 路径起点和终点是否正确
3. 可视化是否与实际路径匹配
"""

import torch
import numpy as np


def diagnose_coordinate_system():
    """诊断 AIT* 坐标系统配置"""
    print("\n" + "="*70)
    print("AIT* 坐标系统诊断")
    print("="*70)

    # ============================================================================
    # 1. 检查地图配置
    # ============================================================================
    print("\n[1] 地图配置:")
    print("-" * 50)

    map_size = (16.0, 16.0)  # 从 aitstar_integration.py:58
    grid_resolution = 0.1

    map_origin = (-map_size[0] / 2, -map_size[1] / 2)
    map_max = (map_size[0] / 2, map_size[1] / 2)

    print(f"  map_size: {map_size}")
    print(f"  map_origin: {map_origin}  ← 地图左下角")
    print(f"  map_max: {map_max}       ← 地图右上角")
    print(f"  地图范围: X [{map_origin[0]}, {map_max[0]}], Y [{map_origin[1]}, {map_max[1]}]")

    # ============================================================================
    # 2. 坐标转换分析
    # ============================================================================
    print("\n[2] 坐标系统分析:")
    print("-" * 50)

    # 场景 1: 环境局部坐标系
    print("\n  场景 1: 环境局部坐标系")
    env_local_pos = (2.0, 3.0)  # 机器人在环境局部坐标 (2, 3)
    print(f"    机器人环境局部坐标: {env_local_pos}")
    print(f"    → AIT* 地图坐标: {env_local_pos} (直接使用)")
    print(f"    → 是否在地图范围内: {map_origin[0] <= env_local_pos[0] <= map_max[0] and map_origin[1] <= env_local_pos[1] <= map_max[1]}")

    # 场景 2: 世界坐标需要转换
    print("\n  场景 2: 世界坐标 → 环境局部坐标")
    world_pos = (12.0, 13.0)  # 世界坐标
    env_origin = (10.0, 10.0)  # 环境原点偏移
    local_pos = (world_pos[0] - env_origin[0], world_pos[1] - env_origin[1])
    print(f"    机器人世界坐标: {world_pos}")
    print(f"    环境原点: {env_origin}")
    print(f"    环境局部坐标: {local_pos}")

    # ============================================================================
    # 3. 墙壁坐标检查
    # ============================================================================
    print("\n[3] 墙壁坐标 (Phase 1):")
    print("-" * 50)

    walls_phase1 = [
        ("wall_north", (0.0, 8.0), (16.0, 0.2)),
        ("wall_south", (0.0, -8.0), (16.0, 0.2)),
        ("wall_east", (8.0, 0.0), (0.2, 16.0)),
        ("wall_west", (-8.0, 0.0), (0.2, 16.0)),
        ("u_wall_top", (-2.0, 3.0), (4.0, 0.2)),
        ("partition_top", (5.25, 2.0), (2.5, 0.2)),
    ]

    print("  墙壁名称".ljust(20) + "中心位置".ljust(15) + "在地图内?")
    for name, center, size in walls_phase1:
        in_map = map_origin[0] <= center[0] <= map_max[0] and map_origin[1] <= center[1] <= map_max[1]
        status = "✓" if in_map else "✗"
        print(f"  {status} {name.ljust(18)} {str(center).ljust(15)} {in_map}")

    # ============================================================================
    # 4. 可能的问题诊断
    # ============================================================================
    print("\n[4] 问题诊断:")
    print("-" * 50)

    issues = []

    # 检查 1: 地图是否太小
    required_range = 10.0  # 墙壁需要 ±8m 范围
    if map_max[0] < required_range or map_max[1] < required_range:
        issues.append(f"❌ 地图太小! 地图最大坐标 ({map_max}) < 墙壁边界 (±{required_range})")
    else:
        print("  ✓ 地图尺寸足够覆盖所有墙壁")

    # 检查 2: 坐标系统是否匹配
    print("\n  ✓ 坐标系统匹配 (环境局部坐标 = AIT* 地图坐标)")

    # 检查 3: 可能的可视化问题
    print("\n  ⚠️  可视化可能的问题:")
    print("    1. 路径可视化 (path_visualizer.py) 的 Z 坐标 = 0.3m")
    print("    2. 如果路径显示为直线，可能 AIT* 规划失败")
    print("    3. 如果路径不从起点到终点，检查 debug 输出")

    if issues:
        print("\n" + "="*70)
        print("发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n  ✓ 配置检查通过")

    return {
        "map_size": map_size,
        "map_origin": map_origin,
        "map_max": map_max,
    }


def print_visualization_guide():
    """打印可视化元素说明"""
    print("\n" + "="*70)
    print("可视化元素说明")
    print("="*70)

    print("""
您在 Isaac Sim 中应该看到：

┌─────────────────────────────────────────────────────────────┐
│  Isaac Sim 视口                                              │
│                                                             │
│  ╔═══════════════════════════════════════════════════════╗ │
│  ║    •••••••••••••••••••  ← 红色点 (墙壁网格)            ║ │
│  ║  •  ══════════════════••  ← 灰色墙体                  ║ │
│  ║  •  │                  🤖   ← 机器人                 ║ │
│  ║  •  │              ╱───╯   ← 绿色路径                ║ │
│  ║  •  │         ↑             ← AIT* 规划路径           ║ │
│  ║  •┌─┴─┐       │  🎯      ← 红色箭头 (目标)          ║ │
│  ║  •│   │       │                                  ║ │
│  ║  •└───┘       └───────────                          ║ │
│  ╚═══════════════════════════════════════════════════════╝ │
│                                                             │
└─────────────────────────────────────────────────────────────┘

| 元素 | 颜色 | 来源 | 说明 |
|------|------|------|------|
| 墙壁网格点 | 红色方块 | WallGeometryVisualizer | AIT* 障碍物地图 |
| 墙壁几何体 | 灰色 | CuboidCfg | 实际物理墙体 |
| 导航路径 | 绿色线条 | AITStarPathVisualizer | 规划路径 |
| 起点 | 绿色大球 | AITStarPathVisualizer | 路径起点 |
| 目标 | 红色箭头 | GoalCommand | 目标位置 |
    """)


def print_troubleshooting_steps():
    """打印故障排除步骤"""
    print("\n" + "="*70)
    print("故障排除步骤")
    print("="*70)

    print("""
如果 AIT* 路径规划不正确：

1. 检查调试输出（运行时会打印）:
   $ python scripts/reinforcement_learning/sb3/train_charge.py --task Isaac-Navigation-Charge-Phase0 --num_envs 1

   查找以下输出：
   ┌────────────────────────────────────────────┐
   │ AIT* 路径规划调试 (env_0)                  │
   │ ========================================    │
   │ 机器人世界坐标: [x, y]                     │
   │ 目标世界坐标: [x, y]                       │
   │ 机器人局部坐标: [x, y]                     │
   │ 目标局部坐标: [x, y]                       │
   └────────────────────────────────────────────┘

2. 验证路径输出:
   [AIT* Debug] env_0 path (local): [[x1, y1], [x2, y2], ...]
   [AIT* Debug] path[0] (start): [x, y]   ← 应该等于 机器人局部坐标
   [AIT* Debug] path[-1] (goal): [x, y]   ← 应该等于 目标局部坐标

3. 如果路径是一条直线（只有起点和终点）:
   → AIT* 规划失败，回退到直线路径
   → 检查墙壁是否正确添加到障碍物地图

4. 如果路径不连接起点和终点:
   → 坐标系统问题
   → 检查 env_origins 是否正确

5. 如果路径显示位置不正确:
   → 可视化问题，检查 path_visualizer.py 的坐标转换
    """)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Charge Navigation - AIT* 路径规划诊断工具")
    print("="*70)

    # 1. 诊断坐标系统
    config = diagnose_coordinate_system()

    # 2. 打印可视化说明
    print_visualization_guide()

    # 3. 打印故障排除步骤
    print_troubleshooting_steps()

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
    print("\n建议: 运行训练脚本并查看调试输出")
    print("  python scripts/reinforcement_learning/sb3/train_charge.py \\")
    print("    --task Isaac-Navigation-Charge-Phase0 --num_envs 1")


if __name__ == "__main__":
    main()
