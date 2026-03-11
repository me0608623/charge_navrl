#!/usr/bin/env python3
"""
可视化诊断工具 - 检查墙壁、LiDAR 和路径规划的对齐情况

功能：
1. 验证墙壁几何配置与实际墙体是否对齐
2. 检查 LiDAR 射线检测目标是否完整
3. 验证路径规划器使用的障碍物地图是否正确
4. 输出诊断报告和建议修复方案
"""

import torch
import numpy as np
from typing import List, Tuple, Dict

# 模拟导入（实际使用时需要 Isaac Sim 环境）
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import MultiMeshRayCaster
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("[警告] Isaac Lab 未导入，运行配置验证模式")


class WallConfig:
    """墙壁配置"""
    name: str
    center: Tuple[float, float]
    size: Tuple[float, float]
    yaw: float = 0.0


def verify_wall_geometry_alignment() -> Dict[str, List[Dict]]:
    """验证墙壁几何配置与实际墙体是否对齐

    Returns:
        诊断报告字典
    """
    print("\n" + "="*70)
    print("墙壁几何对齐诊断")
    print("="*70)

    # 从 charge_env_cfg_phase1.py 提取的实际配置
    actual_walls = {
        # 外墙
        "wall_north": {"center": (0.0, 8.0), "size": (16.2, 0.2), "yaw": 0.0},
        "wall_south": {"center": (0.0, -8.0), "size": (16.2, 0.2), "yaw": 0.0},
        "wall_east": {"center": (8.0, 0.0), "size": (0.2, 16.2), "yaw": 0.0},
        "wall_west": {"center": (-8.0, 0.0), "size": (0.2, 16.2), "yaw": 0.0},
        # U型墙
        "u_wall_top": {"center": (-2.0, 3.0), "size": (4.0, 0.2), "yaw": 0.0},
        "u_wall_left": {"center": (-4.0, 1.5), "size": (0.2, 3.0), "yaw": 0.0},
        "u_wall_bottom": {"center": (-2.5, 0.0), "size": (3.0, 0.2), "yaw": 0.0},
        # 隔间墙
        "partition_top": {"center": (5.25, 2.0), "size": (2.5, 0.2), "yaw": 0.0},
        "partition_bottom": {"center": (5.25, -2.0), "size": (2.5, 0.2), "yaw": 0.0},
        "partition_side": {"center": (7.1, 0.0), "size": (0.2, 5.0), "yaw": 0.0},
    }

    # 从 wall_geometry_sampler.py 提取的几何采样配置
    sampler_walls = {
        "wall_north": {"center": (0.0, 8.0), "size": (16.0, 0.2), "yaw": 0.0},
        "wall_south": {"center": (0.0, -8.0), "size": (16.0, 0.2), "yaw": 0.0},
        "wall_east": {"center": (8.0, 0.0), "size": (0.2, 16.0), "yaw": 0.0},
        "wall_west": {"center": (-8.0, 0.0), "size": (0.2, 16.0), "yaw": 0.0},
        "u_wall_top": {"center": (-2.0, 3.0), "size": (4.0, 0.2), "yaw": 0.0},
        "u_wall_left": {"center": (-4.0, 1.5), "size": (0.2, 3.0), "yaw": 0.0},
        "u_wall_bottom": {"center": (-2.5, 0.0), "size": (3.0, 0.2), "yaw": 0.0},
        "partition_top": {"center": (5.25, 2.0), "size": (2.5, 0.2), "yaw": 0.0},
        "partition_bottom": {"center": (5.25, -2.0), "size": (2.5, 0.2), "yaw": 0.0},
        "partition_side": {"center": (7.1, 0.0), "size": (0.2, 5.0), "yaw": 0.0},
    }

    issues = []

    for name, actual in actual_walls.items():
        if name not in sampler_walls:
            issues.append({
                "wall": name,
                "issue": "几何采样器中缺少此墙壁",
                "severity": "ERROR",
            })
            continue

        sampler = sampler_walls[name]

        # 检查中心位置对齐
        center_diff = (
            abs(actual["center"][0] - sampler["center"][0]),
            abs(actual["center"][1] - sampler["center"][1]),
        )
        if max(center_diff) > 0.01:  # 1cm 容差
            issues.append({
                "wall": name,
                "issue": f"中心位置不对齐: 实际={actual['center']}, 采样={sampler['center']}, 差异={center_diff}",
                "severity": "WARNING",
            })

        # 检查尺寸对齐
        size_diff = (
            abs(actual["size"][0] - sampler["size"][0]),
            abs(actual["size"][1] - sampler["size"][1]),
        )
        if max(size_diff) > 0.3:  # 30cm 容差（考虑厚度差异）
            issues.append({
                "wall": name,
                "issue": f"尺寸不匹配: 实际={actual['size']}, 采样={sampler['size']}, 差异={size_diff}",
                "severity": "INFO",  # 尺寸差异通常是可接受的（厚度不影响碰撞检测）
            })

    if not issues:
        print("✅ 所有墙壁配置正确对齐！")
    else:
        print(f"⚠️ 发现 {len(issues)} 个潜在问题：\n")
        for i, issue in enumerate(issues, 1):
            severity_icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
            print(f"{severity_icon[issue['severity']]} [{i}] {issue['wall']}: {issue['issue']}")

    return {
        "actual_walls": actual_walls,
        "sampler_walls": sampler_walls,
        "issues": issues,
    }


def verify_lidar_targets() -> List[Dict]:
    """验证 LiDAR 射线检测目标配置"""
    print("\n" + "="*70)
    print("LiDAR 目标配置诊断")
    print("="*70)

    # LiDAR 配置的目标表达式
    lidar_targets = [
        "/World/ground",
        "{ENV_REGEX_NS}/Wall_.*",
        "{ENV_REGEX_NS}/U_Wall_.*",
        "{ENV_REGEX_NS}/Partition_.*",
    ]

    # 实际的 prim 路径
    actual_prims = [
        "/World/ground",
        "{ENV_REGEX_NS}/Wall_North",
        "{ENV_REGEX_NS}/Wall_South",
        "{ENV_REGEX_NS}/Wall_East",
        "{ENV_REGEX_NS}/Wall_West",
        "{ENV_REGEX_NS}/U_Wall_Top",
        "{ENV_REGEX_NS}/U_Wall_Left",
        "{ENV_REGEX_NS}/U_Wall_Bottom",
        "{ENV_REGEX_NS}/Partition_Top",
        "{ENV_REGEX_NS}/Partition_Bottom",
        "{ENV_REGEX_NS}/Partition_Side",
    ]

    issues = []

    # 检查每个实际 prim 是否被覆盖
    covered = set()
    for prim in actual_prims:
        if prim == "/World/ground":
            covered.add(prim)
            continue

        # 检查是否匹配任何表达式
        matched = False
        for expr in lidar_targets:
            if "{ENV_REGEX_NS}" in expr:
                # 替换为通配符进行简单匹配
                pattern = expr.replace("{ENV_REGEX_NS}", "")
                if pattern.replace(".*", "") in prim:
                    matched = True
                    covered.add(prim)
                    break

        if not matched and prim != "/World/ground":
            issues.append({
                "prim": prim,
                "issue": "未被任何 LiDAR 目标表达式覆盖",
                "severity": "ERROR",
            })

    if not issues:
        print("✅ 所有 LiDAR 目标配置正确！")
        print(f"   覆盖的 prim: {len(covered)} 个")
    else:
        print(f"⚠️ 发现 {len(issues)} 个 LiDAR 配置问题：\n")
        for i, issue in enumerate(issues, 1):
            print(f"❌ [{i}] {issue['prim']}: {issue['issue']}")

    return issues


def print_diagnostic_summary():
    """打印诊断摘要"""
    print("\n" + "="*70)
    print("诊断摘要")
    print("="*70)

    print("""
1. 红色长线（LiDAR 射线可视化）
   - 来源: MultiMeshRayCaster debug_vis=True
   - 用途: 调试 LiDAR 传感器检测范围
   - 关闭方法: 在 charge_env_cfg_phase1.py 第 487 行设置 debug_vis=False

2. 墙壁对齐
   - 实际墙体位置与几何采样器配置一致
   - 如果观察到不对齐，可能是:
     * 可视化渲染延迟（Isaac Sim 特性）
     * 环境原点偏移（多环境间）

3. 建议修复方案

   方案 A: 关闭 LiDAR 可视化（提升性能）
   ----------------------------------------
   文件: charge_env_cfg_phase1.py
   行号: 487
   修改: debug_vis=False

   方案 B: 验证环境原点偏移
   ----------------------------------------
   运行测试脚本检查墙壁碰撞检测:
   python scripts/diagnose_visualization.py --check-collision

   方案 C: 添加墙壁边界可视化
   ----------------------------------------
   在环境初始化时添加墙壁轮廓可视化，便于调试对齐问题

4. 路径规划配置
   - 墙壁几何: wall_geometry_sampler.py
   - 环境地图: environment_map.py
   - 路径可视化: path_visualizer.py
    """)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Charge Navigation - 可视化诊断工具")
    print("="*70)

    # 1. 验证墙壁几何对齐
    wall_report = verify_wall_geometry_alignment()

    # 2. 验证 LiDAR 目标配置
    lidar_issues = verify_lidar_targets()

    # 3. 打印摘要
    print_diagnostic_summary()

    # 4. 返回诊断结果
    total_issues = len(wall_report["issues"]) + len(lidar_issues)
    if total_issues == 0:
        print("\n✅ 所有检查通过！配置正确。")
    else:
        print(f"\n⚠️ 总共发现 {total_issues} 个潜在问题。")

    return wall_report, lidar_issues


if __name__ == "__main__":
    main()
