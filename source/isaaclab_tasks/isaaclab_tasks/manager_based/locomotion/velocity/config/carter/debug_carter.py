"""
Carter 診斷腳本

此腳本用於診斷和測試 Carter 機器人環境的配置是否正確。
它會檢查：
1. 機器人資訊（關節、Body 等）
2. 動作配置（動作空間、動作項等）
3. 執行器配置（剛度、阻尼等）
4. 手動控制測試（發送動作並觀察機器人移動）

使用方法：
    python debug_carter.py --num_envs 4
"""

import argparse  # 命令列參數解析
import torch  # PyTorch 張量運算庫

from isaaclab.app import AppLauncher  # Isaac Lab 應用啟動器

# 解析命令列參數
parser = argparse.ArgumentParser(description="Carter 機器人診斷腳本")
parser.add_argument("--num_envs", type=int, default=4, help="並行環境數量（預設：4）")
AppLauncher.add_app_launcher_args(parser)  # 添加應用啟動器的參數（如 --device, --headless 等）
args_cli = parser.parse_args()

# 啟動 Isaac Sim 應用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 導入（必須在 AppLauncher 之後，因為需要先初始化 Isaac Sim）
import gymnasium as gym  # Gymnasium 環境庫
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # 環境配置解析工具

# 正確創建環境配置
env_cfg = parse_env_cfg(
    "Isaac-Navigation-Carter-v0",  # 環境 ID
    device=args_cli.device,  # 設備（CPU 或 GPU）
    num_envs=args_cli.num_envs,  # 並行環境數量
)

# 創建環境
env = gym.make("Isaac-Navigation-Carter-v0", cfg=env_cfg)

print("\n" + "="*70)
print("🔍 Carter 診斷報告")
print("="*70)

# 1. 機器人資訊
robot = env.unwrapped.scene["robot"]  # 獲取機器人資產
print(f"\n【機器人資訊】")
print(f"  關節數量: {robot.num_joints}")  # 關節總數
print(f"  關節名稱: {robot.joint_names}")  # 關節名稱列表
print(f"  Body數量: {robot.num_bodies}")  # 剛體總數
print(f"  Body名稱: {robot.body_names}")  # 剛體名稱列表

# 2. 動作資訊
print(f"\n【動作配置】")
print(f"  動作空間: {env.action_space}")  # Gymnasium 動作空間
print(f"  動作維度: {env.unwrapped.action_manager.total_action_dim}")  # 總動作維度
print(f"  動作項數量: {len(env.unwrapped.action_manager._terms)}")  # 動作項數量

# 遍歷所有動作項，顯示詳細資訊
for name, term in env.unwrapped.action_manager._terms.items():
    print(f"\n  動作項 '{name}':")
    print(f"    類型: {type(term).__name__}")  # 動作項類別名稱
    print(f"    維度: {term.action_dim}")  # 動作維度
    
    # 嘗試獲取更多資訊
    if hasattr(term, '_asset'):
        print(f"    控制資產: {term._asset.cfg.prim_path}")  # 控制的資產路徑
    if hasattr(term, 'cfg'):
        cfg = term.cfg
        if hasattr(cfg, 'joint_names'):
            print(f"    目標關節模式: {cfg.joint_names}")  # 目標關節名稱模式

# 3. 執行器資訊
print(f"\n【執行器配置】")
for name, actuator in robot.actuators.items():
    print(f"  '{name}':")
    print(f"    關節: {actuator.joint_names}")  # 控制的關節名稱
    print(f"    數量: {len(actuator.joint_names)}")  # 關節數量
    print(f"    剛度: {actuator.stiffness}")  # 執行器剛度
    print(f"    阻尼: {actuator.damping}")  # 執行器阻尼

# 4. 手動測試
print(f"\n【手動控制測試】")
obs, _ = env.reset()  # 重置環境

# 創建最大動作（用於測試機器人是否能正常移動）
# 注意：這裡使用 5.0 作為動作值，但實際動作空間是 [-1, 1]，會被裁剪
max_action = torch.ones(args_cli.num_envs, env.action_space.shape[0], device=env.unwrapped.device) * 5.0
print(f"  發送動作: shape={max_action.shape}, 值={max_action[0]}")

print(f"\n  執行 10 步...")
initial_pos = robot.data.root_pos_w[0, :2].clone()  # 記錄初始位置（僅 X, Y）

# 執行 10 個時間步，觀察機器人移動
for i in range(10):
    obs, reward, terminated, truncated, info = env.step(max_action.cpu().numpy())
    
    if i % 3 == 0:  # 每 3 步打印一次
        print(f"\n  Step {i}:")
        print(f"    關節位置: {robot.data.joint_pos[0, :4]}")  # 前 4 個關節的位置
        print(f"    關節速度: {robot.data.joint_vel[0, :4]}")  # 前 4 個關節的速度
        print(f"    根部線速度: {robot.data.root_lin_vel_w[0]}")  # 基座線速度（世界座標系）
        print(f"    根部角速度: {robot.data.root_ang_vel_w[0]}")  # 基座角速度（世界座標系）
        current_pos = robot.data.root_pos_w[0, :2]  # 當前位置（僅 X, Y）
        print(f"    機器人位置XY: {current_pos}")
        print(f"    移動距離: {torch.norm(current_pos - initial_pos).item():.4f}m")  # 從初始位置的移動距離

# 計算總移動距離
final_pos = robot.data.root_pos_w[0, :2]
total_distance = torch.norm(final_pos - initial_pos).item()

# 輸出診斷結果
print("\n" + "="*70)
print(f"✅ 診斷完成")
print(f"   總移動距離: {total_distance:.4f}m")
if total_distance < 0.01:
    print("   ⚠️  機器人幾乎沒有移動！")  # 警告：機器人可能配置有問題
else:
    print("   ✅ 機器人正常移動")  # 正常：機器人能夠移動
print("="*70 + "\n")

# 清理資源
env.close()  # 關閉環境
simulation_app.close()  # 關閉模擬應用