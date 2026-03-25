"""
模組化工作訓練測試

此腳本用於驗證模組化後的環境是否可以正常進行訓練。
測試內容：
1. 環境創建和初始化
2. 環境重置
3. 基本訓練步驟（多個迭代）
4. 檢查觀測、動作、獎勵是否正常
5. 驗證所有模組化組件是否正常工作

使用方法:
  cd ~/IsaacLab
  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py

或者直接運行（需要先啟動 Isaac Sim）:
  python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py --task Isaac-Navigation-Charge-v0 --num_envs 4 --max_iterations 10
"""

import argparse
import sys
import os

# 添加路徑以便導入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

# 解析命令行參數
parser = argparse.ArgumentParser(description="模組化工作訓練測試")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-v0", help="環境任務名稱")
parser.add_argument("--num_envs", type=int, default=4, help="並行環境數量（測試用較少數量）")
parser.add_argument("--max_iterations", type=int, default=10, help="最大訓練迭代次數（測試用較少次數）")
parser.add_argument("--headless", action="store_true", default=False, help="無頭模式（不顯示 GUI）")
parser.add_argument("--device", type=str, default="cuda:0", help="設備（cuda:0 或 cpu）")

# 解析參數
args_cli = parser.parse_args()

# 啟動 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 現在可以導入 Isaac Lab 模組
import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_environment_creation(task_name: str, num_envs: int, device: str):
    """測試環境創建"""
    print("\n" + "=" * 70)
    print(f"測試 1: 環境創建測試 - {task_name}")
    print("=" * 70)
    
    try:
        # 解析環境配置
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        print(f"✅ 環境配置解析成功")
        print(f"   - 環境數量: {env_cfg.scene.num_envs}")
        print(f"   - 設備: {env_cfg.sim.device}")
        
        # 創建環境
        env = gym.make(task_name, cfg=env_cfg)
        print(f"✅ 環境創建成功")
        print(f"   - 動作空間: {env.action_space}")
        print(f"   - 觀測空間: {env.observation_space}")
        
        # 獲取實際環境對象（解包 gymnasium 包裝）
        # gym.make() 返回的環境可能被 OrderEnforcing 包裝，需要訪問實際環境
        unwrapped_env = env
        # 嘗試解包：gymnasium 使用 .unwrapped 屬性
        while hasattr(unwrapped_env, 'unwrapped'):
            unwrapped_env = unwrapped_env.unwrapped
        # 如果還有 .env 屬性（某些包裝器使用這個）
        while hasattr(unwrapped_env, 'env') and unwrapped_env.env is not unwrapped_env:
            unwrapped_env = unwrapped_env.env
        
        print(f"   - 環境類型: {type(unwrapped_env).__name__}")
        print(f"   - 環境數量: {unwrapped_env.num_envs}")
        print(f"   - 設備: {unwrapped_env.device}")
        
        return env, env_cfg, unwrapped_env
        
    except Exception as e:
        print(f"❌ 環境創建失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_environment_reset(env):
    """測試環境重置"""
    print("\n" + "=" * 70)
    print("測試 2: 環境重置測試")
    print("=" * 70)
    
    try:
        # 重置環境
        obs, info = env.reset()
        print(f"✅ 環境重置成功")
        
        # 檢查觀測
        if isinstance(obs, dict):
            print(f"   - 觀測類型: 字典")
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    print(f"     - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                else:
                    print(f"     - {key}: type={type(value)}")
        elif isinstance(obs, torch.Tensor):
            print(f"   - 觀測類型: Tensor")
            print(f"     - shape: {tuple(obs.shape)}")
            print(f"     - dtype: {obs.dtype}")
        else:
            print(f"   - 觀測類型: {type(obs)}")
        
        # 檢查 info
        if info:
            print(f"   - Info 鍵: {list(info.keys())}")
        
        return obs, info
        
    except Exception as e:
        print(f"❌ 環境重置失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_training_steps(env, unwrapped_env, num_steps: int = 10):
    """測試訓練步驟"""
    print("\n" + "=" * 70)
    print(f"測試 3: 訓練步驟測試（{num_steps} 步）")
    print("=" * 70)
    
    try:
        # 重置環境
        obs, info = env.reset()
        
        # 獲取環境屬性（從實際環境對象）
        num_envs = unwrapped_env.num_envs
        device = unwrapped_env.device
        
        # 統計信息
        total_reward = torch.zeros(num_envs, device=device)
        episode_lengths = torch.zeros(num_envs, device=device)
        success_count = 0
        collision_count = 0
        
        for step in range(num_steps):
            # 生成隨機動作
            if isinstance(env.action_space, gym.spaces.Box):
                action = torch.rand(num_envs, *env.action_space.shape, device=device)
                # 縮放到動作空間範圍
                action = action * (env.action_space.high - env.action_space.low) + env.action_space.low
            else:
                action = env.action_space.sample()
                if isinstance(action, torch.Tensor):
                    action = action.to(device)
            
            # 執行步驟
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 累積獎勵
            total_reward += reward
            
            # 統計終止原因
            done = terminated | truncated
            if done.any():
                done_envs = done.nonzero(as_tuple=False).squeeze(-1)
                episode_lengths[done_envs] = step + 1
                
                # 檢查終止原因
                for env_id in done_envs:
                    env_id_int = env_id.item()
                    if "goal_reached" in info and info["goal_reached"][env_id_int]:
                        success_count += 1
                    if "collision" in info and info["collision"][env_id_int]:
                        collision_count += 1
                
                # 重置完成的環境
                if done.any():
                    reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    obs, info = env.reset(seed=None, options={"env_ids": reset_ids})
            
            # 每 5 步打印一次進度
            if (step + 1) % 5 == 0:
                print(f"   ✅ 步驟 {step + 1}/{num_steps} 完成")
                print(f"      - 平均獎勵: {total_reward.mean().item():.4f}")
                print(f"      - 終止環境數: {done.sum().item()}")
        
        print(f"\n✅ 訓練步驟測試完成")
        print(f"   - 總步數: {num_steps}")
        print(f"   - 平均總獎勵: {total_reward.mean().item():.4f}")
        print(f"   - 成功次數: {success_count}")
        print(f"   - 碰撞次數: {collision_count}")
        print(f"   - 平均 episode 長度: {episode_lengths[episode_lengths > 0].mean().item() if (episode_lengths > 0).any() else 0:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 訓練步驟測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modularized_components(env, unwrapped_env):
    """測試模組化組件是否正常工作"""
    print("\n" + "=" * 70)
    print("測試 4: 模組化組件檢查")
    print("=" * 70)
    
    try:
        # 檢查環境配置中的模組化組件（從實際環境對象獲取）
        env_cfg = unwrapped_env.cfg
        
        # 檢查觀測配置
        if hasattr(env_cfg, 'observations'):
            print("✅ 觀測配置存在")
            if hasattr(env_cfg.observations, 'policy'):
                obs_terms = env_cfg.observations.policy
                print(f"   - 觀測項數量: {len([k for k in dir(obs_terms) if not k.startswith('_')])}")
        
        # 檢查獎勵配置
        if hasattr(env_cfg, 'rewards'):
            print("✅ 獎勵配置存在")
            reward_terms = env_cfg.rewards
            reward_count = len([k for k in dir(reward_terms) if not k.startswith('_') and not k.startswith('__')])
            print(f"   - 獎勵項數量: {reward_count}")
        
        # 檢查終止條件配置
        if hasattr(env_cfg, 'terminations'):
            print("✅ 終止條件配置存在")
            term_terms = env_cfg.terminations
            term_count = len([k for k in dir(term_terms) if not k.startswith('_') and not k.startswith('__')])
            print(f"   - 終止條件數量: {term_count}")
        
        # 檢查事件配置
        if hasattr(env_cfg, 'events'):
            print("✅ 事件配置存在")
            event_terms = env_cfg.events
            event_count = len([k for k in dir(event_terms) if not k.startswith('_') and not k.startswith('__')])
            print(f"   - 事件數量: {event_count}")
        
        # 檢查動作配置
        if hasattr(env_cfg, 'actions'):
            print("✅ 動作配置存在")
            action_terms = env_cfg.actions
            action_count = len([k for k in dir(action_terms) if not k.startswith('_') and not k.startswith('__')])
            print(f"   - 動作項數量: {action_count}")
        
        print("\n✅ 所有模組化組件檢查完成")
        return True
        
    except Exception as e:
        print(f"❌ 模組化組件檢查失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("\n" + "=" * 70)
    print("模組化工作訓練測試")
    print("=" * 70)
    print(f"任務: {args_cli.task}")
    print(f"環境數量: {args_cli.num_envs}")
    print(f"最大迭代次數: {args_cli.max_iterations}")
    print(f"設備: {args_cli.device}")
    print("=" * 70)
    
    # 測試結果
    test_results = []
    
    # 測試 1: 環境創建
    env, env_cfg, unwrapped_env = test_environment_creation(
        args_cli.task,
        args_cli.num_envs,
        args_cli.device
    )
    test_results.append(("環境創建", env is not None))
    
    if env is None or unwrapped_env is None:
        print("\n❌ 環境創建失敗，無法繼續測試")
        return
    
    # 測試 2: 環境重置
    obs, info = test_environment_reset(env)
    test_results.append(("環境重置", obs is not None))
    
    if obs is None:
        print("\n❌ 環境重置失敗，無法繼續測試")
        env.close()
        return
    
    # 測試 3: 訓練步驟
    training_success = test_training_steps(env, unwrapped_env, num_steps=args_cli.max_iterations)
    test_results.append(("訓練步驟", training_success))
    
    # 測試 4: 模組化組件檢查
    component_check = test_modularized_components(env, unwrapped_env)
    test_results.append(("模組化組件", component_check))
    
    # 關閉環境
    env.close()
    
    # 打印測試總結
    print("\n" + "=" * 70)
    print("測試總結")
    print("=" * 70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    print("-" * 70)
    print(f"總計: {passed}/{total} 通過")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 所有測試通過！模組化工作驗證成功！")
    else:
        print(f"\n⚠️  有 {total - passed} 個測試失敗，請檢查錯誤信息")
    
    # 關閉 Isaac Sim
    simulation_app.close()


if __name__ == "__main__":
    main()
