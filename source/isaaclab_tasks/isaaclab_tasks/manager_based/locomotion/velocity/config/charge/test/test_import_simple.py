#!/usr/bin/env python3
"""簡單的導入測試"""

import sys
import os

# 添加當前目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 60)
print("測試 charge_mdp 模組導入")
print("=" * 60)
print(f"當前目錄: {current_dir}")
print(f"charge_mdp.py 存在: {os.path.exists(os.path.join(current_dir, 'charge_mdp.py'))}")

try:
    import charge_mdp
    print("✅ charge_mdp 模組導入成功")
    
    # 測試函數是否存在
    funcs = ['set_obstacle_metadata', 'get_obstacle_metadata', 'reset_obstacle_targets']
    for func_name in funcs:
        if hasattr(charge_mdp, func_name):
            print(f"✅ {func_name} 存在")
        else:
            print(f"❌ {func_name} 不存在")
    
    # 測試基本功能
    print("\n測試基本功能...")
    charge_mdp.set_obstacle_metadata(5, [0.5, 0.6, 0.7, 0.8, 0.9])
    num, sizes = charge_mdp.get_obstacle_metadata()
    print(f"設置數量: 5, 獲取數量: {num}")
    print(f"設置尺寸: [0.5, 0.6, 0.7, 0.8, 0.9], 獲取尺寸: {sizes}")
    
    if num == 5 and sizes == [0.5, 0.6, 0.7, 0.8, 0.9]:
        print("✅ 基本功能測試通過")
    else:
        print("❌ 基本功能測試失敗")
    
    print("\n" + "=" * 60)
    print("✅✅✅ 所有測試通過! ✅✅✅")
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ 發生錯誤: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
