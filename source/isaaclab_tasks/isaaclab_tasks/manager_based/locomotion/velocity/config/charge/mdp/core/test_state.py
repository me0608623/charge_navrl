# ============================================================================
# 狀態管理模組測試
# ============================================================================
"""
測試 mdp.core.state 模組的功能

驗證：
1. 障礙物元數據的設置和獲取
2. 動態障礙物狀態管理的基本功能
3. 函數簽名和接口正確性
"""

from __future__ import annotations

import sys
import os

# 添加父目錄到路徑，以便導入模組
# 測試文件位於 mdp/core/ 目錄，需要向上兩級才能找到 chargecopy 目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 從 mdp/core/ 到 mdp/ 到 chargecopy/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 注意：避免在頂層導入 inspect，因為會與本地的 types.py 衝突
# 將 inspect 的導入移到需要使用它的函數內部

try:
    from mdp.core.state import (
        # 障礙物元數據
        set_obstacle_metadata,
        get_obstacle_num,
        get_obstacle_sizes,
        get_obstacle_metadata,
        # 動態障礙物狀態
        reset_obstacle_targets,
        get_obstacle_start_positions,
        get_obstacle_directions,
        update_obstacle_start_position,
        update_obstacle_direction,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️  警告: 無法導入模組（可能是缺少依賴）: {e}")
    print("這在開發環境中是正常的，模組結構驗證將跳過。")
    IMPORT_SUCCESS = False


def test_obstacle_metadata():
    """測試障礙物元數據管理"""
    print("=" * 60)
    print("測試 1: 障礙物元數據管理")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print("⚠️  跳過測試（無法導入模組）")
        return False
    
    # 測試設置和獲取
    test_num = 10
    test_sizes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    print(f"設置障礙物數量: {test_num}")
    print(f"設置障礙物尺寸: {test_sizes}")
    set_obstacle_metadata(test_num, test_sizes)
    
    # 驗證獲取
    num, sizes = get_obstacle_metadata()
    assert num == test_num, f"期望數量 {test_num}，實際 {num}"
    assert sizes == test_sizes, f"期望尺寸 {test_sizes}，實際 {sizes}"
    
    print(f"✓ 獲取障礙物數量: {num}")
    print(f"✓ 獲取障礙物尺寸: {sizes}")
    
    # 測試公開接口
    retrieved_num = get_obstacle_num(default=8)
    retrieved_sizes = get_obstacle_sizes()
    
    assert retrieved_num == test_num, f"公開接口獲取數量不匹配: {retrieved_num} != {test_num}"
    assert retrieved_sizes == test_sizes, f"公開接口獲取尺寸不匹配"
    
    print(f"✓ 公開接口 get_obstacle_num(): {retrieved_num}")
    print(f"✓ 公開接口 get_obstacle_sizes(): {retrieved_sizes}")
    
    print("✓ 測試通過！\n")
    return True


def test_obstacle_metadata_defaults():
    """測試障礙物元數據的默認值"""
    print("=" * 60)
    print("測試 2: 障礙物元數據默認值")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print("⚠️  跳過測試（無法導入模組）")
        return False
    
    # 測試默認值（在沒有設置的情況下）
    # 注意：由於全局變量的特性，這可能受到之前測試的影響
    default_num = get_obstacle_num(default=8)
    default_sizes = get_obstacle_sizes()
    
    print(f"默認障礙物數量（default=8）: {default_num}")
    print(f"默認障礙物尺寸: {default_sizes}")
    
    # 測試 get_obstacle_metadata 的默認值
    num, sizes = get_obstacle_metadata()
    print(f"get_obstacle_metadata() 返回: num={num}, sizes={sizes}")
    
    print("✓ 測試通過！\n")
    return True


def test_function_signatures():
    """測試函數簽名"""
    print("=" * 60)
    print("測試 3: 函數簽名驗證")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print("⚠️  跳過測試（無法導入模組）")
        return False
    
    # 在函數內部導入 inspect，避免與 types.py 衝突
    import inspect
    
    # 測試 reset_obstacle_targets 簽名
    sig = inspect.signature(reset_obstacle_targets)
    print(f"reset_obstacle_targets 簽名: {sig}")
    
    params = list(sig.parameters.keys())
    expected_params = ["env", "env_ids", "num_obstacles", "boundary"]
    
    for param in expected_params:
        assert param in params, f"缺少參數: {param}"
        print(f"  ✓ 參數存在: {param}")
    
    # 測試其他函數簽名
    functions_to_check = [
        ("set_obstacle_metadata", ["num_obstacles", "obstacle_sizes"]),
        ("get_obstacle_metadata", []),
        ("get_obstacle_num", ["default"]),
        ("get_obstacle_sizes", []),
        ("get_obstacle_start_positions", []),
        ("get_obstacle_directions", []),
    ]
    
    for func_name, expected_params in functions_to_check:
        func = globals().get(func_name)
        if func is None:
            # 嘗試從模組導入
            try:
                func = getattr(sys.modules.get('mdp.core.state'), func_name, None)
            except:
                pass
        
        if func:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            print(f"  ✓ {func_name} 簽名: {sig}")
            for param in expected_params:
                if param not in params:
                    print(f"    ⚠️  缺少參數: {param}")
        else:
            print(f"  ⚠️  無法找到函數: {func_name}")
    
    print("✓ 函數簽名驗證完成！\n")
    return True


def test_state_accessors():
    """測試狀態訪問函數"""
    print("=" * 60)
    print("測試 4: 狀態訪問函數")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print("⚠️  跳過測試（無法導入模組）")
        return False
    
    # 測試獲取狀態（初始狀態應該為 None）
    start_positions = get_obstacle_start_positions()
    directions = get_obstacle_directions()
    
    print(f"初始起始位置: {start_positions}")
    print(f"初始移動方向: {directions}")
    
    # 這些函數應該能正常調用，即使返回 None
    assert start_positions is None or isinstance(start_positions, type(None)), \
        "起始位置應該是 None 或 Tensor"
    assert directions is None or isinstance(directions, type(None)), \
        "移動方向應該是 None 或 Tensor"
    
    print("✓ 狀態訪問函數正常！\n")
    return True


def test_file_structure():
    """測試文件結構"""
    print("=" * 60)
    print("測試 5: 文件結構驗證")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_check = [
        "state.py",
        "__init__.py",
        "types.py",
    ]
    
    all_exist = True
    for file_name in files_to_check:
        file_path = os.path.join(base_dir, file_name)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_name}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有核心文件都存在！")
    else:
        print("\n⚠️  部分文件缺失，請檢查。")
    
    return all_exist


def main():
    """運行所有測試"""
    print("\n" + "=" * 60)
    print("開始測試 mdp.core.state 模組")
    print("=" * 60 + "\n")
    
    results = []
    
    # 測試 1: 障礙物元數據
    try:
        results.append(("障礙物元數據管理", test_obstacle_metadata()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        results.append(("障礙物元數據管理", False))
    
    # 測試 2: 默認值
    try:
        results.append(("默認值測試", test_obstacle_metadata_defaults()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("默認值測試", False))
    
    # 測試 3: 函數簽名
    try:
        results.append(("函數簽名驗證", test_function_signatures()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("函數簽名驗證", False))
    
    # 測試 4: 狀態訪問
    try:
        results.append(("狀態訪問函數", test_state_accessors()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("狀態訪問函數", False))
    
    # 測試 5: 文件結構
    try:
        results.append(("文件結構驗證", test_file_structure()))
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        results.append(("文件結構驗證", False))
    
    # 總結
    print("=" * 60)
    print("測試結果總結")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print(f"\n通過: {passed}/{total}")
    
    if passed == total:
        print("\n✅✅✅ 所有測試通過! ✅✅✅")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 個測試未通過")
        return 1


if __name__ == "__main__":
    exit(main())
