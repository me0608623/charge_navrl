# Charge Robot URDF 檢查報告

## 已修復的問題 ✅

1. **空的 mesh 標籤** (line 436) - 已刪除 `<mesh/>` 空標籤
2. **URDF 表達式語法** - 將 `${3.1415926/2}` 替換為 `1.5707963`（π/2 的數值）
3. **Fixed joints 的 axis 標籤** - 已移除所有 fixed joints 中不必要的 axis 標籤

## Link 位置分析

### 基礎結構
- **base_link** (root): 機器人主體
  - Collision box: size="0.66 0.4 1.55" (長x寬x高，單位:米)
  - Collision offset: xyz="-0.128 0 0.645" (相對於視覺中心)

- **base_inertial**: 慣性中心 link
  - Position: xyz="0 0 0" (與 base_link 同位置)

- **base_footprint**: 足跡參考點
  - Position: xyz="0 0 -0.13366" (在 base_link 下方約 13.4cm)

### 驅動輪
- **right_wheel**: 右驅動輪
  - Position: xyz="0 -0.277 0" (base_link 右側 27.7cm)
  - Axis: xyz="0 1 0" (Y軸旋轉，正確)
  - Collision: 圓柱體 radius=0.134m, length=0.085m

- **left_wheel**: 左驅動輪
  - Position: xyz="0 0.277 0" (base_link 左側 27.7cm)
  - Axis: xyz="0 1 0" (Y軸旋轉，正確)
  - Collision: 圓柱體 radius=0.134m, length=0.085m

**輪距**: 0.277 + 0.277 = 0.554m (55.4cm) ✓

### 腳輪 (Caster Wheels)
- **right_caster_origin_link**: 右側腳輪支架
  - Position: xyz="-0.333 -0.12 0.01675" (後方33.3cm, 右側12cm, 上方1.675cm)
  - Axis: xyz="0 0 -1" (Z軸旋轉，用於轉向)

- **right_caster_wheel_link**: 右側腳輪
  - Position: xyz="-0.034301 0 -0.081" (相對於 origin，前移3.4cm，下移8.1cm)
  - Axis: xyz="0 1 0" (Y軸旋轉，滾動軸)

- **left_caster_origin_link**: 左側腳輪支架
  - Position: xyz="-0.333 0.12 0.01675" (後方33.3cm, 左側12cm, 上方1.675cm)
  - Axis: xyz="0 0 -1" (Z軸旋轉，用於轉向)

- **left_caster_wheel_link**: 左側腳輪
  - Position: xyz="-0.034301 0.0 -0.081" (相對於 origin，前移3.4cm，下移8.1cm)
  - Axis: xyz="0 1 0" (Y軸旋轉，滾動軸)

### 傳感器
- **imu_link**: IMU 傳感器
  - Position: xyz="0.00073991 -0.0116 0.087735" (近中心，右側1.16cm，上方8.77cm)
  - 位置合理 ✓

- **front_yd_lidar**: 前方 YD LiDAR
  - Position: xyz="0.16585 -0.16385 0.2225" (前方16.6cm, 右側16.4cm, 上方22.25cm)
  - Rotation: rpy="0 0 0" (註釋中提到可能是 rpy="0 0 2.3562")

- **back_yd_lidar**: 後方 YD LiDAR
  - Position: xyz="-0.42185 0.16385 0.2225" (後方42.2cm, 左側16.4cm, 上方22.25cm)
  - Rotation: rpy="0 0 -3.14159" (180度旋轉)

- **velodyne**: Velodyne LiDAR
  - Position: xyz="-0.103 0 1.3375" (後方10.3cm, 中心, 上方133.75cm)
  - Rotation: rpy="0 0 3.1416" (180度旋轉)
  - Collision: 圓柱體 radius=0.0516m, length=0.0717m

## 潛在問題檢查

### ✅ 正常
1. 所有 joint 都有正確的 parent 和 child
2. 所有 continuous 類型 joint 都有 axis 定義
3. Wheel joints 使用 Y 軸旋轉（正確）
4. Caster origin joints 使用 Z 軸旋轉（轉向，正確）
5. Caster wheel joints 使用 Y 軸旋轉（滾動，正確）

### ⚠️ 需要注意
1. **base_link 的 collision 偏移較大** (xyz="-0.128 0 0.645")
   - 這可能導致碰撞檢測時視覺和碰撞體不一致
   - 建議: 檢查碰撞體位置是否正確

2. **base_link 沒有 inertial 屬性**
   - 目前使用 base_inertial link 作為慣性中心
   - 這是可接受的設計模式

3. **一些 link 缺少 inertial 屬性**
   - imu_link, front_yd_lidar, back_yd_lidar 沒有 inertial
   - 對於固定連接的小型傳感器，這通常是可接受的

## 機器人尺寸總結

- **長度**: 約 0.66m (collision box)
- **寬度**: 約 0.4m (collision box) 
- **高度**: 約 1.55m (collision box，不含 velodyne)
- **輪距**: 0.554m
- **驅動輪半徑**: 0.134m
- **驅動輪寬度**: 0.085m

## 建議

1. ✅ 已修復所有語法問題
2. ⚠️ 建議驗證 collision box 的位置和尺寸是否與實際機器人匹配
3. ⚠️ 如果需要更精確的碰撞檢測，可以為 lidar 和 imu 添加簡化的 collision geometry
4. ✅ URDF 結構完整，所有 joint 連接正確
