---
name: deploy-charge-rviz-62
description: 在 62 用 demo.rviz 顯示 charge 車體模型的部署（charge_description package + charge_ 前綴 overlay，不碰 campusrover TF）
metadata: 
  node_type: memory
  type: project
  originSessionId: fc07a025-245e-49c5-8a4a-d4725d3647b9
---

在 viz 筆電 `aa@192.168.3.62`(ROS Humble) 用 `~/demo.rviz` 顯示 charge 車體模型的部署。

**關鍵事實**：真機(`aa@192.168.3.14`, host `ubuntu`)現役 `/robot_description` 是 **campusrover** 模型（`campusrover_description/urdf/campusrover.urdf`，root frame `base_footprint`），由 `rover2_ws/.../campusrover_sensors/launch/start_transform_launch.py` 的 robot_state_publisher 發布。`deploy_rl_shell.sh` 只啟動 RL 策略(`rover_rl_bringup deploy_full.launch.py`)，不含機器人描述。charge.urdf(SolidWorks "charger_description")與 campusrover 是**同一台車的兩套模型**，frame 名稱多數不同（charge 用 velodyne/front_yd_lidar/left_wheel…；campusrover 用 velodyne_link/ydlidar_front_link… 且無輪子 frame）。charge.urdf root link 是 `base_link`（base_footprint 是子節點），與 campusrover 相反。

**部署方式（並存只看，不取代 campusrover）**：
- package `~/charge_ws/src/charge_description`（ament_cmake, `package://` mesh, colcon built, `.bashrc` 已 source install/setup.bash）
- **關鍵：用 `charge_prefixed.urdf`（URDF 裡 link 名稱本身就加 charge_ 前綴）**，robot_state_publisher **不可用 `frame_prefix`**！frame_prefix 只改 TF 輸出 frame 名、不改 URDF link 名 → RobotModel 拿裸 link 名(velodyne…)查 TF 查不到(只有 charge_velodyne) → 整台不顯示。正解是 link 名 == TF frame 名。
- `view_charge_overlay.launch.py`：robot_state_publisher(讀 charge_prefixed.urdf) + remap 描述→`/charge_description`、joint→`/charge_joint_states`；joint_state_publisher 供連續關節；static TF `base_link → charge_base_link`(identity) 橋接。**零污染現役 TF**。
- 已整合進 `~/rviz_on_laptop.sh`（alias `rviz_rl`）：背景啟動 charge 節點(zenoh/55 同 rviz) + 開 demo.rviz，rviz 關閉時 trap 自動清理。備份 `~/rviz_on_laptop.sh.bak_charge`。**charge 節點必須與 rviz 同 RMW/domain(rmw_zenoh_cpp/55) 才看得到**。
- `demo.rviz` RobotModel Description Topic = `/charge_description`，TF Prefix 留空（原始備份 `~/demo.rviz.bak_charge`）。
- Fixed Frame：真機在線用 `map`；真機沒跑用 `charge_base_link`。

**修過的 URDF 瑕疵**（SolidWorks 原檔就有，母本 `~/Downloads/charge_description/`）：`${3.1415926/2}`→`1.5707963`(2 處)；移除 caster 多餘空 `<mesh/>`；11 個空名 `<material name="">` 給唯一名(charge_mat_N) 否則 RViz 整台白色；全 link 名加 charge_ 前綴生成 charge_prefixed.urdf。

若要讓 charge 取代 campusrover 成真機正式描述 → 需改 14 的 start_transform_launch.py。相關 [[deploy_v3e_rover_rl]]。
