# Play / Eval — 回放與評估腳本

用訓練好的 checkpoint 進行展示、評估與部署。

| 檔案 | 功能 |
|------|------|
| `play_charge.py` | 通用回放腳本。支援 Frame Stacking 架構的 checkpoint 載入與 GUI 展示 |
| `play_charge_ac_curriculum.py` | 課程逐階段展示。自動從 Stage 1→8 每階段 200 steps，展示 agent 在不同難度下的表現 |
| `play_charge_dynamic_test.py` | 動態障礙物避障能力測試。驗證 agent 在高密度動態場景的真實避障能力 |
| `play_charge_ros2_bridge.py` | ROS2 Bridge 部署。將 IsaacLab 環境作為 ROS2 模擬伺服器，發布 /odom, /velodyne_points, /tf, /tracked_label_obstacle，訂閱 /cmd_vel |
| `play_phase0.py` | Phase 0 評估。訓練完成後的統計指標輸出，支援多種評估模式 |
| `play_rnn_car.py` | RNN Car checkpoint 回放。載入 modular-RNN 模型在 GUI 中展示，預設從 logs/rnn_car 找最新 checkpoint |
