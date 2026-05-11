#!/bin/bash
# Launch Isaac Sim (conda env_isaaclab) with MCP Extension + rosbridge
# Usage:
#   ./isaacrunmcp.sh
#   ./isaacrunmcp.sh --open-usd /home/aa/charge_rl/assets/3F/3floor_clean.usd

MCP_EXT_FOLDER="/home/aa/Documents/isaac-sim-mcp"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/aa/miniconda3)"

# Isaac Sim 內建 jazzy lib 路徑（Python 3.11 相容版 rclpy）
ISAAC_JAZZY_LIB="/home/aa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"

echo "=============================="
echo " Isaac Sim + MCP Extension"
echo " MCP server  : localhost:8766"
echo " rosbridge   : localhost:9090"
echo ""
echo " 首次載入場景後，透過 MCP 執行："
echo "   fix_velodyne_qos.py  (設 VOLATILE QoS 並存檔)"
echo "=============================="

# 啟動 rosbridge（系統 Python 3.12，獨立 subshell 避免 conda 污染）
bash -c '
    unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV
    source /opt/ros/jazzy/setup.bash
    source /home/aa/Documents/jazzy_ws/install/setup.bash 2>/dev/null || true
    exec /usr/bin/python3 /opt/ros/jazzy/lib/rosbridge_server/rosbridge_websocket
' &
ROSBRIDGE_PID=$!
echo "[rosbridge] started (pid=$ROSBRIDGE_PID)"

# 確保退出時一起關掉 rosbridge
trap "echo '[rosbridge] stopping...'; kill $ROSBRIDGE_PID 2>/dev/null" EXIT

# 設定 Isaac Sim 使用內建 jazzy rclpy（Python 3.11 相容）
ISAAC_JAZZY_RCLPY="/home/aa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/rclpy"
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH="${ISAAC_JAZZY_LIB}:${LD_LIBRARY_PATH}"
export PYTHONPATH="${ISAAC_JAZZY_RCLPY}:${PYTHONPATH}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate env_isaaclab

isaacsim \
    --ext-folder "${MCP_EXT_FOLDER}" \
    --enable isaac.sim.mcp_extension \
    "$@"
