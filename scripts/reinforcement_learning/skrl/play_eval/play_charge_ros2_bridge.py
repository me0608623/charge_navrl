"""Isaac Sim ROS2 Bridge — 用 IsaacLab 環境發布感測器資料到 ROS2。

將訓練好的 Charge VLP-16 環境作為 ROS2 模擬伺服器：
  - 發布 /odom (nav_msgs/Odometry)
  - 發布 /velodyne_points (sensor_msgs/PointCloud2)
  - 發布 /tf (tf2_msgs/TFMessage)
  - 發布 /tracked_label_obstacle (campusrover_msgs/TrackedObstacleArray)
  - 訂閱 /cmd_vel (geometry_msgs/Twist)

使用方法:
    # 基本啟動 (3floor 場景, 靜態+動態障礙物)
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ros2_bridge.py

    # 指定場景 USD
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ros2_bridge.py \\
        --terrain_usd /home/aa/usd/charge/3floor_ver_1.usd

    # 自訂障礙物
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_ros2_bridge.py \\
        --num_static 5 --num_dynamic 3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Isaac Sim ROS2 Bridge for Charge VLP-16.")
parser.add_argument("--task", type=str,
                    default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL",
                    help="Task name.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (should be 1 for ROS bridge).")
parser.add_argument("--terrain_usd", type=str,
                    default="/home/aa/usd/charge/3floor_ver_1.usd",
                    help="Path to terrain USD file. Use 'plane' for flat ground.")
parser.add_argument("--num_static", type=int, default=5,
                    help="Number of static obstacles.")
parser.add_argument("--num_dynamic", type=int, default=3,
                    help="Number of dynamic obstacles.")
parser.add_argument("--num_walls", type=int, default=0,
                    help="Number of internal walls (0 for 3floor scene which has its own walls).")
parser.add_argument("--camera", type=str, default="follow",
                    choices=["top", "follow", "side"],
                    help="Camera view.")
parser.add_argument("--ros_domain_id", type=int, default=None,
                    help="ROS_DOMAIN_ID (default: use env variable).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Render settings for bridge mode
if not args_cli.headless:
    if args_cli.rendering_mode is None or args_cli.rendering_mode == "balanced":
        args_cli.rendering_mode = "performance"
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401

# ROS2 imports
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
if args_cli.ros_domain_id is not None:
    os.environ['ROS_DOMAIN_ID'] = str(args_cli.ros_domain_id)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import (
    Twist, TransformStamped, Quaternion, Vector3,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time as TimeMsg
from campusrover_msgs.msg import TrackedObstacle, TrackedObstacleArray


# ============================================================================
# ROS2 Bridge Node
# ============================================================================
class IsaacSimBridge(Node):
    """ROS2 node that publishes Isaac Sim sensor data."""

    def __init__(self):
        super().__init__('isaac_sim_bridge')

        # Latest cmd_vel from subscriber
        self.cmd_linear_x = 0.0
        self.cmd_angular_z = 0.0
        self.cmd_received = False

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/velodyne_points', sensor_qos)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', tf_qos)
        self.obs_pub = self.create_publisher(
            TrackedObstacleArray, '/tracked_label_obstacle', 10)

        # Subscriber
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        self.get_logger().info('Isaac Sim ROS2 Bridge initialized')

    def _cmd_vel_cb(self, msg: Twist):
        self.cmd_linear_x = msg.linear.x
        self.cmd_angular_z = msg.angular.z
        self.cmd_received = True

    def publish_odom(self, pos_w, quat_w, lin_vel_w, ang_vel_w, sim_time: float):
        """Publish odometry from Isaac Sim robot state."""
        msg = Odometry()
        msg.header.stamp = self._to_ros_time(sim_time)
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Position
        msg.pose.pose.position.x = float(pos_w[0])
        msg.pose.pose.position.y = float(pos_w[1])
        msg.pose.pose.position.z = float(pos_w[2])

        # Orientation
        msg.pose.pose.orientation.w = float(quat_w[0])
        msg.pose.pose.orientation.x = float(quat_w[1])
        msg.pose.pose.orientation.y = float(quat_w[2])
        msg.pose.pose.orientation.z = float(quat_w[3])

        # Twist (body frame)
        # Convert world velocity to body frame
        w, x, y, z = float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])
        # Inverse quaternion rotation for body-frame velocity
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_w = float(lin_vel_w[0])
        vy_w = float(lin_vel_w[1])
        msg.twist.twist.linear.x = cos_yaw * vx_w + sin_yaw * vy_w
        msg.twist.twist.linear.y = -sin_yaw * vx_w + cos_yaw * vy_w
        msg.twist.twist.angular.z = float(ang_vel_w[2])

        self.odom_pub.publish(msg)

        # Publish TF: map -> odom (identity) and odom -> base_link
        tf_msg = TFMessage()
        stamp = msg.header.stamp

        # map -> odom (identity transform)
        t_map = TransformStamped()
        t_map.header.stamp = stamp
        t_map.header.frame_id = 'map'
        t_map.child_frame_id = 'odom'
        t_map.transform.rotation.w = 1.0
        tf_msg.transforms.append(t_map)

        # odom -> base_link
        t_odom = TransformStamped()
        t_odom.header.stamp = stamp
        t_odom.header.frame_id = 'odom'
        t_odom.child_frame_id = 'base_link'
        t_odom.transform.translation.x = msg.pose.pose.position.x
        t_odom.transform.translation.y = msg.pose.pose.position.y
        t_odom.transform.translation.z = msg.pose.pose.position.z
        t_odom.transform.rotation = msg.pose.pose.orientation
        tf_msg.transforms.append(t_odom)

        # base_link -> velodyne (static offset: z=1.6m from env config)
        t_lidar = TransformStamped()
        t_lidar.header.stamp = stamp
        t_lidar.header.frame_id = 'base_link'
        t_lidar.child_frame_id = 'velodyne'
        t_lidar.transform.translation.z = 1.6
        t_lidar.transform.rotation.w = 1.0
        tf_msg.transforms.append(t_lidar)

        self.tf_pub.publish(tf_msg)

    def publish_pointcloud(self, ray_hits_w, sensor_pos_w, sim_time: float):
        """Publish VLP-16 ray hits as PointCloud2.

        Args:
            ray_hits_w: [total_rays, 3] tensor of hit points in world frame
            sensor_pos_w: [3] sensor position in world frame
            sim_time: simulation time
        """
        # Convert to sensor-local frame (subtract sensor position)
        points = ray_hits_w.cpu().numpy().astype(np.float32)
        sensor_pos = sensor_pos_w.cpu().numpy().astype(np.float32)
        points_local = points - sensor_pos[np.newaxis, :]

        # Filter out inf/nan and very far points
        valid = np.isfinite(points_local).all(axis=1)
        dist = np.linalg.norm(points_local, axis=1)
        valid &= dist < 20.0
        points_local = points_local[valid]

        n_points = len(points_local)

        # Build PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = self._to_ros_time(sim_time)
        msg.header.frame_id = 'velodyne'

        msg.height = 1
        msg.width = n_points
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * float32
        msg.row_step = msg.point_step * n_points

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.data = points_local.tobytes()
        self.pc_pub.publish(msg)

    def publish_obstacles(self, env, sim_time: float, max_obstacles: int = 100):
        """Publish ground-truth obstacle data as TrackedObstacleArray.

        Follows IsaacLab convention: obstacles named obstacle_0..obstacle_N,
        hidden obstacles at z=-10.
        """
        msg = TrackedObstacleArray()
        msg.header.stamp = self._to_ros_time(sim_time)
        msg.header.frame_id = 'odom'

        # Get obstacle sizes from env cache if available
        sizes_raw = getattr(env, "_obstacle_sizes", None)
        vel_cache = getattr(env, "_obstacle_velocities", None)

        obs_id = 0
        for i in range(max_obstacles):
            obstacle_name = f"obstacle_{i}"
            if obstacle_name not in env.scene.keys():
                continue

            entity = env.scene[obstacle_name]
            pos = entity.data.root_pos_w[0]  # [3] (single env)

            # Skip hidden obstacles (z < 0 means hidden at z=-10)
            if float(pos[2]) < 0.0:
                continue

            tracked = TrackedObstacle()
            tracked.id = obs_id

            tracked.pose.position.x = float(pos[0])
            tracked.pose.position.y = float(pos[1])
            tracked.pose.position.z = float(pos[2])

            # Velocity: prefer cached velocity (from move_obstacles_vectorized)
            if vel_cache is not None and obs_id < vel_cache.shape[1]:
                tracked.velocity.linear.x = float(vel_cache[0, obs_id, 0])
                tracked.velocity.linear.y = float(vel_cache[0, obs_id, 1])
            else:
                vel = entity.data.root_lin_vel_w[0]
                tracked.velocity.linear.x = float(vel[0])
                tracked.velocity.linear.y = float(vel[1])

            # Size: prefer cached sizes
            if sizes_raw is not None and obs_id < len(sizes_raw):
                diameter = float(sizes_raw[obs_id]) * 2.0
            else:
                diameter = 0.6  # default
            tracked.dimensions.x = diameter
            tracked.dimensions.y = diameter

            speed = math.sqrt(
                tracked.velocity.linear.x ** 2 + tracked.velocity.linear.y ** 2
            )
            tracked.is_dynamic = speed > 0.05
            tracked.label = 1  # LABEL_PERSON

            msg.obstacles.append(tracked)
            obs_id += 1

        self.obs_pub.publish(msg)

    @staticmethod
    def _to_ros_time(sim_time: float) -> TimeMsg:
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        return TimeMsg(sec=sec, nanosec=nanosec)


# ============================================================================
# Main
# ============================================================================
def main():
    # --- Load env config ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Disable curriculum
    env_cfg.curriculum = None

    # ── Terrain: use 3floor USD or plane ──
    if args_cli.terrain_usd and args_cli.terrain_usd.lower() != 'plane':
        usd_path = args_cli.terrain_usd
        if os.path.exists(usd_path):
            env_cfg.scene.terrain.terrain_type = "usd"
            env_cfg.scene.terrain.usd_path = usd_path
            env_cfg.scene.terrain.env_spacing = 0.0  # single env, no grid spacing
            print(f"[TERRAIN] Using USD: {usd_path}")
        else:
            print(f"[WARN] USD not found: {usd_path}, using plane")

    # ── Obstacle configuration ──
    n_s = args_cli.num_static
    n_d = args_cli.num_dynamic
    n_w = args_cli.num_walls

    if n_s > 0 and n_d > 0:
        obs_params = {
            "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
            "mixed_ratio": 1.0,
            "num_obstacles_static": n_s, "num_obstacles_dynamic": n_d,
        }
    elif n_d > 0:
        obs_params = {
            "empty_ratio": 0.0, "static_ratio": 0.0, "dynamic_ratio": 1.0,
            "num_obstacles_static": 0, "num_obstacles_dynamic": n_d,
        }
    elif n_s > 0:
        obs_params = {
            "empty_ratio": 0.0, "static_ratio": 1.0, "dynamic_ratio": 0.0,
            "num_obstacles_static": n_s, "num_obstacles_dynamic": 0,
        }
    else:
        obs_params = {
            "empty_ratio": 1.0, "static_ratio": 0.0, "dynamic_ratio": 0.0,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
        }

    max_obs = max(n_s + n_d, 10)
    obs_params["max_obstacles"] = max_obs
    for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
        evt_term = getattr(env_cfg.events, evt_attr, None)
        if evt_term is not None:
            evt_term.params.update(obs_params)

    move_evt = getattr(env_cfg.events, "move_dynamic_obstacles", None)
    if move_evt is not None:
        move_evt.params["max_obstacles"] = max_obs

    wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_evt is not None:
        wall_evt.params["min_walls"] = n_w
        wall_evt.params["max_walls"] = n_w

    env_cfg.commands.goal_command.num_goals = 1
    env_cfg.commands.goal_command.num_obstacles = n_s + n_d
    env_cfg.commands.goal_command.ranges.distance = (2.0, 14.0)
    env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)

    # ── Camera ──
    from isaaclab.envs import ViewerCfg
    if args_cli.camera == "top":
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 30.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )
    elif args_cli.camera == "follow":
        env_cfg.viewer = ViewerCfg(
            eye=(-3.0, 0.0, 3.0), lookat=(2.0, 0.0, 0.0),
            origin_type="asset_root", env_index=0, asset_name="robot",
            resolution=(1920, 1080),
        )
    elif args_cli.camera == "side":
        env_cfg.viewer = ViewerCfg(
            eye=(15.0, -15.0, 15.0), lookat=(0.0, 0.0, 0.0),
            origin_type="env", env_index=0, resolution=(1920, 1080),
        )

    # Render every frame for smoother visualization
    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    # ── Create environment ──
    print(f"\n{'='*70}")
    print(f"  Isaac Sim ROS2 Bridge")
    print(f"{'='*70}")
    print(f"  Terrain:   {args_cli.terrain_usd}")
    print(f"  Static:    {n_s}")
    print(f"  Dynamic:   {n_d}")
    print(f"  Walls:     {n_w}")
    print(f"{'='*70}\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped
    while hasattr(raw_env, "unwrapped") and raw_env is not raw_env.unwrapped:
        raw_env = raw_env.unwrapped

    # ── Initialize ROS2 ──
    rclpy.init()
    bridge = IsaacSimBridge()

    # ── Get references ──
    robot = raw_env.scene["robot"]
    lidar_sensor = None
    for sensor_name in raw_env.scene.sensors:
        if 'lidar' in sensor_name.lower():
            lidar_sensor = raw_env.scene.sensors[sensor_name]
            break

    if lidar_sensor is None:
        print("[WARN] No LiDAR sensor found in scene!")

    # ── Action conversion ──
    # DifferentialDriveAction expects continuous [-1, 1] for [linear, angular].
    # We convert cmd_vel to this range using the action term's max velocity.
    # Read max velocities from the action term config.
    action_term = None
    for term in raw_env.action_manager._terms.values():
        if hasattr(term, 'cfg') and hasattr(term.cfg, 'max_linear_velocity'):
            action_term = term
            break
    if action_term is not None:
        max_lin_vel = action_term.cfg.max_linear_velocity
        max_ang_vel = action_term.cfg.max_angular_velocity
        print(f"[ROS2 BRIDGE] Action term found: max_lin={max_lin_vel}, max_ang={max_ang_vel}")
    else:
        max_lin_vel = 1.5
        max_ang_vel = 1.5
        print(f"[ROS2 BRIDGE] No action term found, using defaults: max_lin={max_lin_vel}, max_ang={max_ang_vel}")

    cmd_action = torch.tensor([[0.0, 0.0]], device=raw_env.device)

    # ── Main simulation loop ──
    print("[ROS2 BRIDGE] Starting simulation loop... (Ctrl+C to stop)")
    obs, _ = env.reset()

    sim_time = 0.0
    step_dt = raw_env.step_dt

    try:
        while simulation_app.is_running():
            # Process ROS2 callbacks
            rclpy.spin_once(bridge, timeout_sec=0)

            # Apply cmd_vel by converting to action term's expected format
            if bridge.cmd_received:
                # Convert cmd_vel to continuous [-1, 1] action for DifferentialDriveAction
                cmd_action[0, 0] = max(-1.0, min(1.0, bridge.cmd_linear_x / max_lin_vel))
                cmd_action[0, 1] = max(-1.0, min(1.0, bridge.cmd_angular_z / max_ang_vel))

            # Step environment — action term handles velocity application
            obs, reward, terminated, truncated, infos = env.step(cmd_action)

            sim_time += step_dt

            # ── Publish sensor data ──
            # Odometry
            pos_w = robot.data.root_pos_w[0]
            quat_w = robot.data.root_quat_w[0]
            lin_vel_w = robot.data.root_lin_vel_w[0]
            ang_vel_w = robot.data.root_ang_vel_w[0]
            bridge.publish_odom(pos_w, quat_w, lin_vel_w, ang_vel_w, sim_time)

            # PointCloud2
            if lidar_sensor is not None:
                ray_hits = lidar_sensor.data.ray_hits_w[0]  # [total_rays, 3]
                sensor_pos = lidar_sensor.data.pos_w[0]     # [3]
                bridge.publish_pointcloud(ray_hits, sensor_pos, sim_time)

            # Ground-truth obstacles
            bridge.publish_obstacles(raw_env, sim_time)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        # Cleanup
        bridge.destroy_node()
        rclpy.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
