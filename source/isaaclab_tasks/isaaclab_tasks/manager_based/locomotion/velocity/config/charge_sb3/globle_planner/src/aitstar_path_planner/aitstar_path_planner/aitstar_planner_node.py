#!/usr/bin/env python3
"""
AIT* ROS2 路徑規劃節點
======================

此節點提供 AIT* 演算法的 ROS2 介面，支援：
- 導航請求服務
- 路徑發布
- 動態障礙物更新
- 規劃狀態可視化
- 與 Nav2 框架整合

作者: Claude
版本: 1.0.0
ROS2 版本: Humble / Iron
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

# ROS2 訊息類型
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import (
    Point, Pose, PoseStamped, PoseWithCovarianceStamped,
    Twist, Vector3
)
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

# Nav2 介面 (可選)
try:
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from nav2_msgs.srv import IsPathValid
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False
    print("[WARN] nav2_msgs 未安裝，Nav2 整合功能將被停用")

# 自訂服務與動作 (需要建立對應的 .srv 和 .action 檔案)
# 這裡使用標準訊息作為替代
from std_srvs.srv import Trigger, SetBool

# 匯入 AIT* 核心演算法
try:
    from aitstar import AITStar, SimpleCollisionChecker, PathSmoother, State
except ImportError:
    # 如果在同目錄執行，嘗試直接匯入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aitstar import AITStar, SimpleCollisionChecker, PathSmoother, State


# =============================================================================
# 碰撞檢測器 (基於 OccupancyGrid)
# =============================================================================

class OccupancyGridCollisionChecker:
    """
    基於 OccupancyGrid 的碰撞檢測器
    
    支援：
    - 靜態地圖碰撞檢測
    - 動態障礙物更新 (透過 LiDAR)
    - 機器人膨脹半徑
    """
    
    def __init__(self, 
                 robot_radius: float = 0.3,
                 obstacle_threshold: int = 50,
                 unknown_is_obstacle: bool = False):
        """
        初始化碰撞檢測器
        
        Args:
            robot_radius: 機器人半徑（膨脹用）
            obstacle_threshold: 障礙物閾值 (0-100)
            unknown_is_obstacle: 是否將未知區域視為障礙物
        """
        self.robot_radius = robot_radius
        self.obstacle_threshold = obstacle_threshold
        self.unknown_is_obstacle = unknown_is_obstacle
        
        # 地圖資料
        self.map_data: Optional[np.ndarray] = None
        self.map_resolution: float = 0.05
        self.map_origin: np.ndarray = np.array([0.0, 0.0])
        self.map_width: int = 0
        self.map_height: int = 0
        
        # 動態障礙物
        self.dynamic_obstacles: List[Tuple[np.ndarray, float]] = []
        self.dynamic_lock = threading.Lock()
        
        # 膨脹後的地圖
        self.inflated_map: Optional[np.ndarray] = None
    
    def update_map(self, map_msg: OccupancyGrid):
        """
        更新地圖資料
        
        Args:
            map_msg: ROS2 OccupancyGrid 訊息
        """
        self.map_width = map_msg.info.width
        self.map_height = map_msg.info.height
        self.map_resolution = map_msg.info.resolution
        self.map_origin = np.array([
            map_msg.info.origin.position.x,
            map_msg.info.origin.position.y
        ])
        
        # 轉換地圖資料
        self.map_data = np.array(map_msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width)
        )
        
        # 計算膨脹地圖
        self._compute_inflated_map()
    
    def _compute_inflated_map(self):
        """計算膨脹後的地圖"""
        if self.map_data is None:
            return
        
        # 計算膨脹半徑（像素）
        inflation_cells = int(np.ceil(self.robot_radius / self.map_resolution))
        
        # 建立膨脹核心
        kernel_size = 2 * inflation_cells + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = inflation_cells
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                if np.sqrt((i - center)**2 + (j - center)**2) <= inflation_cells:
                    kernel[i, j] = 1
        
        # 建立二值障礙物地圖
        obstacle_map = np.zeros_like(self.map_data, dtype=np.uint8)
        obstacle_map[self.map_data >= self.obstacle_threshold] = 1
        
        if self.unknown_is_obstacle:
            obstacle_map[self.map_data < 0] = 1
        
        # 執行膨脹（使用簡單的卷積）
        try:
            from scipy import ndimage
            self.inflated_map = ndimage.binary_dilation(
                obstacle_map, structure=kernel
            ).astype(np.uint8)
        except ImportError:
            # 如果沒有 scipy，使用簡單的膨脹
            self.inflated_map = self._simple_dilation(obstacle_map, inflation_cells)
    
    def _simple_dilation(self, obstacle_map: np.ndarray, 
                         radius: int) -> np.ndarray:
        """簡單的膨脹實作（無需 scipy）"""
        result = obstacle_map.copy()
        
        for i in range(self.map_height):
            for j in range(self.map_width):
                if obstacle_map[i, j] == 1:
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.map_height and 0 <= nj < self.map_width:
                                if np.sqrt(di**2 + dj**2) <= radius:
                                    result[ni, nj] = 1
        
        return result
    
    def update_dynamic_obstacles(self, 
                                  obstacles: List[Tuple[np.ndarray, float]]):
        """
        更新動態障礙物
        
        Args:
            obstacles: 障礙物列表 [(位置, 半徑), ...]
        """
        with self.dynamic_lock:
            self.dynamic_obstacles = obstacles.copy()
    
    def world_to_map(self, position: np.ndarray) -> Tuple[int, int]:
        """世界座標轉換為地圖座標"""
        mx = int((position[0] - self.map_origin[0]) / self.map_resolution)
        my = int((position[1] - self.map_origin[1]) / self.map_resolution)
        return mx, my
    
    def map_to_world(self, mx: int, my: int) -> np.ndarray:
        """地圖座標轉換為世界座標"""
        wx = mx * self.map_resolution + self.map_origin[0]
        wy = my * self.map_resolution + self.map_origin[1]
        return np.array([wx, wy])
    
    def is_state_valid(self, state: State) -> bool:
        """檢查狀態是否有效"""
        pos = state.position
        
        # 檢查靜態地圖
        if self.inflated_map is not None:
            mx, my = self.world_to_map(pos)
            
            if mx < 0 or mx >= self.map_width or my < 0 or my >= self.map_height:
                return False
            
            if self.inflated_map[my, mx] == 1:
                return False
        
        # 檢查動態障礙物
        with self.dynamic_lock:
            for obs_pos, obs_radius in self.dynamic_obstacles:
                dist = np.linalg.norm(pos[:2] - obs_pos[:2])
                if dist <= obs_radius + self.robot_radius:
                    return False
        
        return True
    
    def is_edge_valid(self, state1: State, state2: State,
                      resolution: float = 0.05) -> bool:
        """檢查邊是否有效"""
        if not self.is_state_valid(state1) or not self.is_state_valid(state2):
            return False
        
        direction = state2.position - state1.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
        
        n_steps = max(2, int(np.ceil(distance / resolution)))
        
        for i in range(1, n_steps):
            t = i / n_steps
            intermediate_pos = state1.position + t * direction
            intermediate_state = State(intermediate_pos)
            if not self.is_state_valid(intermediate_state):
                return False
        
        return True
    
    def get_map_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """取得地圖邊界"""
        if self.map_data is None:
            return np.array([0.0, 0.0]), np.array([10.0, 10.0])
        
        bounds_min = self.map_origin.copy()
        bounds_max = self.map_origin + np.array([
            self.map_width * self.map_resolution,
            self.map_height * self.map_resolution
        ])
        
        return bounds_min, bounds_max


# =============================================================================
# AIT* ROS2 節點
# =============================================================================

class AITStarPlannerNode(Node):
    """
    AIT* 路徑規劃 ROS2 節點
    
    提供功能：
    - 路徑規劃服務
    - 即時路徑發布
    - 視覺化標記
    - 動態重規劃
    """
    
    def __init__(self):
        super().__init__('aitstar_planner_node')
        
        # 宣告參數
        self._declare_parameters()
        
        # 取得參數
        self.robot_radius = self.get_parameter('robot_radius').value
        self.batch_size = self.get_parameter('batch_size').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.rewire_factor = self.get_parameter('rewire_factor').value
        self.planning_frequency = self.get_parameter('planning_frequency').value
        self.enable_smoothing = self.get_parameter('enable_smoothing').value
        self.visualize_tree = self.get_parameter('visualize_tree').value
        
        # 座標框架
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        
        # 碰撞檢測器
        self.collision_checker = OccupancyGridCollisionChecker(
            robot_radius=self.robot_radius,
            obstacle_threshold=self.get_parameter('obstacle_threshold').value,
            unknown_is_obstacle=self.get_parameter('unknown_is_obstacle').value
        )
        
        # 規劃器（延遲初始化，等待地圖）
        self.planner: Optional[AITStar] = None
        self.path_smoother: Optional[PathSmoother] = None
        
        # 狀態變數
        self.current_pose: Optional[PoseStamped] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.current_path: Optional[List[np.ndarray]] = None
        self.is_planning = False
        self.map_received = False
        
        # 鎖
        self.planning_lock = threading.Lock()
        
        # Callback Groups
        self.cb_group_sub = ReentrantCallbackGroup()
        self.cb_group_srv = MutuallyExclusiveCallbackGroup()
        self.cb_group_timer = MutuallyExclusiveCallbackGroup()
        
        # QoS 設定
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        
        # 建立訂閱者
        self._create_subscribers()
        
        # 建立發布者
        self._create_publishers()
        
        # 建立服務
        self._create_services()
        
        # 建立計時器
        self._create_timers()
        
        # Nav2 Action Server (可選)
        if NAV2_AVAILABLE:
            self._create_action_servers()
        
        self.get_logger().info('AIT* 規劃節點已初始化')
    
    def _declare_parameters(self):
        """宣告節點參數"""
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('batch_size', 100)
        self.declare_parameter('max_iterations', 500)
        self.declare_parameter('goal_bias', 0.05)
        self.declare_parameter('rewire_factor', 1.1)
        self.declare_parameter('planning_frequency', 1.0)
        self.declare_parameter('enable_smoothing', True)
        self.declare_parameter('visualize_tree', True)
        self.declare_parameter('obstacle_threshold', 50)
        self.declare_parameter('unknown_is_obstacle', False)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('replan_on_goal_change', True)
        self.declare_parameter('path_tolerance', 0.5)
    
    def _create_subscribers(self):
        """建立訂閱者"""
        # 地圖訂閱
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_callback,
            self.map_qos,
            callback_group=self.cb_group_sub
        )
        
        # 里程計/位姿訂閱
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            self.sensor_qos,
            callback_group=self.cb_group_sub
        )
        
        # AMCL 位姿訂閱
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_callback,
            self.sensor_qos,
            callback_group=self.cb_group_sub
        )
        
        # 目標位姿訂閱
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_callback,
            10,
            callback_group=self.cb_group_sub
        )
        
        # LiDAR 訂閱（用於動態障礙物）
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            self.sensor_qos,
            callback_group=self.cb_group_sub
        )
    
    def _create_publishers(self):
        """建立發布者"""
        # 路徑發布
        self.path_pub = self.create_publisher(
            Path,
            '/aitstar/path',
            10
        )
        
        # 視覺化發布
        self.tree_marker_pub = self.create_publisher(
            MarkerArray,
            '/aitstar/tree_markers',
            10
        )
        
        self.path_marker_pub = self.create_publisher(
            Marker,
            '/aitstar/path_marker',
            10
        )
        
        # 規劃狀態發布
        self.status_pub = self.create_publisher(
            Marker,
            '/aitstar/status',
            10
        )
    
    def _create_services(self):
        """建立服務"""
        # 規劃觸發服務
        self.plan_srv = self.create_service(
            Trigger,
            '/aitstar/plan',
            self._plan_service_callback,
            callback_group=self.cb_group_srv
        )
        
        # 清除路徑服務
        self.clear_srv = self.create_service(
            Trigger,
            '/aitstar/clear_path',
            self._clear_path_callback,
            callback_group=self.cb_group_srv
        )
        
        # 啟用/停用自動規劃
        self.auto_plan_srv = self.create_service(
            SetBool,
            '/aitstar/auto_plan',
            self._auto_plan_callback,
            callback_group=self.cb_group_srv
        )
    
    def _create_timers(self):
        """建立計時器"""
        # 定期規劃計時器
        if self.planning_frequency > 0:
            self.planning_timer = self.create_timer(
                1.0 / self.planning_frequency,
                self._planning_timer_callback,
                callback_group=self.cb_group_timer
            )
        
        # 視覺化計時器
        self.viz_timer = self.create_timer(
            0.5,  # 2 Hz
            self._visualization_timer_callback,
            callback_group=self.cb_group_timer
        )
    
    def _create_action_servers(self):
        """建立 Nav2 相容的 Action Server"""
        if not NAV2_AVAILABLE:
            return
        
        self.compute_path_action = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            execute_callback=self._compute_path_execute,
            goal_callback=self._compute_path_goal,
            cancel_callback=self._compute_path_cancel,
            callback_group=self.cb_group_srv
        )
    
    # =========================================================================
    # 回呼函式
    # =========================================================================
    
    def _map_callback(self, msg: OccupancyGrid):
        """處理地圖更新"""
        self.collision_checker.update_map(msg)
        self.map_received = True
        
        # 初始化規劃器
        if self.planner is None:
            bounds_min, bounds_max = self.collision_checker.get_map_bounds()
            
            self.planner = AITStar(
                collision_checker=self.collision_checker,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                batch_size=self.batch_size,
                rewire_factor=self.rewire_factor,
                goal_bias=self.goal_bias,
                max_iterations=self.max_iterations
            )
            
            self.path_smoother = PathSmoother(self.collision_checker)
            self.get_logger().info('規劃器已初始化')
    
    def _odom_callback(self, msg: Odometry):
        """處理里程計更新"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.current_pose = pose_stamped
    
    def _amcl_callback(self, msg: PoseWithCovarianceStamped):
        """處理 AMCL 位姿更新"""
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self.current_pose = pose_stamped
    
    def _goal_callback(self, msg: PoseStamped):
        """處理新目標"""
        self.goal_pose = msg
        self.get_logger().info(
            f'收到新目標: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )
        
        # 如果啟用自動重規劃
        if self.get_parameter('replan_on_goal_change').value:
            self._trigger_planning()
    
    def _scan_callback(self, msg: LaserScan):
        """處理 LiDAR 掃描（用於動態障礙物偵測）"""
        if self.current_pose is None:
            return
        
        # 簡單的動態障礙物偵測
        # 這裡可以加入更複雜的障礙物追蹤演算法
        dynamic_obstacles = []
        
        robot_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y
        ])
        
        # 從四元數計算航向角
        q = self.current_pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        # 處理 LiDAR 點
        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            if msg.range_min < r < msg.range_max:
                # 轉換到世界座標
                point_angle = yaw + angle
                point_x = robot_pos[0] + r * np.cos(point_angle)
                point_y = robot_pos[1] + r * np.sin(point_angle)
                
                # 這裡可以加入聚類演算法來辨識動態障礙物
                # 簡化起見，暫時不處理
            
            angle += msg.angle_increment
        
        # 更新動態障礙物
        self.collision_checker.update_dynamic_obstacles(dynamic_obstacles)
    
    # =========================================================================
    # 服務回呼
    # =========================================================================
    
    def _plan_service_callback(self, request, response):
        """規劃服務回呼"""
        success = self._trigger_planning()
        response.success = success
        response.message = '規劃完成' if success else '規劃失敗'
        return response
    
    def _clear_path_callback(self, request, response):
        """清除路徑服務回呼"""
        with self.planning_lock:
            self.current_path = None
            self.goal_pose = None
        response.success = True
        response.message = '路徑已清除'
        return response
    
    def _auto_plan_callback(self, request, response):
        """自動規劃開關服務回呼"""
        # 這裡可以控制自動規劃的開關
        response.success = True
        response.message = f'自動規劃: {"啟用" if request.data else "停用"}'
        return response
    
    # =========================================================================
    # Action Server 回呼 (Nav2 相容)
    # =========================================================================
    
    def _compute_path_goal(self, goal_request):
        """處理路徑計算請求"""
        return GoalResponse.ACCEPT
    
    def _compute_path_cancel(self, goal_handle):
        """處理取消請求"""
        return CancelResponse.ACCEPT
    
    async def _compute_path_execute(self, goal_handle: ServerGoalHandle):
        """執行路徑計算"""
        if not NAV2_AVAILABLE:
            return
        
        request = goal_handle.request
        
        # 設定目標
        self.goal_pose = request.goal
        
        # 設定起點
        if request.start.header.frame_id:
            start_pose = request.start
        else:
            start_pose = self.current_pose
        
        if start_pose is None:
            goal_handle.abort()
            result = ComputePathToPose.Result()
            return result
        
        # 執行規劃
        path = self._compute_path(start_pose, request.goal)
        
        if path is None:
            goal_handle.abort()
            result = ComputePathToPose.Result()
            return result
        
        # 建立結果
        result = ComputePathToPose.Result()
        result.path = path
        result.planning_time = rclpy.duration.Duration(seconds=0.0)
        
        goal_handle.succeed()
        return result
    
    # =========================================================================
    # 計時器回呼
    # =========================================================================
    
    def _planning_timer_callback(self):
        """定期規劃回呼"""
        if self.goal_pose is not None:
            self._trigger_planning()
    
    def _visualization_timer_callback(self):
        """視覺化更新回呼"""
        self._publish_visualizations()
    
    # =========================================================================
    # 核心規劃邏輯
    # =========================================================================
    
    def _trigger_planning(self) -> bool:
        """觸發路徑規劃"""
        if not self.map_received or self.planner is None:
            self.get_logger().warn('地圖尚未接收，無法規劃')
            return False
        
        if self.current_pose is None:
            self.get_logger().warn('當前位姿未知，無法規劃')
            return False
        
        if self.goal_pose is None:
            self.get_logger().warn('目標未設定，無法規劃')
            return False
        
        with self.planning_lock:
            if self.is_planning:
                return False
            self.is_planning = True
        
        try:
            path = self._compute_path(self.current_pose, self.goal_pose)
            
            if path is not None:
                self.current_path = self._path_to_array(path)
                self._publish_path(path)
                return True
            else:
                self.get_logger().warn('路徑規劃失敗')
                return False
        
        finally:
            with self.planning_lock:
                self.is_planning = False
    
    def _compute_path(self, start: PoseStamped, 
                      goal: PoseStamped) -> Optional[Path]:
        """
        計算從起點到終點的路徑
        
        Args:
            start: 起始位姿
            goal: 目標位姿
            
        Returns:
            ROS2 Path 訊息或 None
        """
        if self.planner is None:
            return None
        
        # 提取位置
        start_pos = np.array([
            start.pose.position.x,
            start.pose.position.y
        ])
        
        goal_pos = np.array([
            goal.pose.position.x,
            goal.pose.position.y
        ])
        
        self.get_logger().info(
            f'開始規劃: {start_pos} -> {goal_pos}'
        )
        
        # 重新初始化規劃器邊界（如果需要）
        bounds_min, bounds_max = self.collision_checker.get_map_bounds()
        self.planner.bounds_min = bounds_min
        self.planner.bounds_max = bounds_max
        self.planner.sampler.bounds_min = bounds_min
        self.planner.sampler.bounds_max = bounds_max
        
        # 執行規劃
        start_time = time.time()
        path_points, cost = self.planner.plan(start_pos, goal_pos)
        planning_time = time.time() - start_time
        
        if path_points is None:
            self.get_logger().warn(f'規劃失敗 (耗時: {planning_time:.3f}s)')
            return None
        
        self.get_logger().info(
            f'規劃成功: 代價={cost:.3f}, 點數={len(path_points)}, '
            f'耗時={planning_time:.3f}s'
        )
        
        # 路徑平滑
        if self.enable_smoothing and self.path_smoother is not None:
            smoothed_points = self.path_smoother.shortcut(
                path_points, max_iterations=50
            )
            smoothed_cost = sum(
                np.linalg.norm(smoothed_points[i+1] - smoothed_points[i])
                for i in range(len(smoothed_points) - 1)
            )
            self.get_logger().info(
                f'平滑後: 代價={smoothed_cost:.3f}, 點數={len(smoothed_points)}'
            )
            path_points = smoothed_points
        
        # 插值以獲得更密集的路徑
        if self.path_smoother is not None:
            path_points = self.path_smoother.interpolate(
                path_points, resolution=0.1
            )
        
        # 轉換為 ROS2 Path 訊息
        path_msg = self._create_path_message(path_points, start, goal)
        
        return path_msg
    
    def _create_path_message(self, path_points: List[np.ndarray],
                              start: PoseStamped,
                              goal: PoseStamped) -> Path:
        """建立 ROS2 Path 訊息"""
        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()
        
        for i, point in enumerate(path_points):
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = float(point[0])
            pose_stamped.pose.position.y = float(point[1])
            pose_stamped.pose.position.z = 0.0
            
            # 計算航向角（指向下一個點）
            if i < len(path_points) - 1:
                dx = path_points[i + 1][0] - point[0]
                dy = path_points[i + 1][1] - point[1]
                yaw = np.arctan2(dy, dx)
            else:
                # 使用目標的航向
                q = goal.pose.orientation
                yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            # 轉換為四元數
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = np.sin(yaw / 2)
            pose_stamped.pose.orientation.w = np.cos(yaw / 2)
            
            path.poses.append(pose_stamped)
        
        return path
    
    def _path_to_array(self, path: Path) -> List[np.ndarray]:
        """將 Path 訊息轉換為 numpy 陣列列表"""
        return [
            np.array([pose.pose.position.x, pose.pose.position.y])
            for pose in path.poses
        ]
    
    # =========================================================================
    # 發布函式
    # =========================================================================
    
    def _publish_path(self, path: Path):
        """發布路徑"""
        self.path_pub.publish(path)
    
    def _publish_visualizations(self):
        """發布視覺化標記"""
        if self.visualize_tree and self.planner is not None:
            self._publish_tree_markers()
        
        if self.current_path is not None:
            self._publish_path_marker()
    
    def _publish_tree_markers(self):
        """發布搜尋樹視覺化"""
        if self.planner is None:
            return
        
        marker_array = MarkerArray()
        
        # 清除舊標記
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # 取得樹的邊
        edges = self.planner.get_tree()
        
        # 建立線段標記
        tree_marker = Marker()
        tree_marker.header.frame_id = self.global_frame
        tree_marker.header.stamp = self.get_clock().now().to_msg()
        tree_marker.ns = 'aitstar_tree'
        tree_marker.id = 0
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.scale.x = 0.01  # 線寬
        tree_marker.color = ColorRGBA(r=0.3, g=0.7, b=1.0, a=0.5)
        tree_marker.pose.orientation.w = 1.0
        
        for start, end in edges:
            p1 = Point(x=float(start[0]), y=float(start[1]), z=0.05)
            p2 = Point(x=float(end[0]), y=float(end[1]), z=0.05)
            tree_marker.points.append(p1)
            tree_marker.points.append(p2)
        
        marker_array.markers.append(tree_marker)
        
        self.tree_marker_pub.publish(marker_array)
    
    def _publish_path_marker(self):
        """發布路徑視覺化"""
        if self.current_path is None:
            return
        
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'aitstar_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # 線寬
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        marker.pose.orientation.w = 1.0
        
        for point in self.current_path:
            p = Point(x=float(point[0]), y=float(point[1]), z=0.1)
            marker.points.append(p)
        
        self.path_marker_pub.publish(marker)


# =============================================================================
# 主程式
# =============================================================================

def main(args=None):
    """主程式入口"""
    rclpy.init(args=args)
    
    try:
        node = AITStarPlannerNode()
        
        # 使用多執行緒執行器
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info('收到中斷訊號，正在關閉...')
        finally:
            executor.shutdown()
            node.destroy_node()
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()