#!/usr/bin/env python3
"""
AIT* 路徑規劃可視化節點
========================

提供 AIT* 規劃演算法的即時可視化功能，包含：
- 搜尋樹可視化
- 路徑可視化
- 規劃統計資訊顯示
- 互動式標記控制
- RViz2 整合

作者: Claude
版本: 1.0.0
ROS2 版本: Humble / Iron
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import threading
import time
import colorsys

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ROS2 訊息類型
from std_msgs.msg import Header, ColorRGBA, String, Float32, Int32
from geometry_msgs.msg import Point, Pose, PoseStamped, Vector3
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import SetBool, Trigger

# 嘗試匯入 interactive markers
try:
    from interactive_markers import InteractiveMarkerServer
    INTERACTIVE_MARKERS_AVAILABLE = True
except ImportError:
    INTERACTIVE_MARKERS_AVAILABLE = False
    print("[WARN] interactive_markers 未安裝，互動標記功能將被停用")


# =============================================================================
# 顏色工具
# =============================================================================

class ColorScheme(Enum):
    """顏色方案"""
    DEFAULT = "default"
    COST_GRADIENT = "cost_gradient"
    DEPTH_GRADIENT = "depth_gradient"
    RAINBOW = "rainbow"


def create_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    """建立 ColorRGBA"""
    return ColorRGBA(r=r, g=g, b=b, a=a)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """HSV 轉 RGB"""
    return colorsys.hsv_to_rgb(h, s, v)


def cost_to_color(cost: float, min_cost: float, max_cost: float, 
                  alpha: float = 0.8) -> ColorRGBA:
    """
    根據代價值生成顏色（綠色=低代價，紅色=高代價）
    """
    if max_cost <= min_cost:
        return create_color(0.0, 1.0, 0.0, alpha)
    
    normalized = (cost - min_cost) / (max_cost - min_cost)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # 從綠色 (0.33) 到紅色 (0.0) 的 HSV 漸變
    hue = 0.33 * (1.0 - normalized)
    r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
    
    return create_color(r, g, b, alpha)


def depth_to_color(depth: int, max_depth: int, alpha: float = 0.8) -> ColorRGBA:
    """
    根據樹深度生成顏色（彩虹漸變）
    """
    if max_depth <= 0:
        return create_color(0.0, 0.5, 1.0, alpha)
    
    hue = (depth / max_depth) * 0.8  # 0 到 0.8 避免回到紅色
    r, g, b = hsv_to_rgb(hue, 0.8, 1.0)
    
    return create_color(r, g, b, alpha)


# =============================================================================
# 可視化資料結構
# =============================================================================

@dataclass
class TreeEdge:
    """樹邊資料"""
    start: np.ndarray
    end: np.ndarray
    cost: float = 0.0
    depth: int = 0


@dataclass
class VisualizationConfig:
    """可視化配置"""
    # 樹可視化
    show_tree: bool = True
    tree_line_width: float = 0.02
    tree_alpha: float = 0.6
    tree_color_scheme: ColorScheme = ColorScheme.DEFAULT
    tree_default_color: ColorRGBA = None
    
    # 路徑可視化
    show_path: bool = True
    path_line_width: float = 0.08
    path_alpha: float = 0.9
    path_color: ColorRGBA = None
    show_path_points: bool = True
    path_point_size: float = 0.12
    
    # 頂點可視化
    show_vertices: bool = False
    vertex_size: float = 0.05
    vertex_alpha: float = 0.5
    
    # 起點/終點可視化
    show_start_goal: bool = True
    start_color: ColorRGBA = None
    goal_color: ColorRGBA = None
    marker_size: float = 0.3
    
    # 採樣區域可視化
    show_sampling_region: bool = False
    sampling_region_alpha: float = 0.1
    
    # 統計資訊
    show_stats: bool = True
    stats_position: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    
    def __post_init__(self):
        if self.tree_default_color is None:
            self.tree_default_color = create_color(0.3, 0.7, 1.0, self.tree_alpha)
        if self.path_color is None:
            self.path_color = create_color(0.0, 1.0, 0.0, self.path_alpha)
        if self.start_color is None:
            self.start_color = create_color(0.0, 1.0, 0.0, 1.0)
        if self.goal_color is None:
            self.goal_color = create_color(1.0, 0.0, 0.0, 1.0)


# =============================================================================
# AIT* 可視化節點
# =============================================================================

class AITStarVisualizerNode(Node):
    """
    AIT* 路徑規劃可視化節點
    
    訂閱規劃器的資料並提供豐富的可視化功能
    """
    
    def __init__(self):
        super().__init__('aitstar_visualizer_node')
        
        # 宣告參數
        self._declare_parameters()
        
        # 取得參數
        self.global_frame = self.get_parameter('global_frame').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # 可視化配置
        self.config = VisualizationConfig(
            show_tree=self.get_parameter('show_tree').value,
            tree_line_width=self.get_parameter('tree_line_width').value,
            tree_alpha=self.get_parameter('tree_alpha').value,
            show_path=self.get_parameter('show_path').value,
            path_line_width=self.get_parameter('path_line_width').value,
            show_vertices=self.get_parameter('show_vertices').value,
            show_start_goal=self.get_parameter('show_start_goal').value,
            show_sampling_region=self.get_parameter('show_sampling_region').value,
            show_stats=self.get_parameter('show_stats').value,
        )
        
        # 資料儲存
        self.tree_edges: List[TreeEdge] = []
        self.path_points: List[np.ndarray] = []
        self.vertices: List[np.ndarray] = []
        self.start_pose: Optional[np.ndarray] = None
        self.goal_pose: Optional[np.ndarray] = None
        self.sampling_ellipse: Optional[Dict] = None
        self.planning_stats: Dict = {}
        
        # 鎖
        self.data_lock = threading.Lock()
        
        # Callback Group
        self.cb_group = ReentrantCallbackGroup()
        
        # QoS 設定
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 建立訂閱者
        self._create_subscribers()
        
        # 建立發布者
        self._create_publishers()
        
        # 建立服務
        self._create_services()
        
        # 建立計時器
        self._create_timers()
        
        # 互動式標記伺服器
        if INTERACTIVE_MARKERS_AVAILABLE:
            self._setup_interactive_markers()
        
        # 標記 ID 計數器
        self.marker_id_counter = 0
        
        self.get_logger().info('AIT* 可視化節點已初始化')
    
    def _declare_parameters(self):
        """宣告節點參數"""
        # 座標框架
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('update_rate', 10.0)
        
        # 樹可視化
        self.declare_parameter('show_tree', True)
        self.declare_parameter('tree_line_width', 0.02)
        self.declare_parameter('tree_alpha', 0.6)
        self.declare_parameter('tree_color_scheme', 'default')
        
        # 路徑可視化
        self.declare_parameter('show_path', True)
        self.declare_parameter('path_line_width', 0.08)
        
        # 頂點可視化
        self.declare_parameter('show_vertices', False)
        
        # 起點/終點
        self.declare_parameter('show_start_goal', True)
        
        # 採樣區域
        self.declare_parameter('show_sampling_region', False)
        
        # 統計資訊
        self.declare_parameter('show_stats', True)
    
    def _create_subscribers(self):
        """建立訂閱者"""
        # 路徑訂閱
        self.path_sub = self.create_subscription(
            Path,
            '/aitstar/path',
            self._path_callback,
            self.reliable_qos,
            callback_group=self.cb_group
        )
        
        # 樹標記訂閱（從規劃節點）
        self.tree_sub = self.create_subscription(
            MarkerArray,
            '/aitstar/tree_markers',
            self._tree_markers_callback,
            10,
            callback_group=self.cb_group
        )
        
        # 目標位姿訂閱
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self._goal_callback,
            10,
            callback_group=self.cb_group
        )
        
        # 起始位姿訂閱
        self.start_sub = self.create_subscription(
            PoseStamped,
            '/initialpose',
            self._start_callback,
            10,
            callback_group=self.cb_group
        )
        
        # 規劃統計訂閱
        self.stats_sub = self.create_subscription(
            String,
            '/aitstar/stats',
            self._stats_callback,
            10,
            callback_group=self.cb_group
        )
        
        # 頂點資料訂閱（自訂格式）
        self.vertices_sub = self.create_subscription(
            PointCloud2,
            '/aitstar/vertices',
            self._vertices_callback,
            10,
            callback_group=self.cb_group
        )
        
        # 採樣橢球資料訂閱
        self.ellipse_sub = self.create_subscription(
            Marker,
            '/aitstar/sampling_ellipse',
            self._ellipse_callback,
            10,
            callback_group=self.cb_group
        )
    
    def _create_publishers(self):
        """建立發布者"""
        # 主要可視化發布
        self.viz_markers_pub = self.create_publisher(
            MarkerArray,
            '/aitstar/visualization',
            10
        )
        
        # 路徑可視化
        self.path_viz_pub = self.create_publisher(
            Marker,
            '/aitstar/path_visualization',
            10
        )
        
        # 統計文字
        self.stats_text_pub = self.create_publisher(
            Marker,
            '/aitstar/stats_text',
            10
        )
        
        # 進度指示
        self.progress_pub = self.create_publisher(
            Float32,
            '/aitstar/planning_progress',
            10
        )
    
    def _create_services(self):
        """建立服務"""
        # 開關可視化
        self.toggle_tree_srv = self.create_service(
            SetBool,
            '/aitstar_viz/toggle_tree',
            self._toggle_tree_callback
        )
        
        self.toggle_path_srv = self.create_service(
            SetBool,
            '/aitstar_viz/toggle_path',
            self._toggle_path_callback
        )
        
        self.toggle_vertices_srv = self.create_service(
            SetBool,
            '/aitstar_viz/toggle_vertices',
            self._toggle_vertices_callback
        )
        
        # 清除所有標記
        self.clear_srv = self.create_service(
            Trigger,
            '/aitstar_viz/clear',
            self._clear_callback
        )
        
        # 截圖服務（記錄當前狀態）
        self.snapshot_srv = self.create_service(
            Trigger,
            '/aitstar_viz/snapshot',
            self._snapshot_callback
        )
    
    def _create_timers(self):
        """建立計時器"""
        self.viz_timer = self.create_timer(
            1.0 / self.update_rate,
            self._visualization_timer_callback
        )
    
    def _setup_interactive_markers(self):
        """設定互動式標記"""
        if not INTERACTIVE_MARKERS_AVAILABLE:
            return
        
        # 這裡可以加入互動式起點/終點設定
        pass
    
    # =========================================================================
    # 訂閱回呼
    # =========================================================================
    
    def _path_callback(self, msg: Path):
        """處理路徑更新"""
        with self.data_lock:
            self.path_points = [
                np.array([pose.pose.position.x, 
                         pose.pose.position.y,
                         pose.pose.position.z])
                for pose in msg.poses
            ]
    
    def _tree_markers_callback(self, msg: MarkerArray):
        """處理樹標記更新"""
        # 從標記中提取邊資料
        with self.data_lock:
            self.tree_edges.clear()
            
            for marker in msg.markers:
                if marker.type == Marker.LINE_LIST and len(marker.points) >= 2:
                    for i in range(0, len(marker.points) - 1, 2):
                        start = np.array([
                            marker.points[i].x,
                            marker.points[i].y,
                            marker.points[i].z
                        ])
                        end = np.array([
                            marker.points[i + 1].x,
                            marker.points[i + 1].y,
                            marker.points[i + 1].z
                        ])
                        self.tree_edges.append(TreeEdge(start=start, end=end))
    
    def _goal_callback(self, msg: PoseStamped):
        """處理目標位姿"""
        with self.data_lock:
            self.goal_pose = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
    
    def _start_callback(self, msg: PoseStamped):
        """處理起始位姿"""
        with self.data_lock:
            self.start_pose = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
    
    def _stats_callback(self, msg: String):
        """處理統計資訊"""
        import json
        try:
            with self.data_lock:
                self.planning_stats = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('無法解析統計資訊 JSON')
    
    def _vertices_callback(self, msg: PointCloud2):
        """處理頂點資料"""
        # 解析 PointCloud2 訊息
        # 這裡需要根據實際格式進行解析
        pass
    
    def _ellipse_callback(self, msg: Marker):
        """處理採樣橢球"""
        with self.data_lock:
            self.sampling_ellipse = {
                'center': np.array([msg.pose.position.x,
                                   msg.pose.position.y,
                                   msg.pose.position.z]),
                'scale': np.array([msg.scale.x, msg.scale.y, msg.scale.z]),
                'orientation': msg.pose.orientation
            }
    
    # =========================================================================
    # 服務回呼
    # =========================================================================
    
    def _toggle_tree_callback(self, request, response):
        """開關樹可視化"""
        self.config.show_tree = request.data
        response.success = True
        response.message = f'樹可視化: {"開啟" if request.data else "關閉"}'
        return response
    
    def _toggle_path_callback(self, request, response):
        """開關路徑可視化"""
        self.config.show_path = request.data
        response.success = True
        response.message = f'路徑可視化: {"開啟" if request.data else "關閉"}'
        return response
    
    def _toggle_vertices_callback(self, request, response):
        """開關頂點可視化"""
        self.config.show_vertices = request.data
        response.success = True
        response.message = f'頂點可視化: {"開啟" if request.data else "關閉"}'
        return response
    
    def _clear_callback(self, request, response):
        """清除所有標記"""
        with self.data_lock:
            self.tree_edges.clear()
            self.path_points.clear()
            self.vertices.clear()
            self.planning_stats.clear()
        
        # 發布刪除標記
        self._publish_delete_all()
        
        response.success = True
        response.message = '已清除所有標記'
        return response
    
    def _snapshot_callback(self, request, response):
        """記錄當前狀態快照"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        with self.data_lock:
            snapshot = {
                'timestamp': timestamp,
                'tree_edges': len(self.tree_edges),
                'path_points': len(self.path_points),
                'stats': self.planning_stats.copy()
            }
        
        self.get_logger().info(f'快照已記錄: {snapshot}')
        
        response.success = True
        response.message = f'快照已記錄於 {timestamp}'
        return response
    
    # =========================================================================
    # 可視化計時器
    # =========================================================================
    
    def _visualization_timer_callback(self):
        """定期更新可視化"""
        marker_array = MarkerArray()
        
        # 取得當前資料的副本
        with self.data_lock:
            tree_edges = self.tree_edges.copy()
            path_points = self.path_points.copy()
            start_pose = self.start_pose.copy() if self.start_pose is not None else None
            goal_pose = self.goal_pose.copy() if self.goal_pose is not None else None
            stats = self.planning_stats.copy()
            sampling_ellipse = self.sampling_ellipse.copy() if self.sampling_ellipse else None
        
        # 重置標記 ID
        self.marker_id_counter = 0
        
        # 添加刪除所有舊標記
        delete_marker = self._create_delete_all_marker()
        marker_array.markers.append(delete_marker)
        
        # 樹可視化
        if self.config.show_tree and tree_edges:
            tree_markers = self._create_tree_markers(tree_edges)
            marker_array.markers.extend(tree_markers)
        
        # 路徑可視化
        if self.config.show_path and path_points:
            path_markers = self._create_path_markers(path_points)
            marker_array.markers.extend(path_markers)
        
        # 起點/終點可視化
        if self.config.show_start_goal:
            if start_pose is not None:
                start_marker = self._create_pose_marker(
                    start_pose, self.config.start_color, 'start'
                )
                marker_array.markers.append(start_marker)
            
            if goal_pose is not None:
                goal_marker = self._create_pose_marker(
                    goal_pose, self.config.goal_color, 'goal'
                )
                marker_array.markers.append(goal_marker)
        
        # 採樣區域可視化
        if self.config.show_sampling_region and sampling_ellipse:
            ellipse_marker = self._create_sampling_ellipse_marker(sampling_ellipse)
            marker_array.markers.append(ellipse_marker)
        
        # 統計資訊可視化
        if self.config.show_stats and stats:
            stats_marker = self._create_stats_marker(stats)
            marker_array.markers.append(stats_marker)
        
        # 發布所有標記
        self.viz_markers_pub.publish(marker_array)
    
    # =========================================================================
    # 標記建立函式
    # =========================================================================
    
    def _get_next_marker_id(self) -> int:
        """取得下一個標記 ID"""
        marker_id = self.marker_id_counter
        self.marker_id_counter += 1
        return marker_id
    
    def _create_header(self) -> Header:
        """建立標準 Header"""
        header = Header()
        header.frame_id = self.global_frame
        header.stamp = self.get_clock().now().to_msg()
        return header
    
    def _create_delete_all_marker(self) -> Marker:
        """建立刪除所有標記的 Marker"""
        marker = Marker()
        marker.header = self._create_header()
        marker.ns = 'aitstar_viz'
        marker.id = self._get_next_marker_id()
        marker.action = Marker.DELETEALL
        return marker
    
    def _publish_delete_all(self):
        """發布刪除所有標記"""
        marker_array = MarkerArray()
        marker_array.markers.append(self._create_delete_all_marker())
        self.viz_markers_pub.publish(marker_array)
    
    def _create_tree_markers(self, edges: List[TreeEdge]) -> List[Marker]:
        """建立樹可視化標記"""
        markers = []
        
        if not edges:
            return markers
        
        # 計算代價範圍（用於顏色漸變）
        costs = [e.cost for e in edges if e.cost > 0]
        min_cost = min(costs) if costs else 0.0
        max_cost = max(costs) if costs else 1.0
        
        # 使用 LINE_LIST 繪製所有邊
        line_marker = Marker()
        line_marker.header = self._create_header()
        line_marker.ns = 'aitstar_tree'
        line_marker.id = self._get_next_marker_id()
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = self.config.tree_line_width
        line_marker.pose.orientation.w = 1.0
        
        # 根據顏色方案設定
        color_scheme = self.config.tree_color_scheme
        
        for edge in edges:
            # 添加點
            p1 = Point(x=float(edge.start[0]), 
                      y=float(edge.start[1]), 
                      z=float(edge.start[2]) if len(edge.start) > 2 else 0.05)
            p2 = Point(x=float(edge.end[0]), 
                      y=float(edge.end[1]), 
                      z=float(edge.end[2]) if len(edge.end) > 2 else 0.05)
            line_marker.points.append(p1)
            line_marker.points.append(p2)
            
            # 添加顏色
            if color_scheme == ColorScheme.COST_GRADIENT:
                color = cost_to_color(edge.cost, min_cost, max_cost, 
                                     self.config.tree_alpha)
            elif color_scheme == ColorScheme.DEPTH_GRADIENT:
                color = depth_to_color(edge.depth, 20, self.config.tree_alpha)
            else:
                color = self.config.tree_default_color
            
            line_marker.colors.append(color)
            line_marker.colors.append(color)
        
        markers.append(line_marker)
        
        return markers
    
    def _create_path_markers(self, points: List[np.ndarray]) -> List[Marker]:
        """建立路徑可視化標記"""
        markers = []
        
        if len(points) < 2:
            return markers
        
        # 路徑線條
        line_marker = Marker()
        line_marker.header = self._create_header()
        line_marker.ns = 'aitstar_path'
        line_marker.id = self._get_next_marker_id()
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = self.config.path_line_width
        line_marker.color = self.config.path_color
        line_marker.pose.orientation.w = 1.0
        
        for point in points:
            p = Point(x=float(point[0]), 
                     y=float(point[1]), 
                     z=float(point[2]) if len(point) > 2 else 0.1)
            line_marker.points.append(p)
        
        markers.append(line_marker)
        
        # 路徑點
        if self.config.show_path_points:
            points_marker = Marker()
            points_marker.header = self._create_header()
            points_marker.ns = 'aitstar_path_points'
            points_marker.id = self._get_next_marker_id()
            points_marker.type = Marker.SPHERE_LIST
            points_marker.action = Marker.ADD
            points_marker.scale = Vector3(
                x=self.config.path_point_size,
                y=self.config.path_point_size,
                z=self.config.path_point_size
            )
            points_marker.pose.orientation.w = 1.0
            
            # 漸變顏色（從起點綠色到終點藍色）
            for i, point in enumerate(points):
                p = Point(x=float(point[0]), 
                         y=float(point[1]), 
                         z=float(point[2]) if len(point) > 2 else 0.15)
                points_marker.points.append(p)
                
                t = i / max(1, len(points) - 1)
                color = create_color(0.0, 1.0 - t * 0.5, t, 0.9)
                points_marker.colors.append(color)
            
            markers.append(points_marker)
        
        # 路徑箭頭（顯示方向）
        if len(points) >= 2:
            arrow_marker = self._create_path_arrows(points)
            if arrow_marker:
                markers.append(arrow_marker)
        
        return markers
    
    def _create_path_arrows(self, points: List[np.ndarray]) -> Optional[Marker]:
        """建立路徑方向箭頭"""
        if len(points) < 2:
            return None
        
        marker = Marker()
        marker.header = self._create_header()
        marker.ns = 'aitstar_path_arrows'
        marker.id = self._get_next_marker_id()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color = create_color(0.0, 0.8, 1.0, 0.7)
        marker.pose.orientation.w = 1.0
        
        # 每隔幾個點添加一個箭頭
        step = max(1, len(points) // 10)
        arrow_size = 0.15
        
        for i in range(0, len(points) - 1, step):
            start = points[i]
            end = points[min(i + 1, len(points) - 1)]
            
            direction = end[:2] - start[:2]
            length = np.linalg.norm(direction)
            
            if length < 1e-6:
                continue
            
            direction = direction / length
            normal = np.array([-direction[1], direction[0]])
            
            # 箭頭中點
            mid = (start[:2] + end[:2]) / 2
            z = 0.12
            
            # 箭頭頂點
            tip = mid + direction * arrow_size * 0.5
            left = mid - direction * arrow_size * 0.3 + normal * arrow_size * 0.2
            right = mid - direction * arrow_size * 0.3 - normal * arrow_size * 0.2
            
            # 添加箭頭線段
            marker.points.append(Point(x=float(tip[0]), y=float(tip[1]), z=z))
            marker.points.append(Point(x=float(left[0]), y=float(left[1]), z=z))
            marker.points.append(Point(x=float(tip[0]), y=float(tip[1]), z=z))
            marker.points.append(Point(x=float(right[0]), y=float(right[1]), z=z))
        
        return marker if marker.points else None
    
    def _create_pose_marker(self, position: np.ndarray, 
                            color: ColorRGBA, name: str) -> Marker:
        """建立位姿標記（起點/終點）"""
        marker = Marker()
        marker.header = self._create_header()
        marker.ns = f'aitstar_{name}'
        marker.id = self._get_next_marker_id()
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2]) if len(position) > 2 else 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale = Vector3(
            x=self.config.marker_size,
            y=self.config.marker_size,
            z=0.02
        )
        
        marker.color = color
        
        return marker
    
    def _create_sampling_ellipse_marker(self, ellipse_data: Dict) -> Marker:
        """建立採樣橢球標記"""
        marker = Marker()
        marker.header = self._create_header()
        marker.ns = 'aitstar_sampling_region'
        marker.id = self._get_next_marker_id()
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(ellipse_data['center'][0])
        marker.pose.position.y = float(ellipse_data['center'][1])
        marker.pose.position.z = 0.01
        marker.pose.orientation = ellipse_data['orientation']
        
        marker.scale = Vector3(
            x=float(ellipse_data['scale'][0]) * 2,
            y=float(ellipse_data['scale'][1]) * 2,
            z=0.01
        )
        
        marker.color = create_color(0.5, 0.5, 1.0, self.config.sampling_region_alpha)
        
        return marker
    
    def _create_stats_marker(self, stats: Dict) -> Marker:
        """建立統計資訊文字標記"""
        marker = Marker()
        marker.header = self._create_header()
        marker.ns = 'aitstar_stats'
        marker.id = self._get_next_marker_id()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.config.stats_position[0]
        marker.pose.position.y = self.config.stats_position[1]
        marker.pose.position.z = self.config.stats_position[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = 0.3  # 文字大小
        marker.color = create_color(1.0, 1.0, 1.0, 0.9)
        
        # 格式化統計資訊
        text_lines = ['AIT* 規劃統計']
        text_lines.append('-' * 20)
        
        if 'iterations' in stats:
            text_lines.append(f"迭代次數: {stats['iterations']}")
        if 'total_vertices' in stats:
            text_lines.append(f"頂點數量: {stats['total_vertices']}")
        if 'solution_cost' in stats:
            cost = stats['solution_cost']
            if cost < float('inf'):
                text_lines.append(f"路徑代價: {cost:.3f}")
            else:
                text_lines.append("路徑代價: 尚未找到")
        if 'planning_time' in stats:
            text_lines.append(f"規劃時間: {stats['planning_time']:.3f}s")
        if 'collision_checks' in stats:
            text_lines.append(f"碰撞檢查: {stats['collision_checks']}")
        
        marker.text = '\n'.join(text_lines)
        
        return marker


# =============================================================================
# 主程式
# =============================================================================

def main(args=None):
    """主程式入口"""
    rclpy.init(args=args)
    
    try:
        node = AITStarVisualizerNode()
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info('收到中斷訊號，正在關閉...')
        finally:
            node.destroy_node()
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()