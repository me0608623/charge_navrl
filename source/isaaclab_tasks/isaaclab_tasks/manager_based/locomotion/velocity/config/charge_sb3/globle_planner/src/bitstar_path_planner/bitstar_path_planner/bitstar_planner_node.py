#!/usr/bin/env python3
"""
BIT* Path Planner ROS2 Node

This node provides path planning service using the BIT* algorithm.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from typing import Optional, Tuple
from bitstar_path_planner.bitstar import BITStar


class BITStarPlannerNode(Node):
    """ROS2 node for BIT* path planning"""
    
    def __init__(self):
        super().__init__('bitstar_planner_node')
        
        # Declare parameters
        self.declare_parameter('max_iterations', 5000)
        self.declare_parameter('max_batch_size', 100)
        self.declare_parameter('goal_radius', 0.5)
        self.declare_parameter('rewire_radius', 2.0)
        self.declare_parameter('goal_bias', 0.05)
        self.declare_parameter('map_frame', 'map')
        
        # Get parameters
        self.max_iterations = self.get_parameter('max_iterations').value
        self.max_batch_size = self.get_parameter('max_batch_size').value
        self.goal_radius = self.get_parameter('goal_radius').value
        self.rewire_radius = self.get_parameter('rewire_radius').value
        self.goal_bias = self.get_parameter('goal_bias').value
        self.map_frame = self.get_parameter('map_frame').value
        
        # Map data
        self.map_data: Optional[OccupancyGrid] = None
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # Subscribers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile
        )
        
        # Publishers
        self.path_publisher = self.create_publisher(
            Path,
            '/bitstar_path',
            10
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/bitstar_markers',
            10
        )
        
        # Service (will be implemented as action in future)
        self.get_logger().info('BIT* Planner Node initialized')
        self.get_logger().info(f'Max iterations: {self.max_iterations}')
        self.get_logger().info(f'Max batch size: {self.max_batch_size}')
    
    def map_callback(self, msg: OccupancyGrid):
        """Callback for map updates"""
        self.map_data = msg
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}, '
                              f'resolution: {self.map_resolution}')
    
    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to map indices"""
        if self.map_data is None:
            return None, None
        
        mx = int((x - self.map_origin_x) / self.map_resolution)
        my = int((y - self.map_origin_y) / self.map_resolution)
        
        return mx, my
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """Convert map indices to world coordinates"""
        if self.map_data is None:
            return None, None
        
        x = mx * self.map_resolution + self.map_origin_x
        y = my * self.map_resolution + self.map_origin_y
        
        return x, y
    
    def is_collision_free(self, position: np.ndarray) -> bool:
        """Check if position is collision-free"""
        if self.map_data is None:
            return True  # No map, assume free space
        
        mx, my = self.world_to_map(position[0], position[1])
        
        if mx is None or my is None:
            return False
        
        # Check bounds
        if mx < 0 or mx >= self.map_data.info.width or \
           my < 0 or my >= self.map_data.info.height:
            return False
        
        # Check occupancy
        index = my * self.map_data.info.width + mx
        if index >= len(self.map_data.data):
            return False
        
        occupancy = self.map_data.data[index]
        
        # 0 = free, 100 = occupied, -1 = unknown
        return occupancy == 0
    
    def plan_path(self, start: Pose, goal: Pose) -> Tuple[bool, Path, float]:
        """
        Plan path from start to goal
        
        Args:
            start: Start pose
            goal: Goal pose
            
        Returns:
            Tuple of (success, path, cost)
        """
        if self.map_data is None:
            self.get_logger().warn('No map available, cannot plan')
            return False, Path(), float('inf')
        
        # Extract 2D positions
        start_pos = np.array([start.position.x, start.position.y])
        goal_pos = np.array([goal.position.x, goal.position.y])
        
        # Get map bounds
        map_width = self.map_data.info.width * self.map_resolution
        map_height = self.map_data.info.height * self.map_resolution
        
        bounds_min = np.array([
            self.map_origin_x,
            self.map_origin_y
        ])
        bounds_max = np.array([
            self.map_origin_x + map_width,
            self.map_origin_y + map_height
        ])
        
        # Create planner
        planner = BITStar(
            start=start_pos,
            goal=goal_pos,
            bounds=(bounds_min, bounds_max),
            collision_checker=self.is_collision_free,
            max_iterations=self.max_iterations,
            max_batch_size=self.max_batch_size,
            goal_radius=self.goal_radius,
            rewire_radius=self.rewire_radius,
            goal_bias=self.goal_bias,
        )
        
        # Plan
        self.get_logger().info('Starting BIT* planning...')
        success, path_points, cost = planner.plan()
        
        if success:
            self.get_logger().info(f'Path found! Cost: {cost:.2f}, '
                                  f'Nodes: {len(planner.forward_tree) + len(planner.reverse_tree)}, '
                                  f'Iterations: {planner.iterations}')
        else:
            self.get_logger().warn('Path planning failed')
        
        # Convert to ROS Path message
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in path_points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(point[0])
            pose_stamped.pose.position.y = float(point[1])
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        return success, path_msg, cost
    
    def publish_path(self, path: Path):
        """Publish path"""
        self.path_publisher.publish(path)
    
    def publish_markers(self, start: Pose, goal: Pose, path: Path):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        
        # Start marker
        start_marker = Marker()
        start_marker.header.frame_id = self.map_frame
        start_marker.header.stamp = self.get_clock().now().to_msg()
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose = start
        start_marker.scale.x = 0.3
        start_marker.scale.y = 0.3
        start_marker.scale.z = 0.3
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        marker_array.markers.append(start_marker)
        
        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = self.map_frame
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose = goal
        goal_marker.scale.x = 0.3
        goal_marker.scale.y = 0.3
        goal_marker.scale.z = 0.3
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0
        marker_array.markers.append(goal_marker)
        
        # Path line marker
        if len(path.poses) > 0:
            path_marker = Marker()
            path_marker.header.frame_id = self.map_frame
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.id = 2
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.1
            path_marker.color.r = 0.0
            path_marker.color.g = 0.0
            path_marker.color.b = 1.0
            path_marker.color.a = 1.0
            
            for pose_stamped in path.poses:
                point = Point()
                point.x = pose_stamped.pose.position.x
                point.y = pose_stamped.pose.position.y
                point.z = pose_stamped.pose.position.z
                path_marker.points.append(point)
            
            marker_array.markers.append(path_marker)
        
        self.marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    
    node = BITStarPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
