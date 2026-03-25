#!/usr/bin/env python3
"""
BIT* Visualizer Node

Simple visualization node for testing BIT* planner
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
import sys


class BITStarVisualizer(Node):
    """Simple visualizer for BIT* planning"""
    
    def __init__(self):
        super().__init__('bitstar_visualizer')
        
        self.path_subscriber = self.create_subscription(
            Path,
            '/bitstar_path',
            self.path_callback,
            10
        )
        
        self.get_logger().info('BIT* Visualizer initialized')
        self.get_logger().info('Waiting for path messages on /bitstar_path')
    
    def path_callback(self, msg: Path):
        """Callback for path messages"""
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
        
        if len(msg.poses) > 0:
            self.get_logger().info('Path waypoints:')
            for i, pose_stamped in enumerate(msg.poses):
                pos = pose_stamped.pose.position
                self.get_logger().info(f'  [{i}] x: {pos.x:.2f}, y: {pos.y:.2f}, z: {pos.z:.2f}')


def main(args=None):
    rclpy.init(args=args)
    
    node = BITStarVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
