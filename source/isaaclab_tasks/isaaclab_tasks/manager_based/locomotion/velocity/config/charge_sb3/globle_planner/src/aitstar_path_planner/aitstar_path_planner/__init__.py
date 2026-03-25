"""
AIT* Path Planner Package
=========================

AIT* (Adaptive Informed Trees) 路徑規劃套件

模組：
- aitstar: 核心演算法實作
- aitstar_planner_node: ROS2 規劃節點
- aitstar_visualizer: ROS2 可視化節點

使用方式：
    from aitstar_path_planner.aitstar import AITStar, SimpleCollisionChecker
    from aitstar_path_planner.aitstar import PathSmoother, State
"""

from .aitstar import (
    AITStar,
    State,
    Vertex,
    Edge,
    CollisionChecker,
    SimpleCollisionChecker,
    Sampler,
    PathSmoother,
)

__version__ = '1.0.0'
__author__ = 'Claude'

__all__ = [
    'AITStar',
    'State',
    'Vertex', 
    'Edge',
    'CollisionChecker',
    'SimpleCollisionChecker',
    'Sampler',
    'PathSmoother',
]