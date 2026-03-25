#!/usr/bin/env python3
"""
AIT* 路徑規劃器 Launch 檔案
===========================

啟動 AIT* 規劃節點和可視化節點

使用方式：
    ros2 launch aitstar_path_planner aitstar_planner.launch.py
    
    # 帶參數啟動
    ros2 launch aitstar_path_planner aitstar_planner.launch.py \
        use_sim_time:=true \
        enable_visualization:=true \
        params_file:=/path/to/custom_params.yaml

作者: Claude
版本: 1.0.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """生成 Launch 描述"""
    
    # =========================================================================
    # 取得套件路徑
    # =========================================================================
    
    pkg_share = FindPackageShare('aitstar_path_planner')
    
    # 預設參數檔案路徑
    default_params_file = PathJoinSubstitution([
        pkg_share, 'config', 'aitstar_params.yaml'
    ])
    
    # =========================================================================
    # 宣告 Launch 參數
    # =========================================================================
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='使用模擬時間 (Gazebo/Isaac Sim)'
    )
    
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='AIT* 參數檔案的完整路徑'
    )
    
    declare_enable_visualization = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='啟用可視化節點'
    )
    
    declare_namespace = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='節點命名空間'
    )
    
    declare_global_frame = DeclareLaunchArgument(
        'global_frame',
        default_value='map',
        description='全域座標框架'
    )
    
    declare_robot_frame = DeclareLaunchArgument(
        'robot_frame',
        default_value='base_link',
        description='機器人座標框架'
    )
    
    declare_map_topic = DeclareLaunchArgument(
        'map_topic',
        default_value='/map',
        description='地圖 Topic'
    )
    
    declare_odom_topic = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='里程計 Topic'
    )
    
    declare_goal_topic = DeclareLaunchArgument(
        'goal_topic',
        default_value='/goal_pose',
        description='目標位姿 Topic'
    )
    
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        choices=['debug', 'info', 'warn', 'error', 'fatal'],
        description='日誌等級'
    )
    
    declare_respawn = DeclareLaunchArgument(
        'respawn',
        default_value='false',
        description='節點崩潰時自動重啟'
    )
    
    # =========================================================================
    # Launch 配置變數
    # =========================================================================
    
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    enable_visualization = LaunchConfiguration('enable_visualization')
    namespace = LaunchConfiguration('namespace')
    global_frame = LaunchConfiguration('global_frame')
    robot_frame = LaunchConfiguration('robot_frame')
    map_topic = LaunchConfiguration('map_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    goal_topic = LaunchConfiguration('goal_topic')
    log_level = LaunchConfiguration('log_level')
    respawn = LaunchConfiguration('respawn')
    
    # =========================================================================
    # 節點定義
    # =========================================================================
    
    # AIT* 規劃節點
    aitstar_planner_node = Node(
        package='aitstar_path_planner',
        executable='aitstar_planner_node',
        name='aitstar_planner_node',
        namespace=namespace,
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'global_frame': global_frame,
                'robot_frame': robot_frame,
            }
        ],
        remappings=[
            ('/map', map_topic),
            ('/odom', odom_topic),
            ('/goal_pose', goal_topic),
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )
    
    # AIT* 可視化節點
    aitstar_visualizer_node = Node(
        package='aitstar_path_planner',
        executable='aitstar_visualizer_node',
        name='aitstar_visualizer_node',
        namespace=namespace,
        output='screen',
        condition=IfCondition(enable_visualization),
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'global_frame': global_frame,
            }
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )
    
    # =========================================================================
    # 啟動日誌
    # =========================================================================
    
    log_info_start = LogInfo(msg=['========================================'])
    log_info_title = LogInfo(msg=['  AIT* 路徑規劃器啟動中...'])
    log_info_end = LogInfo(msg=['========================================'])
    
    log_info_viz = LogInfo(
        condition=IfCondition(enable_visualization),
        msg=['  可視化節點: 已啟用']
    )
    
    log_info_no_viz = LogInfo(
        condition=UnlessCondition(enable_visualization),
        msg=['  可視化節點: 已停用']
    )
    
    # =========================================================================
    # 組合 Launch Description
    # =========================================================================
    
    return LaunchDescription([
        # 設定環境變數
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        
        # 宣告參數
        declare_use_sim_time,
        declare_params_file,
        declare_enable_visualization,
        declare_namespace,
        declare_global_frame,
        declare_robot_frame,
        declare_map_topic,
        declare_odom_topic,
        declare_goal_topic,
        declare_log_level,
        declare_respawn,
        
        # 日誌輸出
        log_info_start,
        log_info_title,
        log_info_viz,
        log_info_no_viz,
        log_info_end,
        
        # 啟動節點
        aitstar_planner_node,
        aitstar_visualizer_node,
    ])