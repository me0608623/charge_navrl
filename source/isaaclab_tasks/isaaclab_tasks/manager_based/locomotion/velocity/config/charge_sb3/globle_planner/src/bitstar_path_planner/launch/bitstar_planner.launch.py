"""
Launch file for BIT* Path Planner
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description"""
    
    # Launch arguments
    max_iterations_arg = DeclareLaunchArgument(
        'max_iterations',
        default_value='5000',
        description='Maximum number of iterations for BIT* planning'
    )
    
    max_batch_size_arg = DeclareLaunchArgument(
        'max_batch_size',
        default_value='100',
        description='Maximum batch size for sampling'
    )
    
    goal_radius_arg = DeclareLaunchArgument(
        'goal_radius',
        default_value='0.5',
        description='Goal radius for planning'
    )
    
    rewire_radius_arg = DeclareLaunchArgument(
        'rewire_radius',
        default_value='2.0',
        description='Rewiring radius for tree optimization'
    )
    
    goal_bias_arg = DeclareLaunchArgument(
        'goal_bias',
        default_value='0.05',
        description='Probability of sampling goal directly'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Map frame ID'
    )
    
    # BIT* Planner Node
    bitstar_planner_node = Node(
        package='bitstar_path_planner',
        executable='bitstar_planner_node',
        name='bitstar_planner_node',
        parameters=[{
            'max_iterations': LaunchConfiguration('max_iterations'),
            'max_batch_size': LaunchConfiguration('max_batch_size'),
            'goal_radius': LaunchConfiguration('goal_radius'),
            'rewire_radius': LaunchConfiguration('rewire_radius'),
            'goal_bias': LaunchConfiguration('goal_bias'),
            'map_frame': LaunchConfiguration('map_frame'),
        }],
        output='screen'
    )
    
    # Visualizer Node (optional)
    visualizer_node = Node(
        package='bitstar_path_planner',
        executable='bitstar_visualizer',
        name='bitstar_visualizer',
        output='screen'
    )
    
    return LaunchDescription([
        max_iterations_arg,
        max_batch_size_arg,
        goal_radius_arg,
        rewire_radius_arg,
        goal_bias_arg,
        map_frame_arg,
        bitstar_planner_node,
        visualizer_node,
    ])
