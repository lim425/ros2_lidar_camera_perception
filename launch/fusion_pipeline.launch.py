from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_python = LaunchConfiguration('use_python')

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_python',
            default_value='false',
        ),

        # Python
        Node(
            package='lidar_camera_perception',
            executable='lidar_camera_fusion_node.py',
            name='lidar_camera_fusion_node',
            output='screen',
            parameters=[
                {'overlap_threshold': 0.15}                 
            ],
            condition=IfCondition(use_python)
        ),

        # C++
        Node(
            package='lidar_camera_perception',
            executable='lidar_camera_fusion_node',
            name='lidar_camera_fusion_node',
            output='screen',
            parameters=[
                {'overlap_threshold': 0.15}    
            ],
            condition=UnlessCondition(use_python)
        ),

    ])