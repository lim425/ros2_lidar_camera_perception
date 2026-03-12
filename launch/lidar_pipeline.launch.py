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
            executable='lidar_preprocessing_node.py',
            name='lidar_preprocessing_node',
            output='screen',
            condition=IfCondition(use_python)
        ),

        Node(
            package='lidar_camera_perception',
            executable='lidar_cluster_detector_node.py',
            name='lidar_cluster_detector_node',
            output='screen',
            condition=IfCondition(use_python)
        ),

        Node(
            package='lidar_camera_perception',
            executable='lidar_tracker_node.py',
            name='lidar_tracker_node',
            output='screen',
            parameters=[
                {'max_age': 3},                  
                {'distance_threshold': 4.0} 
            ],
            condition=IfCondition(use_python)
        ),

        # C++
        Node(
            package='lidar_camera_perception',
            executable='lidar_preprocessing_node',
            name='lidar_preprocessing_node',
            output='screen',
            condition=UnlessCondition(use_python)
        ),

        Node(
            package='lidar_camera_perception',
            executable='lidar_cluster_detector_node',
            name='lidar_cluster_detector_node',
            output='screen',
            condition=UnlessCondition(use_python)
        ),

        Node(
            package='lidar_camera_perception',
            executable='lidar_tracker_node',
            name='lidar_tracker_node',
            output='screen',
            parameters=[
                {'max_age': 3},
                {'distance_threshold': 4.0}
            ],
            condition=UnlessCondition(use_python)
        ),

    ])