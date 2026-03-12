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

        # YOLO
        Node(
            package='lidar_camera_perception',
            executable='camera_object_detection_node.py',
            name='camera_object_detection_node',
            output='screen',
        ),

        # Python
        Node(
            package='lidar_camera_perception',
            executable='image_overlay_node.py',
            name='image_overlay_node',
            output='screen',
            parameters=[
                {'show_unmatched_tracks': False},
                {'overlap_threshold': 0.15}
            ],
            condition=IfCondition(use_python)
        ),

        # C++
        Node(
            package='lidar_camera_perception',
            executable='image_overlay_node',
            name='image_overlay_node',
            output='screen',
            parameters=[
                {'show_unmatched_tracks': False},
                {'overlap_threshold': 0.15}
            ],
            condition=UnlessCondition(use_python)
        ),

    ])