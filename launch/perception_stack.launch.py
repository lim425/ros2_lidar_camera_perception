from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_dir = get_package_share_directory('lidar_camera_perception')

    use_python = LaunchConfiguration('use_python')

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_python',
            default_value='false'
        ),

        # lidar pipeline
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'lidar_pipeline.launch.py')
            ),
            launch_arguments={
                'use_python': use_python
            }.items()
        ),

        # camera pipeline
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'camera_pipeline.launch.py')
            ),
            launch_arguments={
                'use_python': use_python
            }.items()
        ),

        # sensor fusion pipeline
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'fusion_pipeline.launch.py')
            ),
            launch_arguments={
                'use_python': use_python
            }.items()
        ),

    ])