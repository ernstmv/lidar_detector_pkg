#!/home/user/.anaconda3/envs/openmmlab/bin/python3.8
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lidar_detector_pkg'),
        'config',
        'lidar_detector_pkg.config.yaml'
    )

    return LaunchDescription([
        Node(
            package='lidar_detector_pkg',
            executable='lidar_detector_node',
            name='lidar_detector_node',
            parameters=[config],
        )
    ])
