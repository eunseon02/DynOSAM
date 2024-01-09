from launch import LaunchDescription
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    #shoudl really be dynosam/<dataset>/example.flags like Kimera but we leave verbose for now
    flags_file = os.path.join(
        get_package_share_directory('dynosam'),
        'example.flags'
        )


    return LaunchDescription([
        Node(
            package='dynosam_ros',
            executable='dynosam_node',
            parameters=[
                {"params_folder_path": "/home/user/dev_ws/src/DynOSAM/dynosam/params/"},
                {"dataset_path": "/root/data/vdo_slam/kitti/kitti/0000"}
                # {"dataset_path": "/root/data/virtual_kitti"}
            ],
            arguments=['--flagfile=/home/user/dev_ws/src/DynOSAM/dynosam/params/example.flags', '-v=20']
            )
    ])
