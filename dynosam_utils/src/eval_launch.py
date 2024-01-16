from launch import LaunchDescription  # noqa: E402
from launch import LaunchIntrospector  # noqa: E402
from launch import LaunchService  # noqa: E402
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

import os
import sys



def main(argv=sys.argv[1:]):
    """Run demo nodes via launch."""
    ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('dynosam_ros'), 'launch'),
            '/dyno_sam_launch.py']),
        launch_arguments={
            'dataset_path': '/root/data/vdo_slam/kitti/kitti/0000',

        }.items(),
    )

    print('')
    print('Starting launch of launch description...')
    print('')

    # parse arguments down to the actual launch file
    # these will be appended to the Node(arguments=...) parameter and can be used
    # to directly change things like FLAG_ options etc...
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
