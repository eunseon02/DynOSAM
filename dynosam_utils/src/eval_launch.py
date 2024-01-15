from launch import LaunchDescription  # noqa: E402
from launch import LaunchIntrospector  # noqa: E402
from launch import LaunchService  # noqa: E402
from launch.actions import IncludeLaunchDescription

import os
from ament_index_python.packages import get_package_share_directory

import launch_ros.actions  # noqa: E402
import sys




def main(argv=sys.argv[1:]):
    """Run demo nodes via launch."""
    # ld = LaunchDescription([
    #     launch_ros.actions.Node(
    #         package='dynosam', executable='talker', output='screen',
    #         remappings=[('chatter', 'my_chatter')]),
    #     launch_ros.actions.Node(
    #         package='demo_nodes_cpp', executable='listener', output='screen',
    #         remappings=[('chatter', 'my_chatter')]),
    # ])
    ld = IncludeLaunchDescription(
        package='dynosam', launch='dyno_sam_launch.py')

    print('Starting introspection of launch description...')
    print('')

    print(LaunchIntrospector().format_launch_description(ld))

    print('')
    print('Starting launch of launch description...')
    print('')

    # ls = LaunchService(debug=True)
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    main()
