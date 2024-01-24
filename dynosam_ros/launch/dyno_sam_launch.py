from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import OpaqueFunction
import launch.logging

import os
from ament_index_python.packages import get_package_share_directory

logger = launch.logging.get_logger('dynosam_launch.user')

#NOTE: very helpful answers on ROS2 launch files: https://answers.ros.org/question/382000/ros2-makes-launch-files-crazy-too-soon-to-be-migrating/#:~:text=#!/usr/bin/env,condition=UnlessCondition(use_gui)%0A%20%20%20%20)%0A%0A%20%20])

def get_dynosam_params_path():
    #shoudl really be dynosam/<dataset>/example.flags like Kimera but we leave verbose for now
    dynosam_share_dir = get_package_share_directory('dynosam')

    logger.info("Loading default dynosam params folder path which is expected to be in the share directory of the dynosam package")

    if not os.path.exists(dynosam_share_dir):
        raise FileNotFoundError(
            "dynosam package share directory does not exist at path {}".format(dynosam_share_dir)
        )

    #check params folder has been correctly exported
    share_folders = os.listdir(dynosam_share_dir)
    if "params" not in share_folders:
        raise FileNotFoundError(
            "dynosam package share directory exists but \'params'\ folder does not exist at {}. "
            "Has the params folder been exported in the dynosam CMakeLists.txt like:\n"
            "\'install(DIRECTORY\n"
            "\tparams\n"
            "\tDESTINATION share/${PROJECT_NAME}\n"
            ")\'?".format(dynosam_share_dir)
        )

    return os.path.join(
        dynosam_share_dir,
        "params"
        ) + "/"

def validate_dynosam_params_path(params_path: str):
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            "dynosam params path does not exist at {}".format(params_path)
        )
    return params_path



def load_dynosam_node(context, *args, **kwargs):
    dynosam_dataset_path_config = LaunchConfiguration("dataset_path")
    dynosam_params_folder_config = LaunchConfiguration("params_path")
    glog_verbose_flag_config = LaunchConfiguration("v")

    # Construct the flagfile arguments given the params folder
    # where we expect to find some files with the suffix .flags (for gflags)
    # these will be added to the arguments list as --flagfile="/path/to/flags/file.flags"
    def construct_flagfile_arguments(params_folder):
        arguments = []
        from pathlib import Path
        for file in os.listdir(params_folder):
            if Path(file).suffix == ".flags":
                arg = "--flagfile={}".format(os.path.join(params_folder, file))
                arguments.append(arg)
        return arguments

    def contruct_glags_verbose_argument(loaded_glog_verbose_flag):
        return "--v={}".format(loaded_glog_verbose_flag)

    # argv is parsed in by the LaunchContext and will be the sys.argv[1:] (so not including the system name.)
    # we can use this to append to the actaul arguments we want to pass to the node. This is useful for
    # all the non-ros (ie GFLAGS/GLOG libs) we use  and allow us to modify this from otuside this launch file
    argv = context.argv
    def construct_additional_arguments(argv):
        argv = list(argv)
        # if empty (the system name is already removed) then no args to add
        if len(argv) == 0:
            return None

        logger.info("Adding additional sys.argv arguements: {}".format(argv))
        return argv

    loaded_params_folder = validate_dynosam_params_path(dynosam_params_folder_config.perform(context))
    loaded_glog_verbose_flag = glog_verbose_flag_config.perform(context)


    arguments = construct_flagfile_arguments(loaded_params_folder)
    arguments.append(contruct_glags_verbose_argument(loaded_glog_verbose_flag))

    additional_arguments = construct_additional_arguments(argv)
    if additional_arguments:
        arguments.extend(additional_arguments)

    logger.info("Adding arguments {}".format(arguments))

    node = Node(
        package='dynosam_ros',
        executable='dynosam_node',
        output='screen',
        parameters=[
            {"params_folder_path": dynosam_params_folder_config},
            {"dataset_path": dynosam_dataset_path_config}
        ],
        arguments=arguments
    )
    return [node]


def generate_launch_description():
    default_data_set_path = "/root/data/vdo_slam/kitti/kitti/0020"
    default_glog_v = 20
    default_params_path = get_dynosam_params_path()

    # declare params path argument
    dynosam_params_folder_arg = DeclareLaunchArgument(
        'params_path',
        default_value=default_params_path)



    # declare dataset path argument
    dynosam_dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value=default_data_set_path)

    # # devlare verbosity (glog) argument
    glog_verbose_flag_arg = DeclareLaunchArgument(
        'v',
        default_value=str(default_glog_v))

    return LaunchDescription([
        # Must be inside launch description to be registered
        dynosam_params_folder_arg,
        dynosam_dataset_path_arg,
        glog_verbose_flag_arg,
        OpaqueFunction(function=load_dynosam_node)
    ])
