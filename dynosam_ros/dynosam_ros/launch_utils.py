import launch.logging

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import OpaqueFunction, RegisterEventHandler, Shutdown, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit

def _get_utils_logger():
    return launch.logging.get_logger('dynosam_launch_utils.user')

def validate_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "path does not exist at {}".format(path)
        )
    return path

#NOTE: very helpful answers on ROS2 launch files: https://answers.ros.org/question/382000/ros2-makes-launch-files-crazy-too-soon-to-be-migrating/#:~:text=#!/usr/bin/env,condition=UnlessCondition(use_gui)%0A%20%20%20%20)%0A%0A%20%20])
def get_default_dynosam_params_path():
    #shoudl really be dynosam/<dataset>/example.flags like Kimera but we leave verbose for now
    dynosam_share_dir = get_package_share_directory('dynosam')

    logger = _get_utils_logger()
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


def load_dynosam_node(context, *args, **kwargs):
    """
    Constucts the set of launch actions for running a dynosam node - this includes
    a dynosam-style node and all the parameters that would be required to run the system, including ROS
    and GLOG params.

    This function is intended to be used within an OpaqueFunction in the form
    '
        LaunchDescription([OpaqueFunction(function=load_dynosam_node)])
    '
    where by default args=[] and kwargs={} of the OpaqueFunction, which are then parsed as the *args **kwargs
    of this fucntion. Args is unused. Kwargs are used to specify
        - executable: Executable file name (as in the node executable program to run). Default is dynosam_node.
        - should_output: if the program node (running the executable) should output to screen. Default is True.
        - world_to_robot_tf: If a static transform should be setup between the world and robot. Default is True.

    Context is parsed down from the LaunchService and is used to access the cmd line arguments from the user. These arguments
    should be the additional (non-ros) arguments that will be appended to the program Node arguments - this allows GLOG arguments
    to be parsed directly from the cmdline and into google::ParseCommandLineFlags(&non_ros_argc, &non_ros_argv_c, true)

    The function operates with the following logic
        - Load dynosam parameter path from the launch argument 'params_path'
        - Append flagfiles (files that end with .flags) found on the params_path to the arguments list.
            This allows any flag file stored on the params path to be loaded via GLOG
        - Append addional cmd line arguments from the context to the node arguments.
            This may include overwritten glog arguments which are manually specified
        - Get the executable name from kwargs and construct a launch_ros.actions.Node
            with package 'dynosam_ros', the parameters from dataset_path and params_path,
            as well as the additional arguments as above
        - Setup a tf transform between the world and robot (world to camera) if requested in the kwargs


    Args:
        context (_type_): _description_

    Returns:
        _type_: _description_
    """
    logger = _get_utils_logger()

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

    # context.argv is parsed in by the LaunchContext and will be the sys.argv[1:] (so not including the system name.)
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

    loaded_params_folder = validate_path(dynosam_params_folder_config.perform(context))
    loaded_glog_verbose_flag = glog_verbose_flag_config.perform(context)

    # arguments to the dynosam Node, these are passed as non-ros args
    arguments = construct_flagfile_arguments(loaded_params_folder)
    arguments.append(contruct_glags_verbose_argument(loaded_glog_verbose_flag))

    # add additional arguments - this will most often be any gflags passed along the command line
    additional_arguments = construct_additional_arguments(argv)
    if additional_arguments:
        arguments.extend(additional_arguments)

    logger.info("Adding arguments {}".format(arguments))

     # get details for the node to launch from the kwargs
    executable = kwargs.get("executable", "dynosam_node")
    logger.info("Setting executable to: {}".format(executable))

    node_kwargs = {}
    #specifies if the node should set output='screen' as part of the node params
    if kwargs.get("should_output", True):
        node_kwargs['output'] = 'screen'
        logger.info("Outputting to screen")

    nodes = []

    program_node = Node(
        package='dynosam_ros',
        executable=executable,
        parameters=[
            {"params_folder_path": dynosam_params_folder_config},
            {"dataset_path": dynosam_dataset_path_config}
        ],
        arguments=arguments,
        **node_kwargs
    )
    nodes.append(program_node)

    def on_node_completion_lambda(*args):
        reason_string = "Program node {} has exited".format(program_node.name)
        logger.info(reason_string + " - shutting down!")
        #NOTE: the reason text does not seem to propogate to the logger. I guess this is a bug with ROS2?
        action = Shutdown()
        action.visit(context)

    # set up handler to shutdown this context when the program node exists
    # this prevents any additional nodes hanging after the program finishes running
    nodes.append(RegisterEventHandler(OnProcessExit(
            target_action = program_node,
            on_exit = on_node_completion_lambda)))

    # if a static transform should be setup between the world and robot
    if kwargs.get("world_to_robot_tf", True):
        logger.info("Setting up world to robot transform")

        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=["0.0", "0.0", "0.0", "1.57", "-1.57", "0.0", "world", "robot"]
        ))

    return nodes


def generate_dynosam_launch_description(**kwargs):
    # these values can be overwritten by providing launch arguments of the same name, these
    # just provide the default if no launch arguments are given
    default_data_set_path = kwargs.get('dataset_path', "/root/data/VDO/kitti/kitti/0004")
    default_glog_v = kwargs.get('v', 20)
    default_params_path = kwargs.get('params_path', get_default_dynosam_params_path())

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

    # remove the above values from the kwargs so that they are not parsed to the load_dynosam_node function
    clean_kwargs = kwargs

    if "dataset_path" in clean_kwargs:
        del clean_kwargs["dataset_path"]

    if "v" in clean_kwargs:
        del clean_kwargs["v"]

    if "params_path" in clean_kwargs:
        del clean_kwargs["params_path"]


    return LaunchDescription([
        # Must be inside launch description to be registered
        dynosam_params_folder_arg,
        dynosam_dataset_path_arg,
        glog_verbose_flag_arg,
        OpaqueFunction(function=load_dynosam_node, kwargs=clean_kwargs)
    ])
