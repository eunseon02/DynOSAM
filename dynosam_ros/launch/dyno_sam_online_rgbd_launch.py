from dynosam_ros.dynosam_node import DynosamNode
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch_ros.actions import Node
from dynosam_ros.launch_utils import get_default_dynosam_params_path
from ament_index_python.packages import get_package_share_directory
import os


def get_default_rviz_config_path():
    """Get the default rviz config file path from source directory."""
    # Try to find rviz config in source directory first
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    source_rviz_path = os.path.join(launch_file_dir, '..', 'rviz', 'rviz_dsd.rviz')
    source_rviz_path = os.path.abspath(source_rviz_path)
    
    if os.path.exists(source_rviz_path):
        return source_rviz_path
    
    # Fallback to empty string if not found
    return ""


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_path", default_value=get_default_dynosam_params_path()),
        DeclareLaunchArgument("v", default_value="30"),
        DeclareLaunchArgument("output_path", default_value="/root/results/DynoSAM/"),
        # DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("rgb_cam_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("depth_cam_topic", default_value="/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("rescale_width", default_value="640", description="Image width to rescale to"),
        DeclareLaunchArgument("rescale_height", default_value="480", description="Image height to rescale to"),
        DeclareLaunchArgument("use_rviz", default_value="true", description="Launch RViz"),
        DeclareLaunchArgument("rviz_config", default_value=get_default_rviz_config_path(), description="Path to RViz config file"),

        DynosamNode(
                package="dynosam_ros",
                executable="dynosam_node",
                output="screen",
                parameters=[
                    {"params_folder_path": LaunchConfiguration("params_path")},
                    {"rescale_width": LaunchConfiguration("rescale_width")},
                    {"rescale_height": LaunchConfiguration("rescale_height")},
                    {"online": True},
                    {"input_image_mode": 1}, # Corresponds with InputImageMode::RGBD
                    {"wait_for_camera_params": False} # Use CameraParams.yaml instead of camera_info topic
                ],
                remappings=[
                    # ("dataprovider/camera/camera_info",  LaunchConfiguration("camera_info_topic")),
                    ("image/rgb",  LaunchConfiguration("rgb_cam_topic")),
                    ("image/depth",  LaunchConfiguration("depth_cam_topic")),
                ]
            ),
            # Convert compressed images from rosbag to raw format
            Node(
                package='image_transport',
                executable='republish',
                name='republish_rgb',
                output='screen',
                parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
                remappings=[
                    ('in', '/camera/color/image_raw'),
                    ('out', '/camera/color/image_raw'),
                ]
            ),
            Node(
                package='image_transport',
                executable='republish',
                name='republish_depth',
                output='screen',
                parameters=[{'in_transport': 'compressedDepth', 'out_transport': 'raw'}],
                remappings=[
                    ('in', '/camera/aligned_depth_to_color/image_raw'),
                    ('out', '/camera/aligned_depth_to_color/image_raw'),
                ]
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=["0.0", "0.0", "0.0", "1.57", "-1.57", "0.0", "world", "robot"]
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', LaunchConfiguration('rviz_config')],
                condition=IfCondition(LaunchConfiguration('use_rviz'))
            )
        ])
