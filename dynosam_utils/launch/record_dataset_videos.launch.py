from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    dataset_name = "omd_s4u"
    output_folder = "/root/results/misc/"

    tracking_image_record = Node(
        package="dynosam_utils",
        executable="record_image_topic.py",
        parameters=[
            {"image_topic": "/dynosam_dataset/tracking_image"},
            {"video_file_name": f"{dataset_name}_tracking_images.avi"},
            {"output_folder": output_folder}
        ]
    )

    input_image_record = Node(
        package="dynosam_utils",
        executable="record_image_topic.py",
        parameters=[
            {"image_topic": "/dynosam_dataset/input_images"},
            {"video_file_name": f"{dataset_name}_input_images.avi"},
            {"output_folder": output_folder}
        ]
    )

    ld.add_action(tracking_image_record)
    ld.add_action(input_image_record)

    return ld
