from dynosam_ros.launch_utils import generate_dynosam_launch_description

def generate_launch_description():
    return generate_dynosam_launch_description(
        executable = "dynosam_node",
        should_output = True,
        world_to_robot_tf = True,
        v=30
    )
