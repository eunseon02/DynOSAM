from eval_launch import run

def run_sequnce(path, name, data_loader_num, backend_type, run_as_frontend=True):
    parsed_args = {
        "dataset_path": path,
        "output_path": "/root/results/Dynosam_tro2024/",
        "name": name,
        "run_pipeline": True,
        "run_analysis": False,
    }


    if run_as_frontend:
        additional_args = [
            "--use_backend=0",
            "--save_frontend_json=true",
            "--v=0"
        ]
        parsed_args["launch_file"] = "dyno_sam_launch.py"
        run(parsed_args, additional_args)
    else:
        additional_args = [
            f"--backend_updater_enum={backend_type}",
            f"--data_provider_type={data_loader_num}",
            "--use_backend=1",
            "--v=0"
        ]
        parsed_args["launch_file"] = "dyno_sam_experiments_launch.py"
        run(parsed_args, additional_args)


def prep_kitti_sequence(path, name):
    run_sequnce(path, name, 0, 0, True)

def run_kitti_sequence(path, name, backend_type):
    run_sequnce(path, name, 0, backend_type, False)



def run_analysis(name):
    parsed_args = {
        "dataset_path": "",
        "output_path": "/root/results/Dynosam_tro2024/",
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

if __name__ == '__main__':
    # make input dictionary
    world_motion_backend = 0
    ll_backend = 1

    def run_kitti_both_backend(path, name):
        # run_sequnce(path, name, 0, world_motion_backend, False)
        # run_sequnce(path, name, 0, ll_backend, False)
        run_analysis(name)



    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004",
    #     "kitti_0004"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005",
    #     "kitti_0005"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000",
    #     "kitti_0000"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003",
    #     "kitti_0003"
    # )

    # run with world centric
    run_kitti_both_backend(
        "/root/data/vdo_slam/kitti/kitti/0004",
        "kitti_0004"
    )

    # run_kitti_both_backend(
    #     "/root/data/vdo_slam/kitti/kitti/0005",
    #     "kitti_0005"
    # )

    # run_kitti_both_backend(
    #     "/root/data/vdo_slam/kitti/kitti/0000",
    #     "kitti_0000"
    # )

    # run_kitti_both_backend(
    #     "/root/data/vdo_slam/kitti/kitti/0003",
    #     "kitti_0003"
    # )
