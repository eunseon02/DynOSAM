from dynosam_utils.evaluation.runner import run

def run_frontend_sequnce(path, name, data_loader_num, backend_type):
    parsed_args = {
        "dataset_path": path,
        "output_path": "/root/results/Dynosam_tro2024/",
        "name": name,
        "run_pipeline": True,
        "run_analysis": False,
        "launch_file": "dyno_sam_launch.py"
    }

    additional_args = [
        "f--backend_updater_enum={backend_type}",
        f"--data_provider_type={data_loader_num}",
        "--use_backend=0",
        "--save_frontend_json=true"
    ]

    run(parsed_args, additional_args)

def run_kitti_frontend_sequence(path, name):
    run_frontend_sequnce(path, name, 0)

def run_cluster_vo_frontend_sequence(path, name):
    run_frontend_sequnce(path, name, 2)


if __name__ == '__main__':
    # make input dictionary

    # run_kitti_frontend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004",
    #     "kitti_0004"
    # )

    # run_kitti_frontend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005",
    #     "kitti_0005"
    # )

    # run_kitti_frontend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000",
    #     "kitti_0000"
    # )

    # run_kitti_frontend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003",
    #     "kitti_0003"
    # )

    run_cluster_vo_frontend_sequence(
        "/root/data/cluster_slam/CARLA-L1",
        "carla_l1"
    )
