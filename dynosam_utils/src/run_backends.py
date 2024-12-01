from dynosam_utils.evaluation.runner import run

def run_backend_sequnce(path, name, data_loader_num):
    parsed_args = {
        "dataset_path": path,
        "output_path": "/root/results/Dynosam_tro2024/",
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
        "launch_file": "dyno_sam_experiments_launch.py"
    }

    additional_args = [
        f"--data_provider_type={data_loader_num}",
    ]

    run(parsed_args, additional_args)

def run_kitti_backend_sequence(path, name):
    run_backend_sequnce(path, name, 0)

def run_cluster_backend_sequnce(path, name):
    run_backend_sequnce(path, name, 2)

if __name__ == '__main__':
    # run_kitti_backend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004",
    #     "kitti_0004"
    # )

    # run_kitti_backend_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003",
    #     "kitti_0003"
    # )

    run_cluster_backend_sequnce(
        "/root/data/cluster_slam/CARLA-L1",
        "carla_l1"
    )
