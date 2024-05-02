from eval_launch import run

if __name__ == '__main__':
    # make input dictionary

    parsed_args = {
        "dataset_path": "/root/data/vdo_slam/kitti/kitti/0004",
        "output_path": "/root/results/DynoSAM/",
        "name": "kitti_0004",
        "run_pipeline": True,
        "run_analysis": False,
        "launch_file": "dyno_sam_launch.py"
    }

    additional_args = [
        "--backend_updater_enum=1",
        "--data_provider_type=0"
    ]

    run(parsed_args, additional_args)
