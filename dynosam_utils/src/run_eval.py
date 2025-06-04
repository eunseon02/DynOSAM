from dynosam_utils.evaluation.runner import run


def run_analysis(name):
    parsed_args = {
        "dataset_path": "",
        "output_path": "/root/results/DynoSAM/",
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

# def run_sequnce(path, name, data_loader_num):
#     parsed_args = {
#         "dataset_path": path,
#         "output_path": "/root/results/DynoSAM/",
#         "name": name,
#         "run_pipeline": True,
#         "run_analysis": True,
#         "launch_file": "dyno_sam_launch.py"
#     }

#     additional_args = [
#         "--backend_updater_enum=0",
#         f"--data_provider_type={data_loader_num}",
#         "--use_backend=1",
#         "--save_frontend_json=false",
#         f"--constant_object_motion_translation_sigma=0.3",
#         f"--constant_object_motion_rotation_sigma=0.1",
#         f"--ending_frame=500"
#     ]

#     run(parsed_args, additional_args)

if __name__ == '__main__':
    run_analysis("test")
#     run_sequnce(
#         "/root/data/omm/swinging_4_unconstrained",
#         "swinging_4_unconstrained",
#         3
#     )
