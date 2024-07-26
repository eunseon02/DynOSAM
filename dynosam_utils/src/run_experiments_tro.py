from eval_launch import run
import os

def run_sequnce(path, name, data_loader_num, backend_type, run_as_frontend=True, *args):
    parsed_args = {
        "dataset_path": path,
        "output_path": "/root/results/Dynosam_tro2024/",
        "name": name,
        "run_pipeline": True,
        "run_analysis": False,
    }

    additional_args = [
        "--data_provider_type={}".format(data_loader_num),
        "--v=0"
    ]
    if run_as_frontend:
        additional_args.extend([
            "--use_backend=0",
            "--save_frontend_json=true"
        ])
        parsed_args["launch_file"] = "dyno_sam_launch.py"
    else:
        additional_args.extend([
            "--backend_updater_enum={}".format(backend_type),
            "--use_backend=1"
        ])
        parsed_args["launch_file"] = "dyno_sam_experiments_launch.py"

    if len(args) > 0:
        additional_args.extend(list(args))

    # print(additional_args)
    run(parsed_args, additional_args)


kitti_dataset = 0
virtual_kitti_dataset = 1
cluster_dataset = 2
omd_dataset = 3

def prep_dataset(path, name, data_loader_num, *args):
    backend_type = 0
    run_as_frontend=True
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        run_as_frontend,
        *args)

# from saved data
def run_saved_sequence(path, name, data_loader_num, backend_type, *args):
    run_as_frontend=False
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        run_as_frontend,
        *args)


# kitti stuff
def prep_kitti_sequence(path, name, *args):
    args_list = list(args)
    args_list.append("--shrink_row=25")
    args_list.append("--shrink_col=25")
    prep_dataset(path, name, kitti_dataset, *args_list)

def run_kitti_sequence(path, name, backend_type, *args):
    run_saved_sequence(path, name, kitti_dataset, backend_type, *args)

# cluster
def prep_cluster_sequence(path, name, *args):
    prep_dataset(path, name, cluster_dataset, args)

def run_cluster_sequence(path, name, backend_type, *args):
    run_saved_sequence(path, name, cluster_dataset, backend_type, *args)

# omd
def prep_omd_sequence(path, name, *args):
    args_list = list(args)
    args_list.append("--shrink_row=0")
    args_list.append("--shrink_col=0")
    prep_dataset(path, name, omd_dataset, *args_list)

def run_omd_sequence(path, name, backend_type, *args):
    run_saved_sequence(path, name, omd_dataset, backend_type, *args)



def run_all_eval():
    results_path = "/root/results/Dynosam_tro2024/"
    sub_folders = [name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]
    for folder in sub_folders:
        run_analysis(folder)


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

    def run_both_backend(run_sequence_func, path, name, *args):
        run_sequence_func(path, name, world_motion_backend, *args)
        # run_sequence_func(path, name, ll_backend, *args)
        run_analysis(name)



    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0004/",
        "kitti_0004"
    )

    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0005/",
        "kitti_0005"
    )

    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0000/",
        "kitti_0000"
    )

    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0003/",
        "kitti_0003"
    )

    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0018/",
        "kitti_0018"
    )

    prep_kitti_sequence(
        "/root/data/vdo_slam/kitti/kitti/0020/",
        "kitti_0020",
        "--ending_frame=500"
    )


    # # run with world centric
    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500"
    # )


    # omd
    prep_omd_sequence(
        "/root/data/omm/swinging_4_unconstrained/",
        "omd_swinging_4_unconstrained_sliding",
        "--ending_frame=300",
        "--semantic_mask_step_size=4")

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained_sliding_500",
    #     "--use_full_batch_opt=true",
    #     "--ending_frame=200",
    #     "--semantic_mask_step_size=15",
    #     "--constant_object_motion_rotation_sigma=0.001",
    #     "--constant_object_motion_translation_sigma=0.001",
    #     "--odometry_translation_sigma=0.001",
    #     "--odometry_rotation_sigma=0.001")


    ## cluster
    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-L1/",
        "carla_l1",
        "--use_propogate_mask=false"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-L2/",
        "carla_l2",
        "--use_propogate_mask=false"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-S1/",
        "carla_s1",
        "--use_propogate_mask=false"
    )

    prep_cluster_sequence(
        "/root/data/cluster_slam/CARLA-S2/",
        "carla_s2",
        "--use_propogate_mask=false"
    )


    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "carla_l1",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "carla_l2",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-S1/",
    #     "carla_s1",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-S2/",
    #     "carla_s2",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_all_eval()

    # kitti sliding window
    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004_sliding_window"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005_sliding_window"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000_sliding_window"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003_sliding_window"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018_sliding_window"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020_sliding_window",
    #     "--ending_frame=500"
    # )

    # # run with world centric
    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004_sliding_window",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005_sliding_window",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000_sliding_window",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003_sliding_window",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018_sliding_window",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020_sliding_window",
    #     "--ending_frame=500",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )
