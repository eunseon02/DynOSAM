from dynosam_utils.evaluation.runner import run
import os
import sys

# runs new incremental backend
increment_backend_type = 6
# runs object centric backend as batch
object_centric_batch=2
# runs world centric backend as batch
motion_world_backend_type = 0

def run_sequnce(path, name, data_loader_num, backend_type, run_as_frontend=True, *args):
    parsed_args = {
        "dataset_path": path,
        "output_path": "/root/results/Dynosam_ecmr2024/",
        "name": name,
        "run_pipeline": True,
        "run_analysis": False,
    }

    additional_args = [
        "--data_provider_type={}".format(data_loader_num),
        "--v=20"
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

def run_analysis(name):
    parsed_args = {
        "dataset_path": "",
        "output_path": "/root/results/Dynosam_ecmr2024/",
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

kitti_dataset = 0
virtual_kitti_dataset = 1
cluster_dataset = 2
omd_dataset = 3

def prep_dataset(path, name, data_loader_num, *args):
    backend_type = increment_backend_type
    run_as_frontend=True
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        run_as_frontend,
        *args)

# from saved data
def run_saved_sequence(path, name, data_loader_num, *args, **kwargs):
    backend_type = kwargs.get("backend_type", increment_backend_type)
    run_as_frontend=False
    args_list = list(args)
    # args_list.append("--init_object_pose_from_gt=true")
    run_sequnce(
        path,
        name,
        data_loader_num,
        backend_type,
        run_as_frontend,
        *args_list)


# kitti stuff
def prep_kitti_sequence(path, name, *args):
    args_list = list(args)
    args_list.append("--shrink_row=25")
    args_list.append("--shrink_col=50")
    # args_list.append("--use_propogate_mask=true")
    prep_dataset(path, name, kitti_dataset, *args_list)

def run_kitti_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, kitti_dataset, *args, **kwargs)
    # run_analysis(name)

# cluster
def prep_cluster_sequence(path, name, *args, **kwargs):
    prep_dataset(path, name, cluster_dataset, *args, **kwargs)

def run_cluster_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, cluster_dataset, *args, **kwargs)

# omd
def prep_omd_sequence(path, name, *args, **kwargs):
    args_list = list(args)
    args_list.append("--shrink_row=0")
    args_list.append("--shrink_col=0")
    prep_dataset(path, name, omd_dataset, *args_list, **kwargs)

def run_omd_sequence(path, name, *args, **kwargs):
    run_saved_sequence(path, name, omd_dataset, *args, **kwargs)

if __name__ == '__main__':
    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     "--ending_frame=150"
    # )


    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000"
    # )
    # pass

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--updater_suffix=inc",
    #     backend_type = motion_world_backend_type
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--updater_suffix=inc",
    #     backend_type = object_centric_batch
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--updater_suffix=inc",
    #     backend_type = motion_world_backend_type
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=beta_01",
    #     "--relinearize_threshold=0.1"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--updater_suffix=beta_1",
    #     "--relinearize_threshold=1.0"
    # )
    # run_analysis("kitti_0000")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )
    # run_analysis("kitti_0003")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     backend_type = motion_world_backend_type
    # )
    # run_analysis("kitti_0004")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006",
    #     backend_type = increment_backend_type
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005",
    #     backend_type = motion_world_backend_type
    # )
    # run_analysis("kitti_0006")
    # run_analysis("kitti_0005")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )
    # run_analysis("kitti_0006")


    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018"
    # )

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     backend_type=object_centric_batch
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     backend_type=motion_world_backend_type
    # )
    # run_analysis("kitti_0018")

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=550"
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500",
    #     backend_type=motion_world_backend_type
    # )
    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0020")

    # prep_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--ending_frame=318")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     backend_type=increment_backend_type,
    #     # backend_type=motion_world_backend_type
    # )
    # run_analysis("kitti_0000")

    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=beta_01",
    #     "--relinearize_threshold=0.1"
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=beta_1",
    #     "--relinearize_threshold=1.0"
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=inc",
    #     backend_type=motion_world_backend_type
    # )
    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_long",
    #     "--updater_suffix=inc",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("omd_swinging_4_unconstrained_long")

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "cluster_l2",
    #     "--starting_frame=33",
    #     "--ending_frame=385",
    #     "--init_object_pose_from_gt=true"
    # )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "cluster_l2",
    #     "--init_object_pose_from_gt=true",
    #     "--num_dynamic_optimize=4",
    #     backend_type=increment_backend_type

    # )
    # run_analysis("cluster_l2")

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--starting_frame=33",
    #     "--ending_frame=385",
    #     "--init_object_pose_from_gt=true"
    # )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--init_object_pose_from_gt=false",
    #     "--num_dynamic_optimize=4"
    # )

    # run_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1",
    #     "--init_object_pose_from_gt=false",
    #     "--num_dynamic_optimize=4",
    #     # backend_type=object_centric_batch

    # )
    # run_analysis("cluster_l1")

    # run_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained",
    #     "--relinearize_thereshold=0.1"
    #     # backend_type=motion_world_backend_type
    # )

    # run_analysis("omd_swinging_4_unconstrained")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     # backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0004")

    # run_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003",
    #     backend_type=object_centric_batch
    # )
    # run_analysis("kitti_0003")
