from dynosam_utils.evaluation.runner import run
import os
import sys

output_path = "/root/results/TRO2025/"

def run_sequnce(path, name, data_loader_num, backend_type, run_as_frontend=True, run_as_experiment=True, run_analysis=False, *args,):
    parsed_args = {
        "dataset_path": path,
        "output_path": output_path,
        "name": name,
        "run_pipeline": True,
        "run_analysis": run_analysis,
    }

    additional_args = [
        "--data_provider_type={}".format(data_loader_num),
        "--v=20"
    ]

    parsed_args["launch_file"] = "dyno_sam_launch.py"

    if run_as_frontend:
        additional_args.extend([
            "--use_backend=0",
            "--save_frontend_json=true"
        ])
    else:
        additional_args.extend([
            "--backend_updater_enum={}".format(backend_type),
            "--use_backend=1"
        ])
        if run_as_experiment:
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
    args_list.append("--shrink_col=50")
    args_list.append("--use_propogate_mask=true")
    prep_dataset(path, name, kitti_dataset, *args_list)

def run_kitti_sequence(path, name, backend_type, *args):
    run_saved_sequence(path, name, kitti_dataset, backend_type, *args)

# cluster
def prep_cluster_sequence(path, name, *args):
    prep_dataset(path, name, cluster_dataset, *args)

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
    results_path = output_path
    sub_folders = [name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]
    for folder in sub_folders:
        run_analysis(folder)


def run_analysis(name):
    parsed_args = {
        "dataset_path": "",
        "output_path": output_path,
        "name": name,
        "run_pipeline": False,
        "run_analysis": True,
    }
    parsed_args["launch_file"] = "dyno_sam_launch.py"
    run(parsed_args, [])

def run_POM_tests(run_prep_sequence_func, path, base_name, *args):
    from copy import deepcopy
    name_P = base_name + "_P"
    name_PO = base_name + "_PO"
    name_POM = base_name + "_POM"

    args_list = list(args)
    args_list.append("--use_backend=0")

    args_list_P =  deepcopy(args_list)
    args_list_PO = deepcopy(args_list)
    args_list_POM = deepcopy(args_list)

    args_list_P.append("--refine_motion_estimate=false")
    args_list_P.append("--refine_with_optical_flow=false")

    args_list_PO.append("--refine_motion_estimate=false")
    args_list_PO.append("--refine_with_optical_flow=true")

    args_list_POM.append("--refine_motion_estimate=true")
    args_list_POM.append("--refine_with_optical_flow=true")

    # run_prep_sequence_func(path, name_P, *args_list_P)
    # run_prep_sequence_func(path, name_PO, *args_list_PO)
    run_prep_sequence_func(path, name_POM, *args_list_POM)

    # run_analysis(name_P)
    # run_analysis(name_PO)
    # run_analysis(name_POM)

def run_viodes():
    # run_sequnce("/root/data/VIODE/city_day/mid", "viode_city_day_mid", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/VIODE/city_day/high", "viode_city_day_high", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    run_sequnce("/root/data/VIODE/city_night/mid", "viode_city_night_mid", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    run_sequnce("/root/data/VIODE/city_night/high", "viode_city_night_high", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/VIODE/parking_lot/mid", "viode_parking_lot_mid", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/VIODE/parking_lot/high", "viode_parking_lot_high", 6, 3, run_as_frontend=False, run_as_experiment=False, run_analysis=True)

if __name__ == '__main__':
    # make input dictionary
    world_motion_backend = 0
    ll_backend = 1

    run_viodes()
    sys.exit(0)

    # def run_both_backend(run_sequence_func, path, name, *args):
    #     # run_sequence_func(path, name, world_motion_backend, *args)
    #     run_sequence_func(path, name, ll_backend, *args)
    #     run_analysis(name)

    # run_sequnce("/root/data/cluster_slam/CARLA-L1/", "carla_l1_new", cluster_dataset, world_motion_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/cluster_slam/CARLA-L1/", "carla_l1_new", cluster_dataset, ll_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    # run_sequnce("/root/data/cluster_slam/CARLA-L2/", "carla_l2_new", cluster_dataset, world_motion_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/cluster_slam/CARLA-L2/", "carla_l2_new", cluster_dataset, ll_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    # run_sequnce("/root/data/cluster_slam/CARLA-S1/", "carla_s1_new", cluster_dataset, world_motion_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/cluster_slam/CARLA-S1/", "carla_s1_new", cluster_dataset, ll_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    # run_sequnce("/root/data/cluster_slam/CARLA-S2/", "carla_s2_new", cluster_dataset, world_motion_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)
    # run_sequnce("/root/data/cluster_slam/CARLA-S2/", "carla_s2_new", cluster_dataset, ll_backend, run_as_frontend=False, run_as_experiment=False, run_analysis=True)

    run_analysis("carla_s1_new")
    run_analysis("carla_s2_new")

    sys.exit(0)
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0004/", "kitti_0004")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0001/", "kitti_0001")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0000/", "kitti_0000")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0002/", "kitti_0002")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0003/", "kitti_0003")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0005/", "kitti_0005")
    # run_POM_tests(prep_kitti_sequence, "/root/data/vdo_slam/kitti/kitti/0018/", "kitti_0018")

    # run_POM_tests(prep_cluster_sequence, "/root/data/cluster_slam/CARLA-L1/", "carla_l1", "--use_propogate_mask=false", "--use_dynamic_track=false", "--ending_frame=300")

    # run_POM_tests(prep_omd_sequence, "/root/data/omm/swinging_4_unconstrained/","omd_swinging_4_unconstrained", "--use_dynamic_track=false", "--ending_frame=300", "--semantic_mask_step_size=6", "--v=0")
    # run_analysis("carla_l1")
    # run_analysis("carla_l2")
    # run_analysis("carla_s1")
    # run_analysis("carla_s2")

    # run_analysis("kitti_0000")
    # run_analysis("kitti_0001")
    # run_analysis("kitti_0002")
    # run_analysis("kitti_0003")
    # run_analysis("kitti_0004")
    # run_analysis("kitti_0005")
    # run_analysis("kitti_0006")
    # run_analysis("kitti_0018")
    # run_analysis("kitti_0020")
    # run_analysis("omd_swinging_4_unconstrained_sliding")
    # run_all_eval()
    # sys.exit()

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004"
    # )
    # sys.exit()

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--v=20"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--use_propogate_mask=false"
    # )

    # prep_kitti_sequence(
    #     "/root/data/falsevdo_slam/kitti/kitti/0020/",
    #     "kitti_0020",
    #     "--ending_frame=500"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0001/",
    #     "kitti_0001"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0002/",
    #     "kitti_0002"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0001/",
    #     "kitti_0001"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0002/",
    #     "kitti_0002"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006"
    # )


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
    #     "kitti_0000_sliding",
    #     "--use_full_batch_opt=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004_sliding",
    #     "--use_full_batch_opt=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000",
    #     "--use_full_batch_opt=true"
    # )

    # prep_kitti_sequence(
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003_far_origin",
    #     "--v=20"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003_far_origin",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004_far_origin",
    #     "--use_full_batch_opt=true",
    # )



    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0000/",
    #     "kitti_0000_far_origin",
    #     "--use_full_batch_opt=true",
    # )

    # prep_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_far_origin",
    #     "--use_dynamic_track=false",
    #     "--ending_frame=150",
    #     "--semantic_mask_step_size=15")

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_far_origin",
    #     "--use_full_batch_opt=true")


    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0001/",
    #     "kitti_0001",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0002/",
    #     "kitti_0002",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true",
    #     "--use_vo_factor=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true",
    #     "--use_vo_factor=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0005/",
    #     "kitti_0005",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true",
    #     "--use_vo_factor=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0006/",
    #     "kitti_0006",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true",
    #     "--use_vo_factor=false"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true",
    #     "--use_vo_factor=false"
    # )

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_vo_test",
    #     "--use_full_batch_opt=false",
    #     "--ending_frame=500",
    #     "--semantic_mask_step_size=15",
    #     "--constant_object_motion_rotation_sigma=1.0",
    #     "--constant_object_motion_translation_sigma=0.2",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend")


    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "carla_l1_new",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "cluster_l2_new",
    #     "--starting_frame=33",
    #     "--ending_frame=385",
    #     "--init_object_pose_from_gt=true"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "cluster_l1_new",
    #     "--starting_frame=33",
    #     "--ending_frame=600",
    #     "--init_object_pose_from_gt=true"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-S1/",
    #     "cluster_s1_new",
    #     "--init_object_pose_from_gt=true"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-S2/",
    #     "cluster_s2_new",
    #     "--init_object_pose_from_gt=true"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "cluster_l2_new",
    #     "--use_full_batch_opt=true",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    # )


    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "carla_l2",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-S2/",
    #     "carla_s2",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-S2/",
    #     "carla_s1",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend"
    # )


    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0004/",
    #     "kitti_0004",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0018/",
    #     "kitti_0018",
    #     "--init_H_with_identity=false",
    #     "--updater_suffix=init_frontend",
    #     "--use_full_batch_opt=true"
    # )

    # run_both_backend(
    #     run_kitti_sequence,
    #     "/root/data/vdo_slam/kitti/kitti/0003/",
    #     "kitti_0003",
    #     "--use_full_batch_opt=true"
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
    # prep_omd_sequence(
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained",
    #     "--use_dynamic_track=false",
    #     "--ending_frame=150",
    #     "--semantic_mask_step_size=4")

    # prep_omd_sequence(
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained",
    #     "--use_dynamic_track=false",
    #     "--ending_frame=300",
    #     "--semantic_mask_step_size=4")


    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained",
    #     "--use_full_batch_opt=true",
    #     "--constant_object_motion_rotation_sigma=0.01",
    #     "--constant_object_motion_translation_sigma=0.03",
    #     )

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained_sliding_compare",
    #     "--use_full_batch_opt=false",
    #     "--constant_object_motion_rotation_sigma=0.01",
    #     "--constant_object_motion_translation_sigma=0.03",
    #     )

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/omm/swinging_4_unconstrained/",
    #     "omd_swinging_4_unconstrained",
    #     "--use_full_batch_opt=true",
    #     "--ending_frame=300",
    #     "--semantic_mask_step_size=15",
    #     "--constant_object_motion_rotation_sigma=0.001",
    #     "--constant_object_motion_translation_sigma=0.001",
    #     "--odometry_translation_sigma=0.001",
    #     "--odometry_rotation_sigma=0.001")

    # prep_omd_sequence(
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_vo_test",
    #     "--use_dynamic_track=false",
    #     "--semantic_mask_step_size=15",
    #     "--ending_frame=500")

    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_batch",
    #     "--use_full_batch_opt=true",
    #     "--ending_frame=300",
    #     "--semantic_mask_step_size=15",
    #     "--constant_object_motion_rotation_sigma=1.0",
    #     "--constant_object_motion_translation_sigma=0.2",
    #     "--init_H_with_identity=true")


    # run_both_backend(
    #     run_omd_sequence,
    #     "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/",
    #     "omd_swinging_4_unconstrained_sliding",
    #     "--use_full_batch_opt=false",
    #     "--ending_frame=300",
    #     "--semantic_mask_step_size=15",
    #     "--constant_object_motion_rotation_sigma=1.0",
    #     "--constant_object_motion_translation_sigma=0.2",
    #     "--init_H_with_identity=true")




    ## cluster
    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "carla_l1_new",
    #     "--use_propogate_mask=false",
    #     "--use_dynamic_track=false",
    #     "--ending_frame=250"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "carla_l2",
    #     "--use_propogate_mask=false",
    #     "--use_dynamic_track=false"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-S1/",
    #     "carla_s1",
    #     "--use_propogate_mask=false",
    #     "--use_dynamic_track=false"
    # )

    # prep_cluster_sequence(
    #     "/root/data/cluster_slam/CARLA-S2/",
    #     "carla_s2",
    #     "--use_propogate_mask=false",
    #     "--use_dynamic_track=false"
    # )


    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L1/",
    #     "carla_l1_dense",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    #     "--init_LL_with_identity=true"
    # )

    # run_both_backend(
    #     run_cluster_sequence,
    #     "/root/data/cluster_slam/CARLA-L2/",
    #     "carla_l2",
    #     "--use_full_batch_opt=false",
    #     "--opt_window_size=20",
    #     "--opt_window_overlap=5",
    #     "--init_LL_with_identity=true"
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
