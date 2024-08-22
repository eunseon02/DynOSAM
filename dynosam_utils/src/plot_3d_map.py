import evaluation.evaluation_lib as eval

from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np

import matplotlib.pyplot as plt


def make_plot(results_folder_path, prefix):
    dataset_eval = eval.DatasetEvaluator(results_folder_path)
    data_files = dataset_eval.make_data_files(prefix)

    map_points_log_path = dataset_eval.create_existing_file_path(data_files.map_point_log)
    if map_points_log_path is None:
        print("Cannot find map points file")
        return

    camera_pose_eval = dataset_eval.create_camera_pose_evaluator(data_files)
    motion_eval = dataset_eval.create_motion_error_evaluator(data_files)

    object_pose_traj = motion_eval.object_poses_traj
    object_motion_traj = motion_eval.object_motion_traj

    object_key = list(object_pose_traj.keys())[0]

    print(object_motion_traj)

    object_trajectory = motion_eval.make_object_trajectory(object_key)
    # print(object_trajectory)

    map_fig = plt.figure(figsize=(8,8))
    # ax = evo_plot.prepare_axis(map_fig, evo_plot.PlotMode.xyz)
    # ax = map_fig.add_subplot(111, projection="3d", proj_type = 'ortho')
    ax = map_fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    eval.tools.plot_velocities(ax, object_trajectory)
    # map_fig.tight_layout()

    eval.tools.plot_object_trajectories(map_fig, object_pose_traj,
                                    plot_mode=evo_plot.PlotMode.xyz,
                                    # colours=colour_list,
                                #    plot_axis_est=True,
                                    plot_start_end_markers=True,
                                    axis_marker_scale=1.5,
                                    traj_zorder=30,
                                    traj_linewidth=1.0)

    plt.show()

    # iter = object_trajectory.get_motion_with_pose_previous_iterator()

    # for a, b in object_trajectory.get_motion_with_pose_previous_iterator():
    #     print(a, b)
    # object_pose_traj = object_pose_traj[object_key]
    # object_motion_traj = object_motion_traj[object_key]

    # print(object_motion_traj)
    # print(object_pose_traj)

    # object

    # plotter = eval.MapPlotter3D(map_points_log_path, camera_pose_eval, motion_eval)

    # plot_collection = evo_plot.PlotCollection("Map")
    # results = {}

    # plotter.process(plot_collection, results)
    # plot_collection.show()

    # MapPlotter3D,
    #             map_points_log_path,
    #             camera_pose_eval,
    #             motion_eval


# make_plot("/root/results/DynoSAM/acfr_1_moving_small", "rgbd_motion_world_backend")
make_plot("/root/results/Dynosam_tro2024/kitti_0000", "rgbd_motion_world_backend")
