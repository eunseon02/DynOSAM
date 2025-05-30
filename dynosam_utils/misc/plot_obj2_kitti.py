import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.formatting_utils as dyno_formatting

import dynosam_utils.evaluation.core as core
import evo.tools.plot as evo_plot
import numpy as np

import matplotlib.pyplot as plt

import evo.core.sync as evo_sync


dataset_eval = eval.DatasetEvaluator("/root/results/Dynosam_ecmr2024/kitti_0000")
data_files_opt = dataset_eval.make_data_files("object_centric_backend")
data_files_fontend = dataset_eval.make_data_files("frontend")

opt_motion = dataset_eval.create_motion_error_evaluator(data_files_opt)
frontend_motion = dataset_eval.create_motion_error_evaluator(data_files_fontend)

map_fig = plt.figure(figsize=(14,9))
        # ax = evo_plot.prepare_axis(map_fig, evo_plot.PlotMode.xyz)
# ax = map_fig.add_subplot(111, projection="3d")
ax = map_fig.add_subplot(111)


ax.set_ylabel(r"Y(m)")
ax.set_xlabel(r"X(m)")
# ax.set_zlabel(r"Z(m)")
# ax.patch.set_facecolor('white')
eval.tools.set_clean_background(ax)


frontend_obj2 = frontend_motion.object_poses_traj[2]
opt_obj2 = opt_motion.object_poses_traj[2]
gt_obj2 = opt_motion.object_poses_traj_ref[2]

frontend_obj2, opt_obj2 = evo_sync.associate_trajectories(
    frontend_obj2,
    opt_obj2,
    max_diff=0.05)

opt_obj2, gt_obj2 = evo_sync.associate_trajectories(
    opt_obj2,
    gt_obj2,
    max_diff=0.05)


core.plotting.plot_object_trajectories(map_fig,
                                       { "Before Opt": frontend_obj2, "After Opt": opt_obj2},
                                    #    { "Ground Truth": gt_obj2},
                                       plot_mode=evo_plot.PlotMode.xy,
                                       colours=[dyno_formatting.get_nice_blue(), dyno_formatting.get_nice_green()],
                                       plot_axis_est=True,
                                       plot_start_end_markers=False,
                                       axis_marker_scale=3.0,
                                       traj_zorder=30,
                                       downscale=0.05,
                                    #    est_name_prefix="Object",
                                       traj_linewidth=3.0)

core.plotting.plot_object_trajectories(map_fig,
                                       { "Ground Truth": gt_obj2},
                                       plot_mode=evo_plot.PlotMode.xy,
                                    #    colours=colour_list,
                                        colours=[dyno_formatting.get_nice_red()],
                                       plot_axis_est=True,
                                       plot_start_end_markers=False,
                                       axis_marker_scale=3.0,
                                       traj_zorder=30,
                                       est_style="--",
                                       downscale=0.05,
                                    #    est_name_prefix="Object",
                                       traj_linewidth=3.0)

ax.set_box_aspect(1)
ax.set_axisbelow(True)
map_fig.tight_layout(pad=0.05)
ax.grid(which='major', color='#DDDDDD', linewidth=1.0)

map_fig.savefig("/root/results/TRO2025/kitti_0000_trajectory.pdf")
