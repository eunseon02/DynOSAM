import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.metrics as eval_metrics
from dynosam_utils.evaluation.formatting_utils import * #for nice colours


from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np
import sys
import gtsam

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Reset all rcParams to their default values
plt.rcdefaults()
startup_plotting(50)


plt.rcParams["lines.linewidth"] = 4.0
# plt.rcParams.update({
#                     "text.usetex": True,
#                     "font.family": "serif",
#                     "font.serif": ["Computer Modern Roman"],
#                     })


# font_size=40

# # # Change default font sizes.
# plt.rc('font', size=font_size)
# plt.rc('axes', titlesize=font_size)
# plt.rc('axes', labelsize=font_size)
# plt.rc('xtick', labelsize=0.6*font_size)
# plt.rc('ytick', labelsize=0.6*font_size)
# plt.rc('legend', fontsize=0.7*font_size)

# plt.rcParams['axes.titlesize'] = 29    # Title font size
# plt.rcParams['axes.labelsize'] = 37    # X/Y label font size
# plt.rcParams['figure.titlesize'] = 25    # Title font size
# plt.rcParams['xtick.labelsize'] = 33   # X tick label font size
# plt.rcParams['ytick.labelsize'] = 33   # Y tick label font size
# plt.rcParams['legend.fontsize']= 29


# plt.rcParams['axes.titlesize'] = 25    # Title font size
# plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
# plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
# plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size
# plt.rcParams['legend.fontsize']=14


def make_plot_all_objects(
        prefix,
        dataset_evaluator:eval.DatasetEvaluator,
        objects:List[int],
        suptitle:bool = True,
        **kwargs):
    data_files = dataset_evaluator.make_data_files(prefix)


    if not data_files.check_is_dynosam_results():
        print(f"Invalid data file {data_files}")
        sys.exit(0)

    motion_eval = dataset_evaluator.create_motion_error_evaluator(data_files)

    # for object_id, object_motion_traj_est, object_motion_traj_ref in eval.common_entries(motion_eval.object_motion_traj, motion_eval.object_motion_traj_ref):
    for object_id, object_motion_traj_est, object_pose_traj_ref in eval.common_entries(motion_eval.object_motion_traj, motion_eval.object_poses_traj_ref):
        if object_motion_traj_est.num_poses < 5:
            continue

        #if objects list provided, check if the object id is in the list, otherwise skip
        if objects is not None and object_id not in objects:
            continue


        fig, (rot_error_axes, trans_error_axes) = plt.subplots(nrows=2, sharex=True, layout="constrained")
        # fig.set_size_inches(15, 15) # for KITTI
        # fig.set_size_inches(15, 11)  # for OMD

        if suptitle:
            fig.suptitle(f"Object {object_id}", fontweight="bold")
        rot_error_axes.margins(0.001)
        set_clean_background(rot_error_axes)
        rot_error_axes.set_ylabel("$E_r$(\N{degree sign})")

        set_clean_background(trans_error_axes)
        trans_error_axes.margins(0.001)
        trans_error_axes.set_ylabel("$E_t$(m)")
        trans_error_axes.set_xlabel("Frame Index [-]")


        rme_E = eval_metrics.RME(eval_metrics.PoseRelation.full_transformation)

        # print(object_motion_traj_est.timestamps)
        # print(object_pose_traj_ref.timestamps)
        # # copied from tools.plot_trajectory_error
        # import evo.core.metrics as metrics
        # ape_E = metrics.APE(metrics.PoseRelation.full_transformation)
        data = (object_pose_traj_ref,object_motion_traj_est)

        # data = (object_motion_traj_ref,object_motion_traj_est)
        # eval_metrics.RME.sync_object_motion_and_pose(data)
        rme_E.process_data(data)

        rot_error = []
        trans_error = []
        for E in rme_E.E[:-10]:
            E_se3 = gtsam.Pose3(E)
            E_trans = E_se3.translation()
            E_rot = E_se3.rotation()
            rx, ry, rz =  np.degrees(E_rot.rpy())

            trans_error.append(E_trans)
            rot_error.append(np.array([rx, ry, rz]))

        rot_error = np.array(rot_error)
        trans_error = np.array(trans_error)


        rot_error_axes.plot(rot_error[:,0], label="roll", color=get_nice_red(), **kwargs)
        rot_error_axes.plot(rot_error[:,1], label="pitch", color=get_nice_green(), **kwargs)
        rot_error_axes.plot(rot_error[:,2], label="yaw", color=get_nice_blue(), **kwargs)

        trans_error_axes.plot(trans_error[:,0], label="x", color=get_nice_red(), **kwargs)
        trans_error_axes.plot(trans_error[:,1], label="y", color=get_nice_green(), **kwargs)
        trans_error_axes.plot(trans_error[:,2], label="z", color=get_nice_blue(), **kwargs)

        # smart_legend(rot_error_axes)
        # smart_legend(trans_error_axes)
        rot_error_axes.legend(loc='upper left',ncol=3)
        # rot_error_axes.legend(loc='upper right')
        trans_error_axes.legend(loc='upper left',ncol=3)
        # rot_error_axes.legend()
        # trans_error_axes.legend()
        fig.tight_layout()

        return fig, rot_error_axes, trans_error_axes



def make_plot(results_folder_path, plot_frontend = True, plot_backend = True, objects=None,suptitle:bool = True):
    dataset_eval = eval.DatasetEvaluator(results_folder_path)

    frontend_fig_axes = None
    backend_fig_axes = None
    if plot_frontend:
       frontend_fig_axes = make_plot_all_objects("frontend", dataset_eval, objects, suptitle,linestyle="-")

    if plot_backend:
       backend_fig_axes = make_plot_all_objects("rgbd_motion_world_backend", dataset_eval, objects, suptitle,linestyle="-")

    if frontend_fig_axes and backend_fig_axes:
        fig_frontend, frontend_rot_axes, frontend_trans_axes = frontend_fig_axes
        fig_backend, backend_rot_axes, backend_trans_axes = backend_fig_axes

        def set_axes_equal(ax1: Axes, ax2: Axes):
            y1_min = min(ax1.get_ylim())
            y1_max = max(ax1.get_ylim())

            y2_min = min(ax2.get_ylim())
            y2_max = max(ax2.get_ylim())

            # Get the min and max values for both plots
            y_min = min(y1_min, y2_min)
            y_max = max(y1_max, y2_max)

            ax2.set_ylim(y_min, y_max)
            ax1.set_ylim(y_min, y_max)

        set_axes_equal(frontend_rot_axes, backend_rot_axes)
        set_axes_equal(frontend_trans_axes, backend_trans_axes)

        # fig_frontend.tight_layout(pad=0.9)
        # fig_backend.tight_layout(pad=0.9)
        fig_frontend.tight_layout()
        fig_backend.tight_layout()

        return (fig_frontend, frontend_rot_axes, frontend_trans_axes), (fig_backend, backend_rot_axes, backend_trans_axes)




# make_plot("/root/results/DynoSAM/omd_vo_test", plot_frontend=False, plot_backend=True, objects=[4], suptitle=False)
# make_plot("/root/results/DynoSAM/test_kitti_main", plot_frontend=True, plot_backend=False, objects=[2], suptitle=True)
# make_plot("/root/results/DynoSAM/test_kitti_vo_0003", plot_frontend=True, plot_backend=True, objects=[1], suptitle=True)
# make_plot("/root/results/DynoSAM/test_kitti_vo_0004", plot_frontend=True, plot_backend=False)


# these are the ones we actually used
# omd_frontend, omd_backend = make_plot("/root/results/Dynosam_tro2024/omd_vo_test", plot_frontend=True, plot_backend=True, objects=[4], suptitle=False)
kitti_frontend, kitti_backend = make_plot("/root/results/DynoSAM/test_kitti_main", plot_frontend=True, plot_backend=True, objects=[2], suptitle=False)




plt.show()
