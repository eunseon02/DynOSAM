import evaluation.evaluation_lib as eval
from evaluation.formatting_utils import * #for nice colours


from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np
import sys
import gtsam

import matplotlib.pyplot as plt

# Reset all rcParams to their default values
plt.rcdefaults()

plt.rcParams['axes.titlesize'] = 25    # Title font size
plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size
plt.rcParams['legend.fontsize']=18


def make_plot_all_objects(prefix, dataset_evaluator:eval.DatasetEvaluator, objects:List[int], suptitle:bool = True, **kwargs):
    data_files = dataset_evaluator.make_data_files(prefix)


    if not data_files.check_is_dynosam_results():
        print(f"Invalid data file {data_files}")
        sys.exit(0)

    motion_eval = dataset_evaluator.create_motion_error_evaluator(data_files)

    for object_id, object_traj, object_traj_ref in eval.common_entries(motion_eval.object_motion_traj, motion_eval.object_motion_traj_ref):
        if object_traj.num_poses < 5:
            continue

        #if objects list provided, check if the object id is in the list, otherwise skip
        if objects is not None and object_id not in objects:
            continue


        fig, (rot_error_axes, trans_error_axes) = plt.subplots(nrows=2, sharex=True)

        fig.set_size_inches(13, 8)

        if suptitle:
            fig.suptitle(f"Object {object_id}", fontweight="bold", fontsize=25)
        rot_error_axes.margins(0.001)
        set_clean_background(rot_error_axes)
        rot_error_axes.set_ylabel("$E_r$(\N{degree sign})", fontsize=19)

        set_clean_background(trans_error_axes)
        trans_error_axes.margins(0.001)
        trans_error_axes.set_ylabel("$E_t$(m)", fontsize=19)
        trans_error_axes.set_xlabel("Frame Index [-]")

        fig.tight_layout(pad=0.5)


        # copied from tools.plot_trajectory_error
        import evo.core.metrics as metrics
        ape_E = metrics.APE(metrics.PoseRelation.full_transformation)
        data = (object_traj, object_traj_ref)
        ape_E.process_data(data)

        rot_error = []
        trans_error = []
        for E in ape_E.E:
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
        rot_error_axes.legend(loc='best')
        # rot_error_axes.legend(loc='upper right')
        trans_error_axes.legend(loc='best')
        # rot_error_axes.legend()
        # trans_error_axes.legend()



def make_plot(results_folder_path, plot_frontend = True, plot_backend = True, objects=None,suptitle:bool = True):
    dataset_eval = eval.DatasetEvaluator(results_folder_path)

    if plot_frontend:
       make_plot_all_objects("frontend", dataset_eval, objects, suptitle,linestyle="-")

    if plot_backend:
       make_plot_all_objects("rgbd_motion_world_backend", dataset_eval, objects, suptitle,linestyle="-")



# make_plot("/root/results/DynoSAM/omd_vo_test", plot_frontend=False, plot_backend=True, objects=[4], suptitle=False)
make_plot("/root/results/DynoSAM/test_kitti_main", plot_frontend=False, plot_backend=True, objects=[1], suptitle=True)
make_plot("/root/results/DynoSAM/test_kitti_main", plot_frontend=True, plot_backend=False, objects=[1], suptitle=True)

# make_plot("/root/results/DynoSAM/test_kitti_vo_0004", plot_frontend=True, plot_backend=False)



plt.show()
