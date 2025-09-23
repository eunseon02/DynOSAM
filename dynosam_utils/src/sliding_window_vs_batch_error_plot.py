import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.core.metrics as eval_metrics

from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np

from copy import deepcopy

import matplotlib.pyplot as plt
from dynosam_utils.evaluation.core.plotting import startup_plotting, nice_colours

from enum import Enum

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding_compare"

# plt.rcParams['figure.facecolor'] = 'white'


# Reset all rcParams to their default values
# https://cduvallet.github.io/posts/2018/03/boxplots-in-python
# plt.rcdefaults()

# plt.rcParams.update({
#                     "text.usetex": True,
#                     "font.family": "serif",
#                     "font.serif": ["Computer Modern Roman"],
#                     })

# # plt.rcParams['axes.titlesize'] = 33    # Title font size
# # plt.rcParams['axes.labelsize'] = 30    # X/Y label font size
# # plt.rcParams['figure.titlesize'] = 30    # Title font size
# # plt.rcParams['xtick.labelsize'] = 30   # X tick label font size
# # plt.rcParams['ytick.labelsize'] = 30   # Y tick label font size
# # plt.rcParams['legend.fontsize']= 35


# font_size=35

# # # Change default font sizes.
# plt.rc('font', size=font_size)
# plt.rc('axes', titlesize=font_size)
# plt.rc('axes', labelsize=font_size)
# plt.rc('xtick', labelsize=0.6*font_size)
# plt.rc('ytick', labelsize=0.6*font_size)
# plt.rc('legend', fontsize=0.7*font_size)

plt.rcdefaults()
startup_plotting(24, line_width=3.0)

# # formatting_utils.startup_plotting(50)
# plt.rcParams["lines.linewidth"] = 4.0




def make_plot(trans_axes, rot_axes, batch_opt_folder_path, sliding_opt_folder_path, metrics_type:  eval_metrics.MetricType = eval_metrics.MetricType.rme,
              full_batch_colour=np.array(nice_colours["sky_blue"])/255.0,
              sliding_window_colour=np.array(nice_colours["vermillion"])/255.0):
    batch_motion_eval = eval.MotionErrorEvaluator(
        batch_opt_folder_path + "/rgbd_motion_world_backend_object_motion_log.csv",
        batch_opt_folder_path + "/rgbd_motion_world_backend_object_pose_log.csv")

    sliding_motion_eval = eval.MotionErrorEvaluator(
        sliding_opt_folder_path + "/rgbd_motion_world_backend_object_motion_log.csv",
        sliding_opt_folder_path + "/rgbd_motion_world_backend_object_pose_log.csv")

    # assert list(batch_motion_eval.object_motion_traj.keys()) == list(sliding_motion_eval.object_motion_traj.keys()), (list(batch_motion_eval.object_motion_traj.keys()), list(sliding_motion_eval.object_motion_traj.keys()))

    def collect_and_process_trajectory(batch_traj_pair: eval_metrics.TrajPair, sliding_traj_pair: eval_metrics.TrajPair, metricType: metrics.Metric):

        batch_errors_translation_per_frame = {}
        batch_errors_rot_per_frame = {}

        sliding_errors_translation_per_frame = {}
        sliding_errors_rot_per_frame = {}


        for object_id, first_batch_traj, second_batch_traj in eval.common_entries(batch_traj_pair[0], batch_traj_pair[1]):

            first_sliding_traj = sliding_traj_pair[0][object_id]
            second_sliding_traj = sliding_traj_pair[1][object_id]

            common_timestamps = np.intersect1d(first_sliding_traj.timestamps, first_batch_traj.timestamps)
            common_timestamps = common_timestamps[:-10]

            batch_ids = []
            sliding_ids = []
            # manually reduce to ids
            for timestamp in common_timestamps:
                batch_ids.append(int(np.where(first_batch_traj.timestamps == timestamp)[0][0]))
                sliding_ids.append(int(np.where(first_sliding_traj.timestamps == timestamp)[0][0]))

            first_sliding_traj.reduce_to_ids(sliding_ids)
            second_sliding_traj.reduce_to_ids(sliding_ids)
            first_batch_traj.reduce_to_ids(batch_ids)
            second_batch_traj.reduce_to_ids(batch_ids)

            batch_ape_trans = metricType(metrics.PoseRelation.translation_part)
            batch_ape_rot = metricType(metrics.PoseRelation.rotation_angle_deg)
            batch_data = (first_batch_traj, second_batch_traj)
            batch_ape_trans.process_data(batch_data)
            batch_ape_rot.process_data(batch_data)

            sliding_ape_trans = metricType(metrics.PoseRelation.translation_part)
            sliding_ape_rot = metricType(metrics.PoseRelation.rotation_angle_deg)
            sliding_data = (first_sliding_traj,second_sliding_traj)
            sliding_ape_trans.process_data(sliding_data)
            sliding_ape_rot.process_data(sliding_data)


            # assert sliding_ape_trans.error.shape == batch_ape_trans.error.shape, (sliding_ape_trans.error.shape, batch_ape_trans.error.shape)
            # assert sliding_ape_trans.error.shape[0] == len(common_timestamps), (sliding_ape_trans.error.shape[0],  len(common_timestamps))

            # update common timestamps to the ones produced by the error metric
            # hack ;)
            if isinstance(sliding_ape_trans, eval_metrics.RME):
                assert sliding_ape_trans.timestamps.shape == sliding_ape_rot.timestamps.shape
                common_timestamps = sliding_ape_trans.timestamps

            for index, timestamp in enumerate(common_timestamps):
                timestamp = int(timestamp)
                sliding_t_error = sliding_ape_trans.error[index]
                batch_t_error = batch_ape_trans.error[index]

                sliding_r_error = sliding_ape_rot.error[index]
                batch_r_error = batch_ape_rot.error[index]

                if timestamp not in batch_errors_translation_per_frame:
                    batch_errors_translation_per_frame[timestamp] = []
                batch_errors_translation_per_frame[timestamp].append(batch_t_error)

                if timestamp not in batch_errors_rot_per_frame:
                    batch_errors_rot_per_frame[timestamp] = []
                batch_errors_rot_per_frame[timestamp].append(batch_r_error)

                if timestamp not in sliding_errors_translation_per_frame:
                    sliding_errors_translation_per_frame[timestamp] = []
                sliding_errors_translation_per_frame[timestamp].append(sliding_t_error)

                if timestamp not in sliding_errors_rot_per_frame:
                    sliding_errors_rot_per_frame[timestamp] = []
                sliding_errors_rot_per_frame[timestamp].append(sliding_r_error)

        # get average at each frame and sort
        def get_average(error_per_frame):
            error_per_frame = deepcopy(error_per_frame)
            for k, v in error_per_frame.items():
                error_per_frame[k] = np.mean(v)

            keys = list(error_per_frame.keys())
            values = list(error_per_frame.values())
            #sort by keys (timestamp) and ensure that values remain in order with the timestamp
            sorted_tuple = [(y, x) for y,x in sorted(zip(keys,values))]
            sorted_timestamps, sorted_values = zip(*sorted_tuple)
            return sorted_timestamps, sorted_values


        batch_errors_timestamp, batch_errors_t = get_average(batch_errors_translation_per_frame)
        batch_errors_timestamp, batch_errors_r = get_average(batch_errors_rot_per_frame)
        sliding_errors_timestamp, sliding_errors_t = get_average(sliding_errors_translation_per_frame)
        sliding_errors_timestamp, sliding_errors_r = get_average(sliding_errors_rot_per_frame)

        # order to ensure they are in order!!! (they are are not is unclear...)

        assert batch_errors_timestamp == sliding_errors_timestamp, (batch_errors_timestamp, sliding_errors_timestamp)

        # trans_fig = plt.figure(figsize=(10,4))
        # ax = trans_fig.gca()
        trans_axes.plot(batch_errors_timestamp, batch_errors_t, label="Full-Batch", color=full_batch_colour)
        # trans_axes.set_ylabel("$E_t$(m)", fontsize=23)
        trans_axes.set_ylabel("ME$_t$(m)")


        trans_axes.plot(batch_errors_timestamp, sliding_errors_t, label="Sliding-Window", color=sliding_window_colour)
        trans_axes.patch.set_facecolor('white')
        trans_axes.margins(x=0)
        # Set the color and width of the border (spines)
        for spine in trans_axes.spines.values():
            spine.set_edgecolor('black')  # Set the color to black
            spine.set_linewidth(1)        # Set the border width (adjust as needed)


        rot_axes.set_ylabel("ME$_r$(\N{degree sign})")
        # rot_axes.set_xlabel("Frame Index [-]")
        # rot_axes.set_title("Batch vs. Sliding Window: AME$_r$ Error Comparison", fontweight="bold",  fontsize=23)
        rot_axes.plot(batch_errors_timestamp, batch_errors_r, label="Full-Batch", color=full_batch_colour)
        rot_axes.plot(batch_errors_timestamp, sliding_errors_r, label="Sliding-Window", color=sliding_window_colour)
        rot_axes.patch.set_facecolor('white')
        rot_axes.margins(x=0)
        # Set the color and width of the border (spines)
        for spine in rot_axes.spines.values():
            spine.set_edgecolor('black')  # Set the color to black
            spine.set_linewidth(1)        # Set the border width (adjust as needed)


        rot_axes.legend(loc="upper right")
        trans_axes.legend(loc="upper right")
        # print average errors
        print(f"Batch average t: {np.mean(batch_errors_t)}")
        print(f"Batch average r: {np.mean(batch_errors_r)}")
        print(f"Sliding average t: {np.mean(sliding_errors_t)}")
        print(f"Sliding average r: {np.mean(sliding_errors_r)}")



    print(str(metrics_type))
    if metrics_type == eval_metrics.MetricType.ame:
        collect_and_process_trajectory(
            (batch_motion_eval.object_motion_traj, batch_motion_eval.object_motion_traj_ref),
            (sliding_motion_eval.object_motion_traj, sliding_motion_eval.object_motion_traj_ref), eval_metrics.AME)

    elif metrics_type == eval_metrics.MetricType.rme:
        collect_and_process_trajectory(
            (batch_motion_eval.object_poses_traj_ref, batch_motion_eval.object_motion_traj),
            (sliding_motion_eval.object_poses_traj_ref, sliding_motion_eval.object_motion_traj), eval_metrics.RME)

    else:
        print("NOT IMPLEMENTED")
        return

# rot_fig = plt.figure(figsize=(13,9), layout='constrained')
# trans_fig = plt.figure(figsize=(13,9), layout='constrained')

# rot_axes_1 = rot_fig.add_subplot(211)
# # rot_axes_1.set_title(r"\textit{KITTI 00}", loc="left")
# rot_axes_1.set_title(r"Motion Error (rotation) on KITTI 00", loc="center")

# rot_axes_2 = rot_fig.add_subplot(212)
# rot_axes_2.set_title(r"Motion Error (rotation) on OMD (S4U)", loc="center")

# trans_axes_1 = trans_fig.add_subplot(211)
# trans_axes_1.set_title(r"Motion Error (translation) on KITTI 00", loc="center")

# trans_axes_2 = trans_fig.add_subplot(212)
# trans_axes_2.set_title(r"Motion Error (translation) on OMD (S4U)", loc="center")

# make_plot(trans_axes_1, rot_axes_1, "/root/results/TRO2025/kitti_0000", "/root/results/TRO2025/kitti_0000_sliding")
# make_plot(trans_axes_2, rot_axes_2, "/root/results/TRO2025/omd_swinging_4_unconstrained_batch", "/root/results/TRO2025/omd_swinging_4_unconstrained_sliding")


kitti_fig = plt.figure(figsize=(13,6), layout='constrained')
omd_fig = plt.figure(figsize=(13,6), layout='constrained')

rot_axes_1 = kitti_fig.add_subplot(211)
# rot_axes_1.set_title(r"\textit{KITTI 00}", loc="left")
# rot_axes_1.set_title(r"Motion Error (rotation) on KITTI 00", loc="center")

rot_axes_2 = omd_fig.add_subplot(211)
# rot_axes_2.set_title(r"Motion Error (rotation) on OMD (S4U)", loc="center")

trans_axes_1 = kitti_fig.add_subplot(212)
# trans_axes_1.set_title(r"Motion Error (translation) on KITTI 00", loc="center")

trans_axes_2 = omd_fig.add_subplot(212)
# trans_axes_2.set_title(r"Motion Error (translation) on OMD (S4U)", loc="center")

make_plot(trans_axes_1, rot_axes_1, "/root/results/TRO2025/kitti_0000", "/root/results/TRO2025/kitti_0000_sliding")
make_plot(trans_axes_2, rot_axes_2, "/root/results/TRO2025/omd_swinging_4_unconstrained_batch", "/root/results/TRO2025/omd_swinging_4_unconstrained_sliding",
          full_batch_colour=np.array(nice_colours["bluish_green"])/255.0,
          sliding_window_colour=np.array(nice_colours["reddish_purple"])/255.0)

# rot_fig.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper center",
            # mode="expand", borderaxespad=0, ncol=3)
# rot_fig.legend( loc="upper center", ncol=2, labels=["Full-Batch", "Sliding"], frameon=False,bbox_to_anchor=(0.5, 1.02))
# trans_fig.legend( loc="upper center", ncol=2, labels=["Full-Batch", "Sliding"], frameon=False,bbox_to_anchor=(0.5, 1.02))
# rot_fig.legend(loc='outside right upper')
# trans_fig.legend(loc='outside right upper')
# rot_fig.legend( ncol=2, labels=["Full-Batch", "Sliding-Window"], frameon=False,bbox_to_anchor=(1.0, 1.02))
# trans_fig.legend( ncol=2, labels=["Full-Batch", "Sliding-Window"], frameon=False,bbox_to_anchor=(1.0, 1.02))

#  Get the bounding boxes of all axes
bbox = rot_axes_1.get_position()
x_center = (bbox.x0 + bbox.x1) / 2
kitti_fig.suptitle("Per-frame Motion Error on  KITTI $00$", x=x_center, ha='center')

bbox = rot_axes_2.get_position()
x_center = (bbox.x0 + bbox.x1) / 2
omd_fig.suptitle("Per-frame Motion Error on OMD (S4U)",  x=x_center, ha='center')



# rot_fig.suptitle("Batch vs. Sliding Window: AME$_r$ Comparison", fontweight="bold", fontsize=30)
kitti_fig.supxlabel("Frame Index [-]")

# trans_fig.suptitle("Batch vs. Sliding Window: AME$_t$ Comparison",  fontweight="bold", fontsize=30)
omd_fig.supxlabel("Frame Index [-]")

# rot_fig.tight_layout(pad=0.1)
# trans_fig.tight_layout(pad=0.1)
# rot_fig.tight_layout()
# trans_fig.tight_layout()

# plt.show()

kitti_fig.savefig("/root/results/misc/batch_vs_sliding_kitti_combined.pdf", format="pdf")
omd_fig.savefig("/root/results/misc/batch_vs_sliding_omd_combined.pdf", format="pdf")
