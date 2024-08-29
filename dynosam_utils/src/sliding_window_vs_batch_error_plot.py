import evaluation.evaluation_lib as eval

from evo.core import lie_algebra, trajectory, metrics, transformations
import evo.tools.plot as evo_plot
import evo.core.units as evo_units
import numpy as np

import matplotlib.pyplot as plt

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding_compare"

# plt.rcParams['figure.facecolor'] = 'white'


# Reset all rcParams to their default values
plt.rcdefaults()



def make_plot(trans_axes, rot_axes, batch_opt_folder_path, sliding_opt_folder_path):
    batch_motion_eval = eval.MotionErrorEvaluator(
        batch_opt_folder_path + "/rgbd_motion_world_backend_object_motion_log.csv",
        batch_opt_folder_path + "/rgbd_motion_world_backend_object_pose_log.csv")

    sliding_motion_eval = eval.MotionErrorEvaluator(
        sliding_opt_folder_path + "/rgbd_motion_world_backend_object_motion_log.csv",
        sliding_opt_folder_path + "/rgbd_motion_world_backend_object_pose_log.csv")

    # assert list(batch_motion_eval.object_motion_traj.keys()) == list(sliding_motion_eval.object_motion_traj.keys()), (list(batch_motion_eval.object_motion_traj.keys()), list(sliding_motion_eval.object_motion_traj.keys()))

    batch_errors_translation_per_frame = {}
    batch_errors_rot_per_frame = {}

    sliding_errors_translation_per_frame = {}
    sliding_errors_rot_per_frame = {}


    for object_id, batch_object_traj, batch_object_traj_ref in eval.common_entries(batch_motion_eval.object_motion_traj, batch_motion_eval.object_motion_traj_ref):
        sliding_object_traj = sliding_motion_eval.object_motion_traj[object_id]
        sliding_object_traj_ref = sliding_motion_eval.object_motion_traj_ref[object_id]

        common_timestamps = np.intersect1d(sliding_object_traj.timestamps, batch_object_traj.timestamps)

        batch_ids = []
        sliding_ids = []
        # manually reduce to ids
        for timestamp in common_timestamps:
            batch_ids.append(int(np.where(batch_object_traj.timestamps == timestamp)[0][0]))
            sliding_ids.append(int(np.where(sliding_object_traj.timestamps == timestamp)[0][0]))

        sliding_object_traj.reduce_to_ids(sliding_ids)
        sliding_object_traj_ref.reduce_to_ids(sliding_ids)
        batch_object_traj.reduce_to_ids(batch_ids)
        batch_object_traj_ref.reduce_to_ids(batch_ids)


        batch_ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        batch_ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        batch_data = (batch_object_traj, batch_object_traj_ref)
        batch_ape_trans.process_data(batch_data)
        batch_ape_rot.process_data(batch_data)

        sliding_ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        sliding_ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        sliding_data = (sliding_object_traj,sliding_object_traj_ref)
        sliding_ape_trans.process_data(sliding_data)
        sliding_ape_rot.process_data(sliding_data)

        assert sliding_ape_trans.error.shape == batch_ape_trans.error.shape, (sliding_ape_trans.error.shape, batch_ape_trans.error.shape)
        assert sliding_ape_trans.error.shape[0] == len(common_timestamps)

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
    trans_axes.plot(batch_errors_timestamp, batch_errors_t, label="Batch")
    trans_axes.set_ylabel("$E_t$(m)", fontsize=19)
    # trans_axes.set_xlabel("Frame Index [-]")
    # trans_axes.set_title("Batch vs. Sliding Window: AME$_t$ Error Comparison", fontweight='heavy', fontsize=23)
    trans_axes.plot(batch_errors_timestamp, sliding_errors_t, label="Sliding")
    trans_axes.patch.set_facecolor('white')
    # Set the color and width of the border (spines)
    for spine in trans_axes.spines.values():
        spine.set_edgecolor('black')  # Set the color to black
        spine.set_linewidth(1)        # Set the border width (adjust as needed)
    # trans_fig.tight_layout()


    # rot_fig = plt.figure(figsize=(10,4))
    # ax = rot_fig.gca()

    rot_axes.set_ylabel("$E_r$(\N{degree sign})", fontsize=19)
    # rot_axes.set_xlabel("Frame Index [-]")
    # rot_axes.set_title("Batch vs. Sliding Window: AME$_r$ Error Comparison", fontweight="bold",  fontsize=23)
    rot_axes.plot(batch_errors_timestamp, batch_errors_r, label="Batch")
    rot_axes.plot(batch_errors_timestamp, sliding_errors_r, label="Sliding")
    rot_axes.patch.set_facecolor('white')
    # Set the color and width of the border (spines)
    for spine in rot_axes.spines.values():
        spine.set_edgecolor('black')  # Set the color to black
        spine.set_linewidth(1)        # Set the border width (adjust as needed)

    # plt.show()

    # rot_fig.savefig("/root/results/misc/swinging_unconstrained_4_batch_vs_sliding_rot.pdf", format="pdf")
    # trans_fig.savefig("/root/results/misc/swinging_unconstrained_4_batch_vs_sliding_trans.pdf", format="pdf")

    # rot_fig.savefig("/root/results/misc/kitti_0004_batch_vs_sliding_rot.pdf", format="pdf")
    # trans_fig.savefig("/root/results/misc/kitti_0004_batch_vs_sliding_trans.pdf", format="pdf")


    # rot_fig.savefig("/root/results/misc/kitti_0000_batch_vs_sliding_rot.pdf", format="pdf")
    # trans_fig.savefig("/root/results/misc/kitti_0000_batch_vs_sliding_trans.pdf", format="pdf")



# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0000_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/kitti_0004_sliding"

# batch_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained"
# sliding_opt_folder_path = "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding_compare"

# Set global font sizes (optional)
plt.rcParams['axes.titlesize'] = 25    # Title font size
plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size

rot_fig = plt.figure(figsize=(20,8))
trans_fig = plt.figure(figsize=(20,8))

rot_axes_1 = rot_fig.add_subplot(211)
rot_axes_1.set_title("KITTI 0000", loc="left")

rot_axes_2 = rot_fig.add_subplot(212)
rot_axes_2.set_title("OMD (swinging 4 unconstrained)", loc="left")

trans_axes_1 = trans_fig.add_subplot(211)
trans_axes_1.set_title("KITTI 0000", loc="left")

trans_axes_2 = trans_fig.add_subplot(212)
trans_axes_2.set_title("OMD (swinging 4 unconstrained)", loc="left")

make_plot(trans_axes_1, rot_axes_1, "/root/results/Dynosam_tro2024/kitti_0000", "/root/results/Dynosam_tro2024/kitti_0000_sliding")
make_plot(trans_axes_2, rot_axes_2, "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained", "/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding_compare")

rot_axes_1.legend(loc="upper right", fontsize=23)
rot_axes_2.legend(loc="upper right", fontsize=23)
trans_axes_1.legend(loc="upper right", fontsize=23)
trans_axes_2.legend(loc="upper right", fontsize=23)





rot_fig.suptitle("Batch vs. Sliding Window: AME$_r$ Comparison", fontweight="bold", fontsize=30)
rot_fig.supxlabel("Frame Index [-]",fontsize=27)

trans_fig.suptitle("Batch vs. Sliding Window: AME$_t$ Comparison",  fontweight="bold", fontsize=30)
trans_fig.supxlabel("Frame Index [-]", fontsize=27)

rot_fig.tight_layout(pad=0.5)
trans_fig.tight_layout(pad=0.5)

# plt.show()

rot_fig.savefig("/root/results/misc/batch_vs_sliding_rot_combined.pdf", format="pdf")
trans_fig.savefig("/root/results/misc/batch_vs_sliding_trans_combined.pdf", format="pdf")
