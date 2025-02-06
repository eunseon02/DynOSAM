import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from evo.core import lie_algebra
import evo.core.trajectory as evo_traj
import evo.tools.plot as evo_plot
import evo.core.sync as evo_sync

import dynosam_utils.evaluation.evaluation_lib as eval
import dynosam_utils.evaluation.formatting_utils as formatting_utils

import sys
import copy

# plt.rcdefaults()

# plt.rcParams['axes.titlesize'] = 25    # Title font size
# plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
# plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
# plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size
# plt.rcParams['legend.fontsize']=14


formatting_utils.startup_plotting(38)

"""
Script to process the results provided by the authors of
Multimotion Visual Odometry (MVO) and compare against DynoSAM

The raw results are not made public (sorry)

"""

# helper functions
# Function to compute the inverse of an SE(3) transform
def invT(Tij):
    Tij = copy.deepcopy(Tij)
    Cij = Tij[:-1, :-1]
    Cji = Cij.T
    r_j_to_i_in_i = Tij[:-1, -1]

    r_i_to_j_in_i = -Cji @ r_j_to_i_in_i

    Tji = np.vstack((np.hstack((Cji, r_i_to_j_in_i[:, np.newaxis])), [0, 0, 0, 1]))
    return Tji

# Function to extract the translation vector from the SE(3) transform
def get_r_j_from_i_in_i_FROM_T_ji(T_ji):
    T_ji = copy.deepcopy(T_ji)
    r_i_from_j_in_j = T_ji[:3, 3]

    C_ji = T_ji[:3, :3]
    r_j_from_i_in_i = -C_ji.T @ r_i_from_j_in_j
    return np.append(r_j_from_i_in_i, 1)


def is_so3(r: np.ndarray) -> bool:
    """
    :param r: a 3x3 matrix
    :return: True if r is in the SO(3) group
    """
    # Check the determinant.
    # det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-3)
    if not det_valid:
        print(f"Det valid {det_valid}")
    # Check if the transpose is the inverse.
    # result = r.transpose().dot(r)
    result = r.transpose() @ r

    inv_valid = np.allclose(result, np.eye(3), atol=1e-3)
    # inv_valid = np.allclose(result, np.eye(3), atol=1e-6)
    # print(f"Inv valid {inv_valid} -> {result}")
    if not inv_valid:
        print(f"Inv valid {inv_valid} -> {result}")


    return det_valid and inv_valid


def is_se3(p: np.ndarray) -> bool:
    """
    :param p: a 4x4 matrix
    :return: True if p is in the SE(3) group
    """
    rot_valid = is_so3(p[:3, :3])
    lower_valid = np.equal(p[3, :], np.array([0.0, 0.0, 0.0, 1.0])).all()
    return rot_valid and bool(lower_valid)


def so3_log(r: np.ndarray, return_skew: bool = False) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param return_skew: return skew symmetric Lie algebra element
    :return:
            rotation vector (axis * angle)
        or if return_skew is True:
             3x3 skew symmetric logarithmic map in so(3) (Ma, Soatto eq. 2.8)
    """
    if not is_so3(r):
        raise lie_algebra.LieAlgebraException("matrix is not a valid SO(3) group element")
    rotation_vector = lie_algebra.sst_rotation_from_matrix(r).as_rotvec()
    if return_skew:
        return lie_algebra.hat(rotation_vector)
    else:
        return rotation_vector


def so3_log_angle(r: np.ndarray, degrees: bool = False) -> float:
    """
    :param r: SO(3) rotation matrix
    :param degrees: whether to return in degrees, default is radians
    :return: the rotation angle of the logarithmic map
    """
    angle = np.linalg.norm(so3_log(r, return_skew=False))
    if degrees:
        angle = np.rad2deg(angle)
    return float(angle)

def closest_rotation_matrix(matrix):
    """
    Find the closest valid rotation matrix to the given matrix.

    :param matrix: The input 3x3 matrix (which may not be a valid rotation matrix)
    :return: The closest valid rotation matrix
    """
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)

    # Ensure that R has a determinant of +1, making it a proper rotation matrix
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    assert lie_algebra.is_so3(R)

    return R

def rpe(gt_traj, mvo_traj, dynosam_traj, dyno_label, mvo_errors, dynosam_errors):
    label = f"RPE {dyno_label}"
    mvo_errors[label] = { "rot": [], "translation": [] }
    dynosam_errors[label] = { "rot": [], "translation": [] }

    # for RPE average
    mean_label = "mean RPE"
    if mean_label not in mvo_errors:
        mvo_errors[mean_label] = { "rot": [], "translation": [] }
    if mean_label not in dynosam_errors:
        dynosam_errors[mean_label] = { "rot": [], "translation": [] }


    for gt_pose_k_1, gt_pose_k, object_pose_k_1, object_pose_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], mvo_traj.poses_se3[:-1], mvo_traj.poses_se3[1:]):
        gt_error = lie_algebra.se3_inverse(gt_pose_k_1) @ gt_pose_k
        est_error = lie_algebra.se3_inverse(object_pose_k_1) @ object_pose_k
        error =  lie_algebra.se3_inverse(gt_error) @ est_error

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        mvo_errors[label]["rot"].append(rot_error)
        mvo_errors[label]["translation"].append(t_error)

        mvo_errors[mean_label]["rot"].append(rot_error)
        mvo_errors[mean_label]["translation"].append(t_error)

    for gt_pose_k_1, gt_pose_k, object_pose_k_1, object_pose_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], dynosam_traj.poses_se3[:-1], dynosam_traj.poses_se3[1:]):
        gt_error = lie_algebra.se3_inverse(gt_pose_k_1) @ gt_pose_k
        est_error = lie_algebra.se3_inverse(object_pose_k_1) @ object_pose_k
        error =  lie_algebra.se3_inverse(gt_error) @ est_error

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        dynosam_errors[label]["rot"].append(rot_error)
        dynosam_errors[label]["translation"].append(t_error)

        dynosam_errors[mean_label]["rot"].append(rot_error)
        dynosam_errors[mean_label]["translation"].append(t_error)


def ape(gt_traj, mvo_traj, dynosam_traj, dyno_label, mvo_errors, dynosam_errors):
    label = f"APE {dyno_label}"
    mvo_errors[label] = { "rot": [], "translation": [] }
    dynosam_errors[label] = { "rot": [], "translation": [] }

    mean_label = "mean APE"
    if mean_label not in mvo_errors:
        mvo_errors[mean_label] = { "rot": [], "translation": [] }
    if mean_label not in dynosam_errors:
        dynosam_errors[mean_label] = { "rot": [], "translation": [] }


    for gt_pose_k, object_pose_k in zip(gt_traj.poses_se3, mvo_traj.poses_se3):
        error =  lie_algebra.se3_inverse(gt_pose_k) @ object_pose_k

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        mvo_errors[label]["rot"].append(rot_error)
        mvo_errors[label]["translation"].append(t_error)

        mvo_errors[mean_label]["rot"].append(rot_error)
        mvo_errors[mean_label]["translation"].append(t_error)

    for gt_pose_k, object_pose_k in zip(gt_traj.poses_se3, dynosam_traj.poses_se3):
        error =  lie_algebra.se3_inverse(gt_pose_k) @ object_pose_k

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        dynosam_errors[label]["rot"].append(rot_error)
        dynosam_errors[label]["translation"].append(t_error)

        dynosam_errors[mean_label]["rot"].append(rot_error)
        dynosam_errors[mean_label]["translation"].append(t_error)



def rme(gt_traj, mvo_motion_traj, dynosam_motion_traj, dyno_label, mvo_errors, dynosam_errors):
    label = f"RME {dyno_label}"
    mvo_errors[label] = { "rot": [], "translation": [] }
    dynosam_errors[label] = { "rot": [], "translation": [] }

    mean_label = "mean RME"
    if mean_label not in mvo_errors:
        mvo_errors[mean_label] = { "rot": [], "translation": [] }
    if mean_label not in dynosam_errors:
        dynosam_errors[mean_label] = { "rot": [], "translation": [] }

    for object_pose_k_1, object_pose_k, object_motion_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], mvo_motion_traj.poses_se3[1:]):
        error_in_L = lie_algebra.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1
        error_in_L = lie_algebra.se3_inverse(error_in_L)

        rot_error = abs(so3_log_angle(error_in_L[:3, :3], True))
        t_error = np.linalg.norm(error_in_L[:3, 3])

        mvo_errors[label]["rot"].append(rot_error)
        mvo_errors[label]["translation"].append(t_error)

        mvo_errors[mean_label]["rot"].append(rot_error)
        mvo_errors[mean_label]["translation"].append(t_error)

    # rme for dynosam
    for object_pose_k_1, object_pose_k, object_motion_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], dynosam_motion_traj.poses_se3[1:]):
        error_in_L = lie_algebra.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1
        error_in_L = lie_algebra.se3_inverse(error_in_L)

        rot_error = abs(so3_log_angle(error_in_L[:3, :3], True))
        t_error = np.linalg.norm(error_in_L[:3, 3])

        dynosam_errors[label]["rot"].append(rot_error)
        dynosam_errors[label]["translation"].append(t_error)

        dynosam_errors[mean_label]["rot"].append(rot_error)
        dynosam_errors[mean_label]["translation"].append(t_error)


def ame(gt_motion_traj, mvo_motion_traj, dynosam_motion_traj, dyno_label, mvo_errors, dynosam_errors):
    label = f"AME {dyno_label}"
    mvo_errors[label] = { "rot": [], "translation": [] }
    dynosam_errors[label] = { "rot": [], "translation": [] }

    mean_label = "mean AME"
    if mean_label not in mvo_errors:
        mvo_errors[mean_label] = { "rot": [], "translation": [] }
    if mean_label not in dynosam_errors:
        dynosam_errors[mean_label] = { "rot": [], "translation": [] }

    for gt_pose_k, object_pose_k in zip(gt_motion_traj.poses_se3, mvo_motion_traj.poses_se3):
        error =  lie_algebra.se3_inverse(gt_pose_k) @ object_pose_k

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        mvo_errors[label]["rot"].append(rot_error)
        mvo_errors[label]["translation"].append(t_error)

        mvo_errors[mean_label]["rot"].append(rot_error)
        mvo_errors[mean_label]["translation"].append(t_error)

    for gt_pose_k, object_pose_k in zip(gt_motion_traj.poses_se3, dynosam_motion_traj.poses_se3):
        error =  lie_algebra.se3_inverse(gt_pose_k) @ object_pose_k

        rot_error = abs(so3_log_angle(error[:3, :3], True))
        t_error = np.linalg.norm(error[:3, 3])

        dynosam_errors[label]["rot"].append(rot_error)
        dynosam_errors[label]["translation"].append(t_error)

        dynosam_errors[mean_label]["rot"].append(rot_error)
        dynosam_errors[mean_label]["translation"].append(t_error)




#hack do icp alignment
def do_icp_trajectory_alignment(traj_est: evo_traj.PoseTrajectory3D, traj_ref: evo_traj.PoseTrajectory3D):
    import open3d as o3d

    ref_points = traj_ref.positions_xyz
    est_poitns = traj_est.positions_xyz

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points)

    est_pcd = o3d.geometry.PointCloud()
    est_pcd.points = o3d.utility.Vector3dVector(est_poitns)


    threshold = 1.0

    reg_p2p = o3d.pipelines.registration.registration_icp(
        est_pcd, ref_pcd, threshold, lie_algebra.se3(),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    traj_est = copy.deepcopy(traj_est)
    traj_est.transform(reg_p2p.transformation)
    print(reg_p2p.transformation)

    return traj_est

def process_mvo_data(path_to_mvo_mat, path_to_dyno_results, camera_id, mvo_to_dyno_labels, dataset, output_results_file):

    #dataset: "kitti" or "omd"

    if dataset == "kitti":
        frame_id_to_timestamp_csv = eval.read_csv(
            "/root/results/DynoSAM/test_kitti/frame_id_timestamp.csv",
            ["frame_id", "timestamp [ns]"]
        )
    elif dataset == "omd":
        frame_id_to_timestamp_csv = eval.read_csv(
            "/root/results/DynoSAM/test_omd/frame_id_timestamp.csv",
            ["frame_id", "timestamp [ns]"]
        )
    else:
        print(f"Unknown dataset {dataset}")
        sys.exit(0)


    print(f"Loading MVO data results from {path_to_mvo_mat}")

    # for OMD, we know we both use the same timestamps so just update later with the ones from MVO
    frame_id_timestamp_dict = {}
    for data in frame_id_to_timestamp_csv:
        frame_id = int(data["frame_id"])
        timestamp = float(data["timestamp [ns]"]) / 1e9 # to seconds
        frame_id_timestamp_dict[frame_id] = timestamp

    data = sio.loadmat(path_to_mvo_mat)
    motions = data['motions']
    T_camera_to_ori = data['T_camera_to_ori']

    def process_single_motion(motion, trajectories, T_camera_to_ori):
        trajectory = motion['trajectory']
        validity = motion['validity']

        print(len(trajectory))

        # originally in nano seconds, but dynosam does everthing in seconds
        timestamps = motion["timestamps"]
        # sys.exit(0)

        # np.savetxt("og_timestamps.csv", np.array(timestamps))

        invalidities = np.concatenate(([0], np.where(np.diff(validity[:, 0]) != 0)[0] + 1, [len(trajectory)]))
        pts = np.zeros((4, len(trajectory)))
        pose_in_world = []

        for i in range(len(trajectory)):

            trajectory_T = trajectory[i][0]
            T = T_camera_to_ori @ trajectory_T @ motion['T_l_1_C_1'] @ invT(T_camera_to_ori)
            t_world = get_r_j_from_i_in_i_FROM_T_ji(T)
            C_ji = T[:3, :3]

            # VERY IMPORTANT to transpose
            R_world = C_ji.T
            pts[:, i] = t_world

            T_se3 = lie_algebra.se3(R_world, t_world[:3])
            # assert lie_algebra.is_so3(R_world), R_world
            # assert lie_algebra.is_se3(T_se3), T_se3
            pose_in_world.append(T_se3)

        print(f"Timestamps shape {timestamps.shape}")
        print(f"Poses in world len {len(pose_in_world)}")
        assert timestamps.shape[1] == len(pose_in_world)


        colorOrder = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i in range(1, len(invalidities)):
            if i % 2 == 0:
                trajs = pose_in_world[invalidities[i-1]:invalidities[i]]
                times = timestamps[0][invalidities[i-1]:invalidities[i]]

                for traj, time in zip(trajs, times):
                    time /= 1e6
                    trajectories[motion['id'][0][0]]["motions"].append(traj)
                    trajectories[motion['id'][0][0]]["times"].append(time)

            else:
                trajs = pose_in_world[invalidities[i-1]:invalidities[i]]
                times = timestamps[0][invalidities[i-1]:invalidities[i]]

                for traj, time in zip(trajs, times):
                    time /= 1e6
                    trajectories[motion['id'][0][0]]["motions"].append(traj)
                    trajectories[motion['id'][0][0]]["times"].append(time)



    # make dictionary of trajectories with key being each object id (int)
    trajectories = {}

    # and also process
    for motion in motions[0]:
        label = motion['id'][0][0]
        trajectories[label] = {}
        trajectories[label]["motions"] = []
        trajectories[label]["times"] = []
        process_single_motion(motion, trajectories, T_camera_to_ori)

    # post process trajectories into evo trajectories
    mvo_trajectories = {}

    start_frame = (motion["start_frame"])[0]
    end_frame = (motion["end_frame"])[0]

    print(start_frame)
    print(end_frame)

    for key, m in trajectories.items():
        traj = m["motions"]
        times = m["times"]

        assert len(times) == len(traj)

        # find valid indicies - some rotations seem to be invalid... not sure how this is possible ;)
        valid_indices = []
        invalid_indices = []
        for index, t in enumerate(traj):
            if lie_algebra.is_se3(t):
                valid_indices.append(index)
            else:
                invalid_indices.append(index)



        traj = np.array(traj)[valid_indices]
        times = np.array(times)[valid_indices]

        if dataset == "kitti":
        # for KITTI
        # times from MVO are offset and are 0.1 seconds apart leading to a total time of 15 seconds
        # our dataset is 0.05 seconds apart per frame leading to a total time of 7 seconds
            mvo_times = (times - times[0])/2
            # the first timestamp also starts at 0, which we throw out
            mvo_times = mvo_times[1:]
            traj = traj[1:]

        #FOR mvo
        # mvo_times = times
        else:
            mvo_times = np.arange(start_frame, end_frame,1, dtype=np.float64)




        pose_trajectory = evo_traj.PoseTrajectory3D(poses_se3=traj, timestamps=mvo_times)
        print(f"MVO object {key} trajectory check {pose_trajectory.check()}")


        # relabel keys so that the first object is 1 as dynosam has the camera t 0
        mvo_trajectories[key+1] = pose_trajectory

    # # hardcoded camera id
    # mvo_camera_traj = None
    # if camera_id is not None:
    mvo_camera_traj = mvo_trajectories[camera_id+1]
    del mvo_trajectories[camera_id+1]

    mvo_trajectories_plotting = {}
    for k, m in mvo_trajectories.items():
        mvo_trajectories_plotting[f"MVO {k}"] = copy.deepcopy(m)

    # eval.tools.plot_object_trajectories(
    #     evo_figure, mvo_trajectories_plotting, plot_mode = evo_plot.PlotMode.xyz
    # )


    # print(frame_id_timestamp_dict[1])
    # print(mvo_times[0])
    # sys.exit(0)


    # # get corresponding dynosam data
    dynosam_motion_eval = eval.MotionErrorEvaluator(
        path_to_dyno_results + "/rgbd_motion_world_backend_object_motion_log.csv",
        path_to_dyno_results + "/rgbd_motion_world_backend_object_pose_log.csv")

    dynosam_camera_eval = eval.CameraPoseEvaluator(
        path_to_dyno_results + "/rgbd_motion_world_backend_camera_pose_log.csv"
    )

    def update_dynosam_trajectory_with_timestamps(traj: evo_traj.PoseTrajectory3D):
        if dataset == "omd":
            traj.reduce_to_time_range(start_frame, end_frame)
            return traj
        else:
            original_timestamp = traj.timestamps #they are actually the frames
            poses = traj.poses_se3
            new_timestamps = []
            for frame in original_timestamp:
                assert int(frame) in frame_id_timestamp_dict, frame
                new_timestamps.append(frame_id_timestamp_dict[int(frame)])
            return evo_traj.PoseTrajectory3D(poses_se3=np.array(poses), timestamps=new_timestamps)

    def update_object_trajdict_with_timestamps(trajs: eval.ObjectTrajDict):
        for object_id, traj in trajs.items():
            trajs.update({object_id: update_dynosam_trajectory_with_timestamps(traj)})

    object_poses_traj_ref = copy.deepcopy(dynosam_motion_eval.object_poses_traj_ref)
    object_motions_traj_ref = copy.deepcopy(dynosam_motion_eval.object_motion_traj_ref)
    dyno_sam_object_poses_traj = copy.deepcopy(dynosam_motion_eval.object_poses_traj)
    dynosam_camera_pose_traj = dynosam_camera_eval.camera_pose_traj
    gt_camera_pose = dynosam_camera_eval.camera_pose_traj_ref

    update_object_trajdict_with_timestamps(object_poses_traj_ref)
    update_object_trajdict_with_timestamps(object_motions_traj_ref)
    update_object_trajdict_with_timestamps(dyno_sam_object_poses_traj)
    dynosam_camera_pose_traj = update_dynosam_trajectory_with_timestamps(dynosam_camera_pose_traj)
    gt_camera_pose = update_dynosam_trajectory_with_timestamps(gt_camera_pose)




    # print(dynosam_camera_pose_traj.timestamps[0])
    # print(mvo_camera_traj.timestamps[0])

    # sys.exit(0)



    #combine all trajs
    all_traj = mvo_trajectories_plotting
    for k, m in object_poses_traj_ref.items():
        all_traj[f"GT {k}"] = m
    for k, m in dyno_sam_object_poses_traj.items():
        all_traj[f"DynoSAM {k}"] = m



    mvo_errors = {}
    dynosam_errors = {}


    from evaluation.formatting_utils import nice_colours

    mvo_camera_traj, gt_camera_pose = evo_sync.associate_trajectories(
        copy.deepcopy(mvo_camera_traj),
        copy.deepcopy(gt_camera_pose),
        max_diff=0.05)

    dynosam_camera_pose_traj, gt_camera_pose = evo_sync.associate_trajectories(
        copy.deepcopy(dynosam_camera_pose_traj),
        copy.deepcopy(gt_camera_pose),
        max_diff=0.05)


    mvo_camera_traj_original = copy.deepcopy(mvo_camera_traj)
    mvo_camera_traj.align_origin(gt_camera_pose)
    dynosam_camera_pose_traj.align_origin(gt_camera_pose)



    rpe(gt_camera_pose, mvo_camera_traj, dynosam_camera_pose_traj, "camera", mvo_errors, dynosam_errors)
    ape(gt_camera_pose, mvo_camera_traj, dynosam_camera_pose_traj, "camera", mvo_errors, dynosam_errors)

    # plus 1 for camera
    num_objects = len(mvo_to_dyno_labels) + 1


    evo_figure = plt.figure(figsize=(8,8))

    if dataset == "kitti":
        axis_marker_scale=1.0
    else:
        axis_marker_scale=0.1

    # camera_traj_figure = plt.figure(figsize=(8,8))
    # ax = evo_figure.add_subplot(111, projection="3d")
    ax = evo_figure.add_subplot(111)

    ax.set_ylabel(r"Y(m)")
    ax.set_xlabel(r"X(m)")
    ax.set_title(f"Camera Trajectory")
    # ax.view_init(azim=-90, elev=90)
    # ax.patch.set_facecolor('white')
    # ax.set_facecolor('white')
    eval.tools.set_clean_background(ax)
    eval.tools.plot_object_trajectories(
        evo_figure, {"MVO" : mvo_camera_traj, "DynoSAM": dynosam_camera_pose_traj},
        plot_mode = evo_plot.PlotMode.xy,
        plot_axis_est=False, plot_axis_ref=False,
        axis_marker_scale=axis_marker_scale,
        plot_start_end_markers=False,
        colours=[
            np.array(nice_colours["blue"]) / 255.0,
            np.array(nice_colours["bluish_green"]) / 255.0
        ],
        traj_linewidth=4.0
        # downscale=0.005,
        # axis_marker_scale=1.0
    )
    eval.tools.plot_object_trajectories(
        evo_figure,
        {"Ground Truth": gt_camera_pose},
        plot_mode = evo_plot.PlotMode.xy,
        plot_start_end_markers=False,
        plot_axis_est=False, plot_axis_ref=False,
        colours=[np.array(nice_colours["vermillion"]) / 255.0],
        est_style="--",
        # downscale=0.005,
        axis_marker_scale=axis_marker_scale,
        traj_linewidth=6.0
    )
    # ax.set_aspect(0.8)
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    evo_figure.tight_layout(pad=0.05)
    ax.grid(which='major', color='#DDDDDD', linewidth=1.0)


    for idx, (mvo_label, dyno_label) in enumerate(mvo_to_dyno_labels.items()):
        # for plotting indices
        idx+=1
        mvo_traj = mvo_trajectories[mvo_label]
        dynosam_traj = dyno_sam_object_poses_traj[dyno_label]
        gt_traj = object_poses_traj_ref[dyno_label]
        gt_motion_traj = object_motions_traj_ref[dyno_label]

        print(f"GT traj {gt_traj}")

        print(f"MVO traj {mvo_traj}")
        if dataset == "omd":
            aligned_mvo_poses = []
            for aligned_camera, original_camera, mvo_pose in zip(mvo_camera_traj.poses_se3, mvo_camera_traj_original.poses_se3, mvo_traj.poses_se3):
                mv_camera_relative = np.linalg.inv(original_camera) @ mvo_pose
                aligned_mvo_pose = aligned_camera @ mv_camera_relative
                aligned_mvo_poses.append(aligned_mvo_pose)

            mvo_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(aligned_mvo_poses), timestamps=mvo_traj.timestamps)

        all_mvo_traj = {}
        all_mvo_motion_traj = {}
        all_gt_traj = {}
        all_gt_motion_traj = {}
        all_dyno_traj = {}

        print(f"MVO traj {mvo_traj}")

        if dataset == "kitti":
            mvo_traj, gt_traj = evo_sync.associate_trajectories(
                mvo_traj,
                gt_traj,
                max_diff=0.05)

            dynosam_traj, gt_traj = evo_sync.associate_trajectories(
                dynosam_traj,
                gt_traj,
                max_diff=0.05)
        else:
            mvo_traj = eval.tools.align_trajectory(mvo_traj, gt_traj)
            dynosam_traj = eval.tools.align_trajectory(dynosam_traj, gt_traj)




        # calculate motion for mvo in world
        mvo_motions = []
        # recalc motions after alignment
        motion_gt = []
        #
        motions_dynosam = []
        aligned_mvo_poses = []

        mvo_poses = mvo_traj.poses_se3
        for index, (gt_pose_k_1, gt_pose_k, pose_k_1, pose_k) in enumerate(zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], mvo_poses[:-1], mvo_poses[1:])):
            #this world is the mvo world
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            mvo_motions.append(H_world)


        dynosam_poses = dynosam_traj.poses_se3
        for index, (pose_k_1, pose_k) in enumerate(zip(dynosam_poses[:-1], dynosam_poses[1:])):
            assert lie_algebra.is_se3(pose_k_1)
            assert lie_algebra.is_se3(pose_k)
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            motions_dynosam.append(H_world)

        gt_poses = gt_traj.poses_se3
        for index, (pose_k_1, pose_k) in enumerate(zip(gt_poses[:-1], gt_poses[1:])):
            assert lie_algebra.is_se3(pose_k_1)
            assert lie_algebra.is_se3(pose_k)
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            motion_gt.append(H_world)

        print(f"GT traj {gt_traj}")


        mvo_motion_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(mvo_motions), timestamps=mvo_traj.timestamps[1:])
        dynosam_motion_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(motions_dynosam), timestamps=dynosam_traj.timestamps[1:])

        # print(motions)

        # rme for mvo
        rme(gt_traj, mvo_motion_traj, dynosam_motion_traj, dyno_label, mvo_errors, dynosam_errors)
        ame(gt_motion_traj, mvo_motion_traj, dynosam_motion_traj, dyno_label, mvo_errors, dynosam_errors)
        rpe(gt_traj, mvo_traj, dynosam_traj, dyno_label, mvo_errors, dynosam_errors)



        all_mvo_traj["MVO"] = mvo_traj
        all_gt_traj["Ground Truth"] = gt_traj
        all_dyno_traj["DynoSAM"] = dynosam_traj

        evo_figure = plt.figure(figsize=(8,8))
        plot_collection = evo_plot.PlotCollection()


        plot_collection.add_figure("MVO trjactories", evo_figure)
        # plot_collection.add_figure("Dynosam trjactories", dynosam_fig)

        # ax = evo_figure.add_subplot(111, projection="3d")
        ax = evo_figure.add_subplot(111)

        ax.set_ylabel(r"Y(m)")
        ax.set_xlabel(r"X(m)")
        ax.set_title(f"Object {dyno_label}")
        # ax.set_facecolor('white')
        eval.tools.set_clean_background(ax)


        eval.tools.plot_object_trajectories(
            evo_figure, all_mvo_traj,
            # None,
            plot_mode = evo_plot.PlotMode.xy,
            plot_axis_est=False,
            plot_start_end_markers=False,
            downscale=0.1,
            # est_name_prefix="MVO",
            colours=[np.array(nice_colours["blue"]) / 255.0],
            traj_linewidth=6.0
            # shift_ref_colour=100,
            # shift_est_colour=300
            # axis_marker_scale=1.0
        )

        trajectory_helper = eval.TrajectoryHelper()
        # trajectory_helper.append(camera_traj)
        trajectory_helper.append(mvo_traj)
        trajectory_helper.append(dynosam_traj)
        # trajectory_helper.set_ax_limits(ax, evo_plot.PlotMode.xy, buffer=2)

        ax = eval.tools.plot_object_trajectories(
            evo_figure,all_gt_traj,
            # None,
            plot_mode = evo_plot.PlotMode.xy,
            plot_axis_est=False,
            plot_start_end_markers=False,
            downscale=0.1,
            # est_name_prefix="Ground Truth",
            # shift_ref_colour=100,
            colours=[np.array(nice_colours["vermillion"]) / 255.0],
            est_style="--",
            traj_linewidth=6.0
            # shift_est_colour=300
            # axis_marker_scale=1.0
        )
        ax = eval.tools.plot_object_trajectories(
            evo_figure, all_dyno_traj, #all_gt_traj,
            plot_mode = evo_plot.PlotMode.xy,
            plot_axis_est=False, plot_axis_ref=True,
            plot_start_end_markers=False,
            colours=[np.array(nice_colours["bluish_green"]) / 255.0],
            traj_linewidth=6.0
            # downscale=0.5
            # axis_marker_scale=1.0
        )
        ax.set_box_aspect(1)
        # ax.se/t_aspect('equal')
        # ax.set_aspect('equal', adjustable='box')
        # ax.set_aspect('equal', 'box')
        # ax.set_aspect(0.8)
        ax.set_axisbelow(True)
        ax.grid(which='major', color='#DDDDDD', linewidth=1.0)
        evo_figure.tight_layout(pad=0.05)

        plt.show()

    import math
    for object_id, error in mvo_errors.items():
        rot_error = np.array(error["rot"])
        t_error = np.array(error["translation"])

        rmse_rot = math.sqrt(np.mean(np.power(rot_error, 2)))
        rmse_t = math.sqrt(np.mean(np.power(t_error, 2)))

        print(f"MVO Errors for object id {object_id} -> rot {rmse_rot}, trans {rmse_t}")

    for object_id, error in dynosam_errors.items():
        rot_error = np.array(error["rot"])
        t_error = np.array(error["translation"])

        rmse_rot = math.sqrt(np.mean(np.power(rot_error, 2)))
        rmse_t = math.sqrt(np.mean(np.power(t_error, 2)))

        print(f"Dynosam Errors for object id {object_id} -> rot {rmse_rot}, trans {rmse_t}")













if __name__ == '__main__':##
    mvo_to_dyno_labels_swinging_dynamic = {
        1 : 1,
        2:  2,
        3 : 3,
        4 : 4
    }
    # process_mvo_data(
    #     "/root/data/mvo_data_scripts_IJRR/swinging_dynamic_wnoa.mat",
    #     # "/root/results/DynoSAM/test_omd_long",
    #     "/root/results/DynoSAM/omd_vo_test",
    #     4,
    #     mvo_to_dyno_labels_swinging_dynamic,
    #     "omd",
    #     "/root/results/Dynosam_tro2024/mvo_analysis_swinging_dynamic_wnoa")

    # for this sequence object id 0 is the camera
    mvo_to_dyno_labels_kitti = {
        2 : 1,
        3:  2,
    }
    process_mvo_data("/root/data/mvo_data_scripts_IJRR/kitti_0005_wnoa.mat", "/root/results/DynoSAM/test_kitti_main/", 0, mvo_to_dyno_labels_kitti, "kitti", "/root/results/Dynosam_tro2024/mvo_analysis_kitti_0000")
