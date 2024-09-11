import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from evo.core import lie_algebra
import evo.core.trajectory as evo_traj
import evo.tools.plot as evo_plot
import evo.core.sync as evo_sync

import evaluation.evaluation_lib as eval
import evaluation.formatting_utils as formatting_utils

import copy

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
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    if not det_valid:
        print(f"Det valid {det_valid}")
    # Check if the transpose is the inverse.
    # result = r.transpose().dot(r)
    result = r.transpose() @ r

    inv_valid = np.allclose(result, np.eye(3), atol=1e-5)
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


# frame_id_to_timestamp_csv = eval.read_csv(
#     "/root/results/DynoSAM/test_omd/frame_id_timestamp.csv",
#     ["frame_id", "timestamp [ns]"]
# )
frame_id_to_timestamp_csv = eval.read_csv(
    "/root/results/DynoSAM/test_kitti/frame_id_timestamp.csv",
    ["frame_id", "timestamp [ns]"]
)



def process_mvo_data(path_to_mvo_mat, path_to_dyno_results, camera_id, mvo_to_dyno_labels, output_results_file):

    print(f"Loading MVO dataresults from {path_to_mvo_mat}")

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

        invalidities = np.concatenate(([0], np.where(np.diff(validity[:, 0]) != 0)[0] + 1, [len(trajectory)]))
        pts = np.zeros((4, len(trajectory)))
        pose_in_world = []

        for i in range(len(trajectory)):

            trajectory_T = trajectory[i][0]
            T = T_camera_to_ori @ trajectory_T @ motion['T_l_1_C_1'] @ invT(T_camera_to_ori)

            # if not lie_algebra.is_se3(T):
            #     T[:3, :3] = closest_rotation_matrix(T[:3, :3])

            # assert lie_algebra.is_se3(T)

            # print(T.dtype)
            # T_inv = invT(T)
            t_world = get_r_j_from_i_in_i_FROM_T_ji(T)
            # R_world = np.linalg.inv(-T[:3, :3].transpose())
            C_ji = T[:3, :3]
            R_world = C_ji
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
                # axes.plot(pts[0, invalidities[i-1]:invalidities[i]],
                #          pts[1, invalidities[i-1]:invalidities[i]],
                #          pts[2, invalidities[i-1]:invalidities[i]],
                #          '-', color=colorOrder[motion['id'][0][0] % len(colorOrder)], label=motion['id'][0][0])
                trajs = pose_in_world[invalidities[i-1]:invalidities[i]]
                times = timestamps[0][invalidities[i-1]:invalidities[i]]

                for traj, time in zip(trajs, times):
                    time /= 1e6
                    trajectories[motion['id'][0][0]]["motions"].append(traj)
                    trajectories[motion['id'][0][0]]["times"].append(time)

                # for t in trajs:
                #     trajectories[motion['id'][0][0]]["motions"].append(t)
                # trajectories[motion['id'][0][0]]["times"].extend(times)

            else:
                # axes.plot(pts[0, invalidities[i-1]:invalidities[i]],
                #          pts[1, invalidities[i-1]:invalidities[i]],
                #          pts[2, invalidities[i-1]:invalidities[i]],
                #          '--', color=colorOrder[motion['id'][0][0] % len(colorOrder)], label=motion['id'][0][0])
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
        print(f"Adding motion {label}")
        trajectories[label] = {}
        trajectories[label]["motions"] = []
        trajectories[label]["times"] = []
        process_single_motion(motion, trajectories, T_camera_to_ori)

    # post process trajectories into evo trajectories
    mvo_trajectories = {}

    for key, m in trajectories.items():
        traj = m["motions"]
        times = m["times"]

        assert len(times) == len(traj)

        # find valid indicies - some rotations seem to be invalid... not sure how this is possible ;)
        valid_indices = []
        for index, t in enumerate(traj):
            if lie_algebra.is_se3(t):
                valid_indices.append(index)

        traj = np.array(traj)[valid_indices]
        times = np.array(times)[valid_indices]

        # print(times)


        pose_trajectory = evo_traj.PoseTrajectory3D(poses_se3=traj, timestamps=times)
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

    evo_figure = plt.figure(figsize=(8,8))

    dynosam_fig = plt.figure(figsize=(8,8))
    # eval.tools.plot_object_trajectories(
    #     evo_figure, mvo_trajectories_plotting, plot_mode = evo_plot.PlotMode.xyz
    # )

    plot_collection = evo_plot.PlotCollection()

    # # get corresponding dynosam data
    dynosam_motion_eval = eval.MotionErrorEvaluator(
        path_to_dyno_results + "/rgbd_motion_world_backend_object_motion_log.csv",
        path_to_dyno_results + "/rgbd_motion_world_backend_object_pose_log.csv")

    dynosam_camera_eval = eval.CameraPoseEvaluator(
        path_to_dyno_results + "/rgbd_motion_world_backend_camera_pose_log.csv"
    )

    def update_dynosam_trajectory_with_timestamps(traj: evo_traj.PoseTrajectory3D):
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

    update_object_trajdict_with_timestamps(object_poses_traj_ref)
    update_object_trajdict_with_timestamps(object_motions_traj_ref)
    update_object_trajdict_with_timestamps(dyno_sam_object_poses_traj)
    dynosam_camera_pose_traj = update_dynosam_trajectory_with_timestamps(dynosam_camera_pose_traj)





    #combine all trajs
    all_traj = mvo_trajectories_plotting
    for k, m in object_poses_traj_ref.items():
        all_traj[f"GT {k}"] = m
    for k, m in dyno_sam_object_poses_traj.items():
        all_traj[f"DynoSAM {k}"] = m


    # eval.tools.plot_object_trajectories(
    #     evo_figure,all_traj, plot_mode = evo_plot.PlotMode.xyz
    # )

    # evo_plot.trajectories(evo_figure, dynosam_motion_eval.object_poses_traj, plot_mode=evo_plot.PlotMode.xyz)

    plot_collection.add_figure("MVO trjactories", evo_figure)
    plot_collection.add_figure("Dynosam trjactories", dynosam_fig)

    # plot_collection.show()

    # after synchronisation with the dynosam gt data
    all_mvo_traj = {}
    all_mvo_motion_traj = {}
    all_gt_traj = {}
    all_gt_motion_traj = {}


    all_dyno_traj = {}

    mvo_errors = {}
    dynosam_errors = {}

    # mvo_camera_traj_aligned = eval.tools.align_trajectory(mvo_camera_traj, dynosam_camera_pose_traj, correct_scale = False)
    mvo_camera_traj_aligned = copy.deepcopy(mvo_camera_traj)
    mvo_camera_traj_aligned.align_origin(dynosam_camera_pose_traj)


    # print(mvo_camera_traj_aligned.timestamps[0])
    # print(dynosam_camera_pose_traj.timestamps[0])

    # print(mvo_camera_traj_aligned.poses_se3[0])
    # print(dynosam_camera_pose_traj.poses_se3[0])


    for mvo_label, dyno_label in mvo_to_dyno_labels.items():
        mvo_traj = mvo_trajectories[mvo_label]
        dynosam_traj = dyno_sam_object_poses_traj[dyno_label]
        gt_traj = object_poses_traj_ref[dyno_label]
        gt_motion_traj = object_motions_traj_ref[dyno_label]

        mvo_traj_orginal = copy.deepcopy(mvo_traj)
        # return

        # print(mvo_traj.poses_se3[0])
        # print(dynosam_traj.poses_se3[0])
        # import sys
        # sys.exit(0)

        # mvo_camera_traj_aligned, mvo_traj = evo_sync.associate_trajectories(mvo_camera_traj_aligned, mvo_traj)
        # mvo_camera_traj_aligned, mvo_traj = evo_sync.associate_trajectories(mvo_camera_traj_aligned, mvo_traj)


        # align mvo camera traj with dynosam camera traj to set the world frame
        aligned_mvo_poses = []
        first_aligned_mvo_pose = None
        for mvo_pose_world, mvo_camera_world, mvo_camera_world_aligned in zip(mvo_traj.poses_se3, mvo_camera_traj.poses_se3, mvo_camera_traj_aligned.poses_se3):
            mvo_pose_camera = lie_algebra.se3_inverse(mvo_camera_world) @ mvo_pose_world
            mvo_pose_world_aligned = mvo_camera_world_aligned @ mvo_pose_camera
            aligned_mvo_poses.append(mvo_pose_world_aligned)

            last_pose = aligned_mvo_poses[-1]
            first_pose = aligned_mvo_poses[0]

            if first_aligned_mvo_pose is None:
                first_aligned_mvo_pose = first_pose

            aligned_mvo_poses[-1] = lie_algebra.se3_inverse(first_aligned_mvo_pose) @ last_pose




        # #calculate realigned mvo traj
        # realigned_mvo_poses = [aligned_mvo_poses[0]]
        # for mvo_pose in realigned_mvo_poses[1:]:
        #     aligned_mvo_poses


        # aligned_mvo_poses = []
        # for mvo_object_pose_world, mvo_camera_world, mvo_camera_world_aligned in zip(mvo_traj.poses_se3, mvo_camera_traj.poses_se3, mvo_camera_traj_aligned.poses_se3):
        #     mvo_pose_camera = lie_algebra.se3_inverse(mvo_camera_world) @ mvo_pose_world
        #     mvo_pose_world_aligned = mvo_camera_world_aligned @ mvo_pose_camera
        #     aligned_mvo_poses.append(mvo_pose_world_aligned)


        # mvo_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(aligned_mvo_poses), timestamps=mvo_traj.timestamps)
        print(f"Before mvo traj {mvo_traj}")

        # gt_traj, mvo_traj = evo_sync.associate_trajectories(
        #     copy.deepcopy(gt_traj),
        #     copy.deepcopy(mvo_traj))


        # dynosam_traj, gt_traj = evo_sync.associate_trajectories(
        #     copy.deepcopy(dynosam_traj),
        #     copy.deepcopy(gt_traj))

        first_dynosam_pose = dynosam_traj.poses_se3[0]
        first_gt_pose = gt_traj.poses_se3[0]
        first_mvo_pose = mvo_traj.poses_se3[0]

        # origin_diff = np.linalg.inv(first_dynosam_pose) @ first_mvo_pose
        origin_diff = np.linalg.inv(first_dynosam_pose) @ first_mvo_pose
        # realign poses to starting dynosam pose
        # realigned_mvo_poses = [first_gt_pose]
        # for pose_k_1, pose_k in zip(mvo_traj.poses_se3[:-1], mvo_traj.poses_se3[1:]):
        #     motion = pose_k @ (np.linalg.inv(pose_k_1))
        #     # motion_in_dynosam = first_gt_pose @ motion @ np.linalg.inv(first_gt_pose)
        #     motion_in_dynosam = np.lifirst_gt_pose @ motion @ np.linalg.inv(first_gt_pose)

        #     new_pose = motion_in_dynosam @ realigned_mvo_poses[-1]
        #     realigned_mvo_poses.append(new_pose)

        # realigned_mvo_poses = [first_mvo_pose]
        # for gt_pose_k_1, pose_k_1, pose_k in zip(gt_traj.poses_se3[:-1], mvo_traj.poses_se3[:-1], mvo_traj.poses_se3[1:]):
        #     motion = pose_k @ (np.linalg.inv(pose_k_1))
        #     # pose_diff = np.linalg.inv(gt_pose_k_1) @ pose_k_1
        #     # motion_in_dynosam = first_gt_pose @ motion @ np.linalg.inv(first_gt_pose)
        #     # motion_in_dynosam = np.lifirst_gt_pose @ motion @ np.linalg.inv(first_gt_pose)
        #     # motion_in_dynosam =  @ motion @ np.linalg.inv(pose_diff)


        #     new_pose = motion @ realigned_mvo_poses[-1]
        #     realigned_mvo_poses.append(new_pose)


        # realign poses to starting dynosam pose
        # realigned_mvo_poses = []
        # for mvo_pose in mvo_traj.poses_se3:
        #     realigned_mvo_poses.append( first_dynosam_pose @ mvo_pose )
        # mvo_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(realigned_mvo_poses), timestamps=mvo_traj.timestamps)

        print(mvo_traj.poses_se3[0])
        print(dynosam_traj.poses_se3[0])

        print(mvo_traj)
        print(dynosam_traj)
        print(gt_traj)

        # return

        # mvo_traj, gt_traj_sync = evo_sync.associate_trajectories(
        #     copy.deepcopy(mvo_traj),
        #     copy.deepcopy(gt_traj))

        # dynosam_traj, gt_traj = evo_sync.associate_trajectories(dynosam_traj,gt_traj)


        # dynosam_motion_traj, gt_traj = evo_sync.associate_trajectories(dynosam_motion_traj,gt_traj)


        # calculate motion for mvo in world
        mvo_motions = []
        # recalc motions after alignment
        motion_gt = []
        #
        motions_dynosam = []
        # # calculate the motion in world
        # print(gt_traj.num_poses)
        # print(mvo_traj.num_poses)
        # return

        mvo_poses = mvo_traj.poses_se3
        for index, (gt_pose_k_1, gt_pose_k, pose_k_1, pose_k) in enumerate(zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], mvo_poses[:-1], mvo_poses[1:])):
            # pose_k_1 = invT(pose_k_1)
            # pose_k = invT(pose_k)

            # pose_k_1_in_gt = lie_algebra.se3_inverse(gt_pose_k_1) @ pose_k_1
            # pose_k_in_gt = lie_algebra.se3_inverse(gt_pose_k) @ pose_k

            #this world is the mvo world
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))
            # H_world_gt = gt_pose_k_1 @ H_world @ lie_algebra.se3_inverse(gt_pose_k_1)

            # if not lie_algebra.is_se3(H_world):
            #     H_world[:3, :3] = closest_rotation_matrix(H_world[:3, :3])

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            mvo_motions.append(H_world)

        dynosam_poses = dynosam_traj.poses_se3
        for index, (pose_k_1, pose_k) in enumerate(zip(dynosam_poses[:-1], dynosam_poses[1:])):
            assert lie_algebra.is_se3(pose_k_1)
            assert lie_algebra.is_se3(pose_k)
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))

            # if not lie_algebra.is_se3(H_world):
            #     H_world[:3, :3] = closest_rotation_matrix(H_world[:3, :3])

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            motions_dynosam.append(H_world)

        gt_poses = gt_traj.poses_se3
        for index, (pose_k_1, pose_k) in enumerate(zip(gt_poses[:-1], gt_poses[1:])):
            assert lie_algebra.is_se3(pose_k_1)
            assert lie_algebra.is_se3(pose_k)
            H_world = pose_k @ (lie_algebra.se3_inverse(pose_k_1))

            # if not lie_algebra.is_se3(H_world):
            #     H_world[:3, :3] = closest_rotation_matrix(H_world[:3, :3])

            assert is_se3(H_world), f"{H_world} at index {index}: {pose_k_1} {pose_k}"
            motion_gt.append(H_world)

        print(f"GT traj {gt_traj}")

        mvo_motion_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(mvo_motions), timestamps=mvo_traj.timestamps[1:])
        # gt_motion_traj = evo_traj.PoseTrajectory3D(poses_se3=motion_gt, timestamps=gt_traj.timestamps[1:])
        dynosam_motion_traj = evo_traj.PoseTrajectory3D(poses_se3=np.array(motions_dynosam), timestamps=dynosam_traj.timestamps[1:])

        mvo_traj, _ = evo_sync.associate_trajectories(
            copy.deepcopy(mvo_traj),
            copy.deepcopy(gt_traj))

        dynosam_traj, _ = evo_sync.associate_trajectories(dynosam_traj,gt_traj)


        # mvo_traj, gt_traj = evo_sync.associate_trajectories(
        #     copy.deepcopy(mvo_traj),
        #     copy.deepcopy(gt_traj))


        print(f"Dynosam motion traj before {dynosam_motion_traj}")
        print(f"Dynosam motion traj times {dynosam_motion_traj.timestamps[0]} {dynosam_motion_traj.timestamps[-1]}")
        print(f"Mvo motion traj before {mvo_motion_traj}")
        print(f"Mvo motion traj times {mvo_motion_traj.timestamps[0]} {mvo_motion_traj.timestamps[-1]}")
        print(f"GT before {gt_traj}")
        dynosam_motion_traj, _ = evo_sync.associate_trajectories(dynosam_motion_traj,gt_traj)
        mvo_motion_traj, _ = evo_sync.associate_trajectories(mvo_motion_traj,gt_traj)
        print(f"Dynosam motion traj after {dynosam_motion_traj}")
        print(f"Dynosam motion traj times {dynosam_motion_traj.timestamps[0]} {dynosam_motion_traj.timestamps[-1]}")
        print(f"Mvo motion traj after {mvo_motion_traj}")
        print(f"Mvo motion traj times {mvo_motion_traj.timestamps[0]} {mvo_motion_traj.timestamps[-1]}")

        print("Trajectories")
        # print(mvo_traj)
        print(gt_traj)
        # print(dynosam_traj)
        print(mvo_motion_traj)
        print(dynosam_motion_traj)

        # print(motions)
        mvo_errors[dyno_label] = { "rot": [], "translation": [] }
        dynosam_errors[dyno_label] = { "rot": [], "translation": [] }


        # eval.tools.plot_object_trajectories(
        #     dynosam_fig, {"MVO camera" : mvo_camera_traj_aligned, "Dyno camera" : dynosam_camera_pose_traj},
        #     plot_mode = evo_plot.PlotMode.xyz,
        #     plot_axis_est=True, plot_axis_ref=True,
        #     downscale=0.05,
        #     # axis_marker_scale=1.0
        # )

        # for mvo_motion, gt_motion in zip(mvo_motion_traj.poses_se3, gt_motion_traj.poses_se3):
        #     error = lie_algebra.se3_inverse(gt_motion) @ mvo_motion
        #     assert lie_algebra.is_se3(error)

        #     rot_error = abs(so3_log_angle(error[:3, :3], True))
        #     t_error = np.linalg.norm(error[:3, 3])

        #     errors[dyno_label]["rot"].append(rot_error)
        #     errors[dyno_label]["translation"].append(t_error)

        # # rpe for mvo
        # for object_pose_k_1_est, object_pose_k_est, object_pose_k_1_ref, object_pose_k_ref in zip(mvo_motion_traj.poses_se3[:-1], mvo_motion_traj.poses_se3[1:], gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:]):
        #     ref = lie_algebra.se3_inverse(lie_algebra.se3_inverse(object_pose_k_1_ref) @ object_pose_k_ref)
        #     assert lie_algebra.is_se3(ref)

        #     est = lie_algebra.se3_inverse(object_pose_k_1_est) @ object_pose_k_est
        #     assert lie_algebra.is_se3(ref)

        #     rpe = ref @ est

        #     rot_error = abs(so3_log_angle(rpe[:3, :3], True))
        #     t_error = np.linalg.norm(rpe[:3, 3])

        #     mvo_errors[dyno_label]["rot"].append(rot_error)
        #     mvo_errors[dyno_label]["translation"].append(t_error)

        #  # rpe for dynosam
        # for object_pose_k_1_est, object_pose_k_est, object_pose_k_1_ref, object_pose_k_ref in zip(dynosam_traj.poses_se3[:-1], dynosam_traj.poses_se3[1:], gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:]):
        #     ref = lie_algebra.se3_inverse(lie_algebra.se3_inverse(object_pose_k_1_ref) @ object_pose_k_ref)
        #     assert lie_algebra.is_se3(ref)

        #     est = lie_algebra.se3_inverse(object_pose_k_1_est) @ object_pose_k_est
        #     assert lie_algebra.is_se3(ref)

        #     rpe = ref @ est

        #     rot_error = abs(so3_log_angle(rpe[:3, :3], True))
        #     t_error = np.linalg.norm(rpe[:3, 3])

        #     dynosam_errors[dyno_label]["rot"].append(rot_error)
        #     dynosam_errors[dyno_label]["translation"].append(t_error)

        # rme for mvo
        for object_pose_k_1, object_pose_k, object_motion_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], mvo_motion_traj.poses_se3[1:]):
            error_in_L = lie_algebra.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1
            error_in_L = lie_algebra.se3_inverse(error_in_L)

            # print(f"MVo error r {(so3_log(error_in_L[:3, :3], return_skew=False))}")

            rot_error = abs(so3_log_angle(error_in_L[:3, :3], True))
            t_error = np.linalg.norm(error_in_L[:3, 3])
            # print(f"MVO error {rot_error}")

            mvo_errors[dyno_label]["rot"].append(rot_error)
            mvo_errors[dyno_label]["translation"].append(t_error)

        # rme for dynosam
        for object_pose_k_1, object_pose_k, object_motion_k in zip(gt_traj.poses_se3[:-1], gt_traj.poses_se3[1:], dynosam_motion_traj.poses_se3[1:]):
            error_in_L = lie_algebra.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1
            error_in_L = lie_algebra.se3_inverse(error_in_L)


            # print(f"Dyno error r {np.linalg.norm(so3_log(error_in_L[:3, :3], return_skew=False))}")

            rot_error = abs(so3_log_angle(error_in_L[:3, :3], True))
            t_error = np.linalg.norm(error_in_L[:3, 3])

            dynosam_errors[dyno_label]["rot"].append(rot_error)
            dynosam_errors[dyno_label]["translation"].append(t_error)



        # all_mvo_traj[dyno_label] = propogated_mvo_traj
        all_mvo_traj[dyno_label] = mvo_traj
        all_gt_traj[dyno_label] = gt_traj
        all_dyno_traj["Dynosam " + str(dyno_label)] = dynosam_traj

    import math
    for object_id, error in mvo_errors.items():
        rot_error = np.array(error["rot"])
        t_error = np.array(error["translation"])

        # print(rot_error)
        # rmse_rot =np.mean(rot_error)
        # rmse_t = np.mean(t_error)
        rmse_rot = math.sqrt(np.mean(np.power(rot_error, 2)))
        rmse_t = math.sqrt(np.mean(np.power(t_error, 2)))

        print(f"MVO Errors for object id {object_id} -> rot {rmse_rot}, trans {rmse_t}")

    for object_id, error in dynosam_errors.items():
        rot_error = np.array(error["rot"])
        t_error = np.array(error["translation"])

        # rmse_rot =np.mean(rot_error)
        # rmse_t = np.mean(t_error)
        rmse_rot = math.sqrt(np.mean(np.power(rot_error, 2)))
        rmse_t = math.sqrt(np.mean(np.power(t_error, 2)))

        print(f"Dynosam Errors for object id {object_id} -> rot {rmse_rot}, trans {rmse_t}")

    eval.tools.plot_object_trajectories(
        evo_figure, all_mvo_traj,
        all_gt_traj,
        # None,
        plot_mode = evo_plot.PlotMode.xyz,
        plot_axis_est=False, plot_axis_ref=False,
        downscale=0.5,
        # axis_marker_scale=1.0
    )

    # eval.tools.plot_object_trajectories(
    #     evo_figure, {"mvo camera" : mvo_camera_traj, "dynosam camera": dynosam_camera_pose_traj},
    #     plot_mode = evo_plot.PlotMode.xyz,
    #     plot_axis_est=True, plot_axis_ref=True,
    #     downscale=0.005,
    #     # axis_marker_scale=1.0
    # )

    eval.tools.plot_object_trajectories(
        evo_figure, all_dyno_traj, #all_gt_traj,
        plot_mode = evo_plot.PlotMode.xyz,
        plot_axis_est=False, plot_axis_ref=True,
        downscale=0.5
        # axis_marker_scale=1.0
    )


    # eval.tools.plot_object_trajectories(
    #     evo_figure, all_gt_traj, plot_mode = evo_plot.PlotMode.xyz
    # )


    # mvo_motion_eval._object_motions_traj = all_mvo_motion_traj
    # mvo_motion_eval._object_motions_traj_ref = all_gt_motion_traj
    # # mvo_motion_eval._object_poses_traj = all_mvo_traj
    # # mvo_motion_eval._object_poses_traj_ref = all_gt_traj

    # mvo_results_dict = {}
    # mvo_motion_eval.process(plot_collection, mvo_results_dict)

    # print(mvo_results_dict)

    plot_collection.show()
    # table_formatter = formatting_utils.LatexTableFormatter()
    # table_formatter.add_results("MVO analysis", mvo_results_dict)
    # print(f"Writing output results to file {output_results_file}")
    # table_formatter.save_pdf(output_results_file)








if __name__ == '__main__':
    mvo_to_dyno_labels_swinging_dynamic = {
        1 : 1,
        2:  2,
        3 : 3,
        4 : 4
    }
    # process_mvo_data(
    #     "/root/data/mvo_data_scripts_IJRR/swinging_dynamic_wnoa.mat",
    #     # "/root/results/DynoSAM/test_omd_long",
    #     "/root/results/DynoSAM/test_omd",
    #     4,
    #     mvo_to_dyno_labels_swinging_dynamic,
    #     "/root/results/Dynosam_tro2024/mvo_analysis_swinging_dynamic_wnoa")

    # for this sequence object id 0 is the camera
    mvo_to_dyno_labels_kitti = {
        2 : 1,
        3:  2,
    }
    process_mvo_data("/root/data/mvo_data_scripts_IJRR/kitti_0005_wnoa.mat", "/root/results/DynoSAM/test_kitti/", 0, mvo_to_dyno_labels_kitti, "/root/results/Dynosam_tro2024/mvo_analysis_kitti_0000")
