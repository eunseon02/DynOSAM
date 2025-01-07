from evo.core.metrics import APE, PoseRelation, PathPair, MetricsException
import evo.core.lie_algebra as lie
import evo.core.trajectory as traj
import evo.core.sync as sync

import numpy as np
import copy
import typing

from enum import Enum, unique

TrajPair = typing.Tuple[traj.PoseTrajectory3D, traj.PoseTrajectory3D]


def reduce_to_largest_strictly_ascending_range(traj: traj.PoseTrajectory3D):
    timestamps = list(traj.timestamps)
    if not timestamps:
        return None, None  # Handle empty list case

    max_start, max_end = 0, 0
    start, end = 0, 0

    for i in range(1, len(timestamps)):
        if timestamps[i] > timestamps[i - 1]:
            end = i
        else:
            # Check if the current range is the largest so far
            if (end - start) > (max_end - max_start):
                max_start, max_end = start, end
            # Reset start and end for the next potential range
            start, end = i, i

    # Final check in case the longest range is at the end of the list
    if (end - start) > (max_end - max_start):
        max_start, max_end = start, end

    reduced_traj = copy.deepcopy(traj)
    reduced_traj.reduce_to_time_range(max_start, max_end)
    return reduced_traj


@unique
class MetricType(Enum):
    ame = "ame"
    rme = "rme"
    rpe = "rpe"


class AME(APE):
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part):
        super().__init__(pose_relation)


class RME(APE):
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part):
        super().__init__(pose_relation)
        # timestamp data after processing (this will be k to N when the size of the timestamps matchs the size of the error array)
        self.timestamps = np.array([])

    def process_data(self, data: TrajPair) -> None:
        """
        Calcualtes RME  on a motion trajectory and associated ground truth pose trajectory
        :param data: tuple (traj_pose_ref, traj_motion_est) with:
        traj_pose_ref: reference evo.trajectory.PosePath or derived and is the Pose associated with the trajectory
        traj_motion_est: estimated evo.trajectory.PosePath or derived and is the Motion associated with the trajectory
        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_pose_ref, traj_motion_est)")
        # traj_pose_ref, traj_motion_est = RME.sync_object_motion_and_pose(data)
        traj_pose_ref, traj_motion_est = data

        # if traj_pose_ref.num_poses - 1 != traj_motion_est.num_poses:
        #     raise MetricsException(
        #         f"Reference pose data must have one more entry than estimated motion data: pose {traj_pose_ref.num_poses} vs. motion {traj_motion_est.num_poses}")

        object_poses_ref = traj_pose_ref.poses_se3
        object_motions_ref = traj_motion_est.poses_se3

        # we have designed the sync_object_motion_and_pose to return the pose data in the from k-1 to N and the motion data k to N
        object_motion_L = []
        object_motion_L_gt = []
        for object_pose_k_1, object_pose_k, object_motion_k in zip(object_poses_ref[:-1], object_poses_ref[1:], object_motions_ref[1:]):
            object_motion_L.append(lie.se3_inverse(object_pose_k) @ object_motion_k @ object_pose_k_1)
            object_motion_L_gt.append(lie.se3())

        # update internal timestamps
        # -1 on timestamps so that the size matches the size of the motion error (ie. k...N)
        self.timestamps = np.array(traj_motion_est.timestamps[:-1])

        # print(f"len object motion L {len(object_motion_L)} len traj motion timestamps {len(traj_motion_est.timestamps)}")

        object_traj_in_L = traj.PoseTrajectory3D(poses_se3=np.array(object_motion_L), timestamps=self.timestamps)
        object_traj_in_L_ref = traj.PoseTrajectory3D(poses_se3=np.array(object_motion_L_gt), timestamps=self.timestamps)

        processed_data = (object_traj_in_L_ref, object_traj_in_L)

        super().process_data(processed_data)

        assert self.error.shape == self.timestamps.shape, ( self.error.shape, self.timestamps.shape)

    @staticmethod
    def sync_object_motion_and_pose(data: TrajPair, max_diff = 0.01):
        # sync so that we get object motions from i to N
        # and object poses from i-1 to N (inclusive)
        traj_pose_ref, traj_motion_est = data

        # takes copies
        traj_pose_ref = reduce_to_largest_strictly_ascending_range(traj_pose_ref)
        traj_motion_est = reduce_to_largest_strictly_ascending_range(traj_motion_est)

        traj_pose_ref_sync, traj_motion_est_sync = sync.associate_trajectories(traj_pose_ref, traj_motion_est,max_diff=max_diff)

        # print("h1 ",traj_pose_ref_sync.timestamps)
        # print("h2", traj_motion_est_sync.timestamps)

        # check if the first timestamp in the pose is less than or equal to the timestamp in the motion
        if traj_pose_ref_sync.timestamps[0] >= traj_motion_est_sync.timestamps[0]:
            found_index = None
            # search through the motion traj to find the first one that is greater than the first pose of this one
            for i in range(1, len(traj_motion_est_sync.timestamps)):  # Start from the second element
                if traj_motion_est_sync.timestamps[i] >= traj_pose_ref_sync.timestamps[0]:
                    found_index = i
                    break

            if found_index is None:
                raise ValueError("No pose in trajectory traj_pose_ref exists before the start of trajectory traj_motion_est.")

            print(f" FI {found_index} pose t { traj_pose_ref_sync.timestamps[0]} motion t {traj_motion_est_sync.timestamps[found_index]}")
            # seems to be a bug in the code here....
            # Grab the last available pose before B's start timestamp
            extra_pose_index = found_index-1
            print(extra_pose_index)

            extra_pose_timestamp = traj_pose_ref_sync.timestamps[extra_pose_index]
            print(f"Extra pose timestamp {extra_pose_timestamp}")
            extra_pose_pose = traj_motion_est_sync.poses_se3[extra_pose_index]

            traj_motion_est_sync.reduce_to_time_range(traj_motion_est_sync.timestamps[found_index], traj_motion_est_sync.timestamps[-1])
            traj_pose_ref_sync.reduce_to_time_range(traj_pose_ref_sync.timestamps[found_index], traj_pose_ref_sync.timestamps[-1])


        else:
            # Find the minimum timestamp of trajectory B for normal adjustment
            min_timestamp =  traj_motion_est.timestamps[0]
            filtered_indices_traj_pose = np.where(traj_pose_ref.timestamps < min_timestamp)[0]

            if len(filtered_indices_traj_pose) == 0:
                raise ValueError("No pose in trajectory traj_pose_ref exists before the start of trajectory traj_motion_est.")

            # Grab the last pose from the filtered part of A (which will be the extra one)
            extra_pose_index = filtered_indices_traj_pose[-1]


            # Grab the last pose from the filtered part of traj pose (which will be the extra one)
            extra_pose_timestamp = traj_pose_ref_sync.timestamps[extra_pose_index]
            extra_pose_pose = traj_pose_ref_sync.poses_se3[extra_pose_index]


        # Filter A and B to the synchronized region (with one extra in A)
        synced_timestamps_traj_pose = np.concatenate(([extra_pose_timestamp], traj_pose_ref_sync.timestamps))
        synced_poses_traj_pose = np.concatenate(([extra_pose_pose], traj_pose_ref_sync.poses_se3))

        traj_pose_ref_sync = traj.PoseTrajectory3D(poses_se3=synced_poses_traj_pose, timestamps=synced_timestamps_traj_pose)

        valid, result = traj_pose_ref_sync.check()
        assert valid, (result, traj_pose_ref_sync.timestamps)

        print(traj_pose_ref_sync.timestamps)
        print(traj_motion_est_sync.timestamps)

        return (traj_pose_ref_sync, traj_motion_est_sync)
