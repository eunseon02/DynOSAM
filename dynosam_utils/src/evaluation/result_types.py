from typing import Optional, List, Tuple, Dict, TypeAlias
import numpy as np
from evo.core import trajectory, sync, metrics
from evo.tools import plot

from .tools import align_trajectory

import matplotlib.pyplot as plt


## {object_id: { frame_id: [t_error: float, r_error: float] }}
MotionErrorDict: TypeAlias = Dict[int, Dict[int, List[float]]]
## {object_id: { frame_id: homogenous_matrix }}
ObjectPoseDict: TypeAlias = Dict[int, Dict[int, np.ndarray]]
## {object_id: trajectory.PosePath3D}
ObjectTrajDict: TypeAlias = Dict[int, trajectory.PosePath3D]

## {object_id: { frame_id: [t_abs_err: float, r_abs_err: float, t_rel_err: float, r_rel_err: float] }}
ObjectPoseErrorDict: TypeAlias = Dict[int, Dict[int, List[float]]]


def analyse_motion_error_dict(motion_error_dict: MotionErrorDict) -> dict:
    motion_error_result_yaml = {}
    for object_id, per_frame_dict in motion_error_dict.items():
        t_error_np = np.array([per_frame_dict[frame][0] for frame in per_frame_dict.keys() if per_frame_dict[frame][0] != -1])
        r_error_np = np.array([per_frame_dict[frame][1] for frame in per_frame_dict.keys() if per_frame_dict[frame][1] != -1])

        motion_error_result_yaml[object_id] = {
            "avg_t_err" : np.average(t_error_np),
            "avg_r_err" : np.average(r_error_np),
            "n_frame" : len(per_frame_dict)
        }
    return motion_error_result_yaml

def analyse_object_pose_errors(object_pose_errors: ObjectPoseErrorDict) -> dict:
    object_pose_error_result_yaml = {}
    for object_id, per_frame_dict in object_pose_errors.items():
        t_abs_error_np = np.array([per_frame_dict[frame][0] for frame in per_frame_dict.keys() if per_frame_dict[frame][0] != -1])
        r_abs_error_np = np.array([per_frame_dict[frame][1] for frame in per_frame_dict.keys() if per_frame_dict[frame][1] != -1])
        t_rel_error_np = np.array([per_frame_dict[frame][2] for frame in per_frame_dict.keys() if per_frame_dict[frame][2] != -1])
        r_rel_error_np = np.array([per_frame_dict[frame][3] for frame in per_frame_dict.keys() if per_frame_dict[frame][3] != -1])

        object_pose_error_result_yaml[object_id] = {
            "avg_t_abs_err" : np.average(t_abs_error_np),
            "avg_r_abs_err" : np.average(r_abs_error_np),
            "avg_t_rel_err" : np.average(t_rel_error_np),
            "avg_r_rel_err" : np.average(r_rel_error_np),
            "n_frame" : len(per_frame_dict)
        }
    return object_pose_error_result_yaml



def sync_and_align_trajectories(traj_est: trajectory.PosePath3D, traj_ref: trajectory.PosePath3D,
                                discard_n_end_poses=-1) -> trajectory.PosePath3D:
    import copy
    # We copy to distinguish another version that may be created
    traj_ref = copy.deepcopy(traj_ref)

    # assume synched and in order!
    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    traj_est = align_trajectory(traj_est, traj_ref, correct_scale = False,
                                                   n=discard_n_end_poses)
    return traj_est, traj_ref

def plot_metric(metric, plot_title="", figsize=(8,8), fig = None):
    """ Adds a metric plot to a plot collection.

        Args:
            plot_collection: a PlotCollection containing plots.
            metric: an evo.core.metric object with statistics and information.
            plot_title: a string representing the title of the plot.
            figsize: a 2-tuple representing the figure size.

        Returns:
            A plt figure.
    """
    if not fig:
        fig = plt.figure(figsize=figsize)
    stats = metric.get_all_statistics()

    ax = fig.gca()

    # remove from stats
    del stats["rmse"]
    del stats["sse"]

    plot.error_array(ax, metric.error, statistics=stats,
                        title=plot_title,
                        xlabel="Frame Index [-]",
                        ylabel=plot_title + " " + metric.unit.value)

    return fig
