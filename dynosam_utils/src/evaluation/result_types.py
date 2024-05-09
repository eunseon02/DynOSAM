from typing import Optional, List, Tuple, Dict, TypeAlias, Union
import numpy as np
from evo.core import trajectory, sync, metrics
from evo.tools import plot

from .tools import align_trajectory

import matplotlib.pyplot as plt

## {object_id: { frame_id: homogenous_matrix }}
ObjectPoseDict: TypeAlias = Dict[int, Dict[int, np.ndarray]]
## {object_id: trajectory.PosePath3D}
ObjectTrajDict: TypeAlias = Dict[int, trajectory.PosePath3D]



def sync_and_align_trajectories(traj_est: Union[trajectory.PosePath3D, trajectory.PoseTrajectory3D],
                                traj_ref: Union[trajectory.PosePath3D, trajectory.PoseTrajectory3D],
                                discard_n_end_poses=-1) -> Union[trajectory.PosePath3D, trajectory.PoseTrajectory3D]:
    import copy
    # We copy to distinguish another version that may be created
    traj_ref = copy.deepcopy(traj_ref)

    # assume synched and in order!
    if isinstance(traj_est, trajectory.PoseTrajectory3D) and isinstance(traj_ref, trajectory.PoseTrajectory3D):
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
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
