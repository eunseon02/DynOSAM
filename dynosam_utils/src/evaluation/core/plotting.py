from __future__ import annotations


import evo.core
import evo.core.geometry
import evo.core.metrics
import evo.core.transformations
import numpy as np
from matplotlib.figure import Figure
from evo.core.lie_algebra import se3
from evo.core.trajectory import PosePath3D
from matplotlib.axes import Axes
import gtsam

from evaluation.formatting_utils import *


# apparently this ordereed needed
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import evo.tools.plot as evo_plot
import evo.core.trajectory as evo_trajectory
import evo.core.lie_algebra as evo_lie_algebra
from evo.core.metrics import PE

import evo

from typing import Tuple, Optional, List, Any, Iterable

from copy import deepcopy
import typing
import math

def plot_metric(metric: PE, plot_title="", figsize=(8,8), fig = None, x_axis=None):
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

    from evo.tools import plot
    plot.error_array(ax, metric.error,
                        x_array=x_axis,
                        statistics=stats,
                        title=plot_title,
                        xlabel="Frame Index [-]",
                        ylabel=plot_title + " " + metric.unit.value)

    return fig


def plot_traj(ax: Axes, plot_mode: evo_plot.PlotMode, traj: evo_trajectory.PosePath3D,
         style: str = '-', color='black', label: str = "", alpha: float = 1.0,
         plot_start_end_markers: bool = False, **kwargs) -> None:
    """
    plot a path/trajectory based on xyz coordinates into an axis.
    Modification of evo.plot_traj, allowing kwargs to be parsed to ax.plot.

    :param ax: the matplotlib axis
    :param plot_mode: PlotMode
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    :param plot_start_end_markers: Mark the start and end of a trajectory
                                   with a symbol.
    :param **kwargs: kwargs parsed to ax.plot
    """
    from evo.tools.settings import SETTINGS
    x_idx, y_idx, z_idx = evo_plot.plot_mode_to_idx(plot_mode)

    x = traj.positions_xyz[:, x_idx]
    y = traj.positions_xyz[:, y_idx]
    if plot_mode == evo_plot.PlotMode.xyz:
        z = traj.positions_xyz[:, z_idx]
        ax.plot(x, y, z, style, color=color, label=label, alpha=alpha, **kwargs)
    else:
        ax.plot(x, y, style, color=color, label=label, alpha=alpha, **kwargs)
    if SETTINGS.plot_xyz_realistic:
        evo_plot.set_aspect_equal(ax)
    if label and SETTINGS.plot_show_legend:
        ax.legend(frameon=True)
    if plot_start_end_markers:
        evo_plot.add_start_end_markers(ax, plot_mode, traj, start_color=color,
                              end_color=color, alpha=alpha)



class ObjectTrajectoryPlotter(object):

    def __init__(self) -> None:
        # internal colours to use. Some complex logic here but is updated everytime plot is used
        # will eventually be a iterable of colours which can be in any valid plt colour form
        self._colours_itr: Iterable[Any] = None
        # the original colour list, over which the _colours_itr operates. Used to reset the colours
        self._colours = None

    def plot(self,
             fig: Figure,
            obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
            obj_trajectories_ref: Optional[typing.Dict[str, evo_trajectory.PosePath3D]] = None,
            plot_mode=evo_plot.PlotMode.xy,
            title: str = "",
            subplot_arg: int = 111,
            plot_start_end_markers: bool = False,
            length_unit: evo_plot.Unit = evo_plot.Unit.meters,
            **kwargs) -> Axes:

        if obj_trajectories_ref is not None and len(obj_trajectories) != len(obj_trajectories_ref):
            raise evo_plot.PlotException(f"Expected trajectories and ref trajectories to have the same length {len(obj_trajectories)} != {len(obj_trajectories_ref)}")


        # get all top level params
        plot_axis_est = kwargs.pop("plot_axis_est", False)
        plot_axis_ref = kwargs.pop("plot_axis_ref", False)
        # downscale as a percentage - how many poses to use when plotting anything over the top of the tajectories
        downscale = kwargs.pop("downscale", 0.1)
        axis_marker_scale = kwargs.pop("axis_marker_scale", 0.1)

        ax = None
        if len(fig.axes) == 0:
            ax = evo_plot.prepare_axis(fig, plot_mode, subplot_arg, length_unit)
        else:
            # use existing axis TODO: check 3d?
            ax = fig.gca()
        if title:
            ax.set_title(title)

        self._set_colours(obj_trajectories, kwargs.get("colours"))

        self._draw_trajectory(
            ax,
            obj_trajectories,
            plot_mode,
            plot_start_end_markers,
            style=kwargs.pop("est_style", '-'),
            shift_color=kwargs.pop("shift_est_colour", None),
            name_prefix=kwargs.pop("est_name_prefix", ""),
            **kwargs)
        if plot_axis_est:
            ObjectTrajectoryPlotter.plot_coordinate_axis(
                ax,
                obj_trajectories,
                plot_mode,
                downscale,
                axis_marker_scale
            )

        self._reset_colours()

        if obj_trajectories_ref is not None:
            self._draw_trajectory(
                ax,
                obj_trajectories_ref,
                plot_mode,
                plot_start_end_markers,
                style=kwargs.pop("ref_style", '--'),
                alpha=0.8,
                shift_color=kwargs.pop("shift_ref_colour", None),
                name_prefix=kwargs.pop("ref_name_prefix", ""),
                **kwargs)

            if plot_axis_ref:
                ObjectTrajectoryPlotter.plot_coordinate_axis(
                    ax,
                    obj_trajectories_ref,
                    plot_mode,
                    downscale,
                    axis_marker_scale
                )
        return ax


    @staticmethod
    def reduce_trajectory(downscale_percentage: float, traj:evo_trajectory.PosePath3D):
        assert downscale_percentage >= 0.0 and downscale_percentage <= 1.0

        reduced_trajectory = deepcopy(traj)

        reduced_pose_num = downscale_percentage * reduced_trajectory.num_poses
        # round UP to the nearest int (so we at least get 1 pose
        reduced_pose_num = int(math.ceil(reduced_pose_num))
        if reduced_pose_num >= 2:
            reduced_trajectory.downsample(reduced_pose_num)
        return reduced_trajectory

    @staticmethod
    def plot_coordinate_axis(
            ax: Axes,
            trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
            plot_mode: evo_plot.PlotMode,
            downscale_percentage:float,
            axis_marker_scale:float):
        """
        Plot coordinate axes on a map of object trajectories.
        This is down by downscaling the numner of poses over which the coordiante axes will
        be drawn for visability.

        Args:
            trajectories (typing.Dict[str, evo_trajectory.PosePath3D]): _description_
        """
        reduced_trajectories = deepcopy(trajectories)
        for _, obj_traj in reduced_trajectories.items():
            obj_traj = ObjectTrajectoryPlotter.reduce_trajectory(downscale_percentage, obj_traj)
            evo_plot.draw_coordinate_axes(
                ax, obj_traj, plot_mode,marker_scale=axis_marker_scale)

    def _draw_trajectory(self,
                         ax: Axes,
                         traj: typing.Any,
                         plot_mode: evo_plot.PlotMode,
                         plot_start_end_markers: bool,
                         **kwargs):
        if isinstance(traj, evo_trajectory.PosePath3D):
            self._draw_impl(ax, traj, plot_mode, plot_start_end_markers, **kwargs)
        elif isinstance(traj, dict):
            for name, t in traj.items():
                if "name_prefix" in kwargs:
                    name = kwargs.get("name_prefix") + " " + str(name)

                kwargs.update({"name":name})
                self._draw_impl(ax,t, plot_mode, plot_start_end_markers,**kwargs)
        else:
            for t in traj:
                self._draw_impl(ax, t, plot_mode, plot_start_end_markers, **kwargs)

    def _draw_impl(self,
                   ax: Axes,
                   traj: typing.Any,
                   plot_mode: evo_plot.PlotMode,
                   plot_start_end_markers: bool,
                   **kwargs):
        color = self._get_next_colour()

        name = kwargs.get("name", "")
        style = kwargs.get("style", '-')
        alpha = kwargs.get("alpha", 1.0)
        shift_color = kwargs.get("shift_colour", False)
        z_order = kwargs.get("traj_zorder", 1)
        linewidth = kwargs.get("traj_linewidth",plt.rcParams["lines.linewidth"])


        if shift_color:
            shift_color = float(shift_color)
            import colorsys
            hsv_color = colorsys.rgb_to_hsv(color[0], color[1], color[1])
            # shift slightly
            h = hsv_color[0] + shift_color
            color = colorsys.hsv_to_rgb(h, hsv_color[1], hsv_color[2])

        if evo_plot.SETTINGS.plot_usetex:
            name = name.replace("_", "\\_")

        plot_traj(ax, plot_mode, traj, style, color, name,
             plot_start_end_markers=plot_start_end_markers,
             alpha=alpha,
             zorder=z_order,
             linewidth=linewidth)

    def _set_colours(self,
                     obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
                     provided_colours: Optional[List[Any]] = None):
        import collections
        import matplotlib.cm as cm

        # use provided colour map
        if provided_colours is not None:
            # should be list of colours, one for each obj traj
            if len(provided_colours) != len(obj_trajectories):
                raise evo_plot.PlotException(f"Expected trajectories and provided colours to have the same length {len(obj_trajectories)} != {len(provided_colours)}")
            #set internal colours
            self._colours = provided_colours

        # use cmap as described by evo plot settings
        elif evo_plot.SETTINGS.plot_multi_cmap.lower() != "none" and isinstance(
                obj_trajectories, collections.abc.Iterable):
            cmap = getattr(cm, evo_plot.SETTINGS.plot_multi_cmap)
            self._colours = cmap(np.linspace(0, 1, len(obj_trajectories)))
        else:
            import seaborn as sns
            self._colours= sns.color_palette()

        self._reset_colours()

    def _reset_colours(self):
        """
        Updates (resets) the colour iterator based on the current colours
        """
        import itertools
        assert self._colours is not None
        self._colours_itr = itertools.cycle(self._colours)


    def _get_next_colour(self) -> Any:
        return  next(self._colours_itr)


def draw_camera_frustum(ax: Axes,
                        transformation_matrix: np.ndarray,
                        far=0.6, fov=50,
                        aspect_ratio=0.9,
                        near=0.0,
                        frustrum_colour = get_nice_blue(),
                        convention="camera",
                        draw_planes: bool = False):
    if convention not in ["camera","world"]:
        raise Exception(f"Unknown convention {convention}. Must either be camera or world")

    # Field of view (fov) should be in degrees
    fov = np.radians(fov)  # Convert to radians

    # Calculate height and width of near and far planes
    h_near = 2 * np.tan(fov / 2) * near
    w_near = h_near * aspect_ratio
    h_far = 2 * np.tan(fov / 2) * far
    w_far = h_far * aspect_ratio

    # Frustum vertices in camera coordinates (default in camera frame)
    frustum_points = np.array([
        [0, 0, 0],  # Camera origin
        [-w_near / 2, -h_near / 2, -near],  # Near plane bottom-left
        [w_near / 2, -h_near / 2, -near],   # Near plane bottom-right
        [w_near / 2, h_near / 2, -near],    # Near plane top-right
        [-w_near / 2, h_near / 2, -near],   # Near plane top-left
        [-w_far / 2, -h_far / 2, -far],     # Far plane bottom-left
        [w_far / 2, -h_far / 2, -far],      # Far plane bottom-right
        [w_far / 2, h_far / 2, -far],       # Far plane top-right
        [-w_far / 2, h_far / 2, -far]       # Far plane top-left
    ])

    # Transform to the world convention if required
    if convention == 'world':
        # Rotate the points to match world convention (x forward, z up, y right)
        rotation_matrix = np.array([
            [0, 0, 1, 0],  # X-axis (forward) becomes Z-axis
            [0, 1, 0, 0],  # Y-axis remains Y-axis
            [-1, 0, 0, 0], # Z-axis (up) becomes -X-axis (to point along X in world)
            [0, 0, 0, 1]   # Homogeneous coordinate
        ])
        frustum_points_h = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
        frustum_points = (rotation_matrix @ frustum_points_h.T).T[:, :3]

    # Apply the transformation matrix to the frustum points
    frustum_points_h = np.hstack([frustum_points, np.ones((frustum_points.shape[0], 1))])
    transformed_points = (transformation_matrix @ frustum_points_h.T).T[:, :3]

    # Define the frustum edges (connecting lines between vertices)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Camera origin to near plane corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # Near plane edges
        (5, 6), (6, 7), (7, 8), (8, 5),  # Far plane edges
        (1, 5), (2, 6), (3, 7), (4, 8)   # Near plane to far plane edges
    ]

    # Plot the frustum edges
    for edge in edges:
        points = transformed_points[np.array(edge)]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], '-', color=frustrum_colour)

    # Optionally, draw the near and far planes as polygons
    if draw_planes:
        near_plane = Poly3DCollection([transformed_points[1:5]], color='cyan', alpha=0.3)
        far_plane = Poly3DCollection([transformed_points[5:]], color='magenta', alpha=0.3)
        ax.add_collection3d(near_plane)
        ax.add_collection3d(far_plane)



def draw_camera_frustrums(ax: Axes, trajectory: evo_trajectory.PosePath3D, downsample: float = 1.0, **kwargs):
    traj = ObjectTrajectoryPlotter.reduce_trajectory(downsample, trajectory)
    for poses in traj.poses_se3:
        draw_camera_frustum(ax, poses, **kwargs)
