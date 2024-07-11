import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from evo.core.lie_algebra import se3
from evo.core.trajectory import PosePath3D

import evo.tools.plot as evo_plot
import evo.core.trajectory as evo_trajectory


from typing import List
import typing

def so3_from_euler(euler_angles: np.ndarray, order:str = "xyz", degrees: bool = False) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_euler(order, euler_angles, degrees=degrees).as_matrix()

def camera_coordinate_to_world() -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return se3(
        R.from_euler("zyx", np.array([0, 0, -90]), degrees=True).as_matrix(),
        np.array([0.0, 0.0, 0.0]))


def transform_camera_trajectory_to_world(traj):
    from copy import deepcopy
    transform = camera_coordinate_to_world()
    traj_copy = deepcopy(traj)
    traj_copy.transform(transform)
    return traj_copy

def common_entries(*dcts):
    """
    Allows zip-like iteration over dictionaries with common keys
    e.g.
    for key, value1, value2 in common_entries(dict, dict2):

    Yields:
        _type_: _description_
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

# values between 0 and 1
def hsv_to_rgb(h:float, s:float, v:float, a:float = 1) -> tuple:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)
    # if s:
    #     if h == 1.0: h = 0.0
    #     i = int(h*6.0); f = h*6.0 - i

    #     w = v * (1.0 - s)
    #     q = v * (1.0 - s * f)
    #     t = v * (1.0 - s * (1.0 - f))

    #     if i==0: return (v, t, w, a)
    #     if i==1: return (q, v, w, a)
    #     if i==2: return (w, v, t, a)
    #     if i==3: return (w, q, v, a)
    #     if i==4: return (t, w, v, a)
    #     if i==5: return (v, w, q, a)
    # else: return (v, v, v, a)

def generate_unique_colour(id: int, saturation: float = 0.5, value: float = 0.95):
    import math
    phi = (1 + math.sqrt(5))/2.0
    n = id * phi - math.floor(id * phi)
    # hue = math.floor(n * 256)

    colour = hsv_to_rgb(n, saturation, value)
    # put values between 0 and 255
    result = list(map(lambda x: math.floor(x * 256), colour))
    #return r,g,b component of result
    return result[:3]


def rgb_to_hex_string(*args) -> str:
    if len(args) == 1 and isinstance(args[0], list) and len(args[0]) == 3:
        # If a single list with 3 elements is provided
        r, g, b = args[0]
    elif len(args) == 3:
        # If three separate arguments are provided
        r,g,b = args
    else:
        raise ValueError("Invalid input. Please provide either a list of 3 elements or 3 separate arguments.")
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


# https://github.com/qcr/quadricslam/blob/master/src/quadricslam/visualisation.py
def set_axes_equal(ax):
    # Matplotlib is really ordinary for 3D plots... here's a hack taken from
    # here to get 'square' in 3D:
    #   https://stackoverflow.com/a/31364297/1386784
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class TrajectoryHelper:

    def __init__(self):

        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        self.min_x = np.inf
        self.min_y = np.inf
        self.min_z = np.inf

    def append(self, trajectories: typing.Union[
            evo_trajectory.PosePath3D, typing.Sequence[evo_trajectory.PosePath3D],
            typing.Dict[str, evo_trajectory.PosePath3D]]):

        def calc_min_max(traj: evo_trajectory.PosePath3D):
            positions_xyz = traj.positions_xyz

            self.max_x = max(self.max_x, np.max(positions_xyz, axis=0)[0])
            self.max_y = max(self.max_y, np.max(positions_xyz, axis=0)[1])
            self.max_z = max(self.max_z, np.max(positions_xyz, axis=0)[2])
            self.min_x = min(self.min_x, np.min(positions_xyz, axis=0)[0])
            self.min_y = min(self.min_y, np.min(positions_xyz, axis=0)[1])
            self.min_z = min(self.min_z, np.min(positions_xyz, axis=0)[2])


        if isinstance(trajectories, evo_trajectory.PosePath3D):
            calc_min_max(trajectories)
        elif isinstance(trajectories, dict):
            for _, t in trajectories.items():
                calc_min_max(t)
        else:
            for t in trajectories:
                calc_min_max(t)

    def set_ax_limits(self, ax: plt.Axes):
        ax.set_zlim3d([self.min_x, self.max_z])
        ax.set_xlim3d([self.min_x, self.max_x])
        ax.set_ylim3d([self.min_y, self.max_y])
        set_axes_equal(ax)



# modification of evo.trajectories
def plot_object_trajectories(
        fig: Figure,
        obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
        obj_trajectories_ref: typing.Dict[str, evo_trajectory.PosePath3D],
        plot_mode=evo_plot.PlotMode.xy,
        title: str = "",
        subplot_arg: int = 111,
        plot_start_end_markers: bool = False,
        length_unit: evo_plot.Unit = evo_plot.Unit.meters) -> None:

    if len(obj_trajectories) != len(obj_trajectories_ref):
        raise evo_plot.PlotException(f"Expected trajectories and ref trajectories to have the same length {len(obj_trajectories)} != {len(obj_trajectories_ref)}")

    ax = evo_plot.prepare_axis(fig, plot_mode, subplot_arg, length_unit)
    if title:
        ax.set_title(title)

    cmap_colors = None
    import collections
    import matplotlib.cm as cm
    import itertools
    import seaborn as sns

    if evo_plot.SETTINGS.plot_multi_cmap.lower() != "none" and isinstance(
            obj_trajectories, collections.abc.Iterable):
        cmap = getattr(cm, evo_plot.SETTINGS.plot_multi_cmap)
        cmap_colors = iter(cmap(np.linspace(0, 1, len(obj_trajectories))))

    color_palette = itertools.cycle(sns.color_palette())

    # helper function
    def draw_impl(t, style: str = '-',name="", alpha: float = 1.0, shift_color: bool = False):
        if cmap_colors is None:
            color = next(color_palette)
        else:
            color = next(cmap_colors)

        if shift_color:
            import colorsys
            hsv_color = colorsys.rgb_to_hsv(color[0], color[1], color[1])
            # shift slightly
            h = hsv_color[0] + 0.1
            color = colorsys.hsv_to_rgb(h, hsv_color[1], hsv_color[2])

        if evo_plot.SETTINGS.plot_usetex:
            name = name.replace("_", "\\_")
        evo_plot.traj(ax, plot_mode, t, style, color, name,
             plot_start_end_markers=plot_start_end_markers, alpha=alpha)

    def draw(traj: typing.Any, **kwargs):
        if isinstance(traj, evo_trajectory.PosePath3D):
            draw_impl(traj, **kwargs)
        elif isinstance(traj, dict):
            for name, t in traj.items():
                draw_impl(t,name=name,**kwargs)
        else:
            for t in traj:
                draw_impl(t, **kwargs)

    draw(obj_trajectories, style='-')
    # reset colours
    color_palette = itertools.cycle(sns.color_palette())
    draw(obj_trajectories_ref, style='+', alpha=0.8, shift_color=False)





#from v1.0 of evo as this function seems to have become depcirated!!
def align_trajectory(traj, traj_ref, correct_scale=False, correct_only_scale=False, n=-1):
    """
    align a trajectory to a reference using Umeyama alignment
    :param traj: the trajectory to align
    :param traj_ref: reference trajectory
    :param correct_scale: set to True to adjust also the scale
    :param correct_only_scale: set to True to correct the scale, but not the pose
    :param n: the number of poses to use, counted from the start (default: all)
    :return: the aligned trajectory
    """
    import copy
    import evo.core.geometry as evo_geometry
    import evo.core.lie_algebra as lie
    import evo.core.transformations as evo_transform
    from evo.tools.user import logger

    traj_aligned = copy.deepcopy(traj)  # otherwise np arrays will be references and mess up stuff
    with_scale = correct_scale or correct_only_scale
    try:
        if n == -1:
            r_a, t_a, s = evo_geometry.umeyama_alignment(traj_aligned.positions_xyz.T,
                                                    traj_ref.positions_xyz.T, with_scale)
        else:
            r_a, t_a, s = evo_geometry.umeyama_alignment(traj_aligned.positions_xyz[:n, :].T,
                                                    traj_ref.positions_xyz[:n, :].T, with_scale)
    except evo_geometry.GeometryException as e:
        logger.warning(f"Could not align trajectories {str(e)}")
        # just align with zero
        traj_aligned.align_origin(traj_ref)

        return traj_aligned

    if not correct_only_scale:
        logger.debug("Rotation of alignment:\n{}"
                     "\nTranslation of alignment:\n{}".format(r_a, t_a))
    logger.debug("Scale correction: {}".format(s))

    if correct_only_scale:
        traj_aligned.scale(s)
    elif correct_scale:
        traj_aligned.scale(s)
        traj_aligned.transform(lie.se3(r_a, t_a))
    else:
        traj_aligned.transform(lie.se3(r_a, t_a))

    return traj_aligned
