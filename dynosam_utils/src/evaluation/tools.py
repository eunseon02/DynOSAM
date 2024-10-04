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

from .formatting_utils import *


# apparently this ordereed needed
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


import evo.tools.plot as evo_plot
import evo.core.trajectory as evo_trajectory
import evo.core.lie_algebra as evo_lie_algebra
import evo.core.sync as evo_sync

import evo

from typing import Tuple, Optional,Union

from copy import deepcopy
import typing
import math

def load_bson(path:str):
    import bson
    with open(path,'rb') as f:
        content = f.read()
        data = bson.decode_all(content)
        return data

def so3_from_euler(euler_angles: np.ndarray, order:str = "xyz", degrees: bool = False) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_euler(order, euler_angles, degrees=degrees).as_matrix()

def so3_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(quaternion, scalar_first=False).as_matrix()

def load_pose_from_row(row) -> Tuple[np.ndarray, np.ndarray]:

    """
    Loads a estimated and reference (ground truth() pose from a given row (from a logged csv file).
    Expects row to contain tx, ty, tz, qx, qy, qz, qw information for the  estimated pose, and prefixed
    with gt_* for the reference trajectory.

    Args:
        row (_type_): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: Estimated pose, reference (gt) pose
    """
    translation = np.array([
        float(row["tx"]),
        float(row["ty"]),
        float(row["tz"])
    ])
    rotation = so3_from_xyzw(np.array([
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"]),
        float(row["qw"])
    ]))

    ref_translation = np.array([
        float(row["gt_tx"]),
        float(row["gt_ty"]),
        float(row["gt_tz"])
    ])
    ref_rotation = so3_from_xyzw(np.array([
        float(row["gt_qx"]),
        float(row["gt_qy"]),
        float(row["gt_qz"]),
        float(row["gt_qw"])
    ]))

    T_est = evo_lie_algebra.se3(rotation, translation)
    T_ref = evo_lie_algebra.se3(ref_rotation, ref_translation)

    return T_est, T_ref

def camera_coordinate_to_world() -> np.ndarray:
    # we construct the transform that takes somethign in the robot convention
    # to the opencv convention and then apply the inverse
    return evo_lie_algebra.se3_inverse(se3(
        np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]]),
        np.array([0.0, 0.0, 0.0]))
    )

def transform_camera_trajectory_to_world(traj):
    from copy import deepcopy
    transform = camera_coordinate_to_world()
    traj_copy = deepcopy(traj)

    poses_in_robotic_convention = []
    poses = traj_copy.poses_se3
    for pose_cv in poses:
        pose_robotic = transform @ pose_cv @ evo_lie_algebra.se3_inverse(transform)
        poses_in_robotic_convention.append(pose_robotic)

    if isinstance(traj, evo_trajectory.PoseTrajectory3D):
        return evo_trajectory.PoseTrajectory3D(
            poses_se3=poses_in_robotic_convention,
            timestamps=traj.timestamps
        )
    elif isinstance(traj, evo_trajectory.PosePath3D):
        return evo_trajectory.PosePath3D(
            poses_se3=poses_in_robotic_convention
        )
    else:
        raise RuntimeError(f"Unknown trajectory type {type(traj)}")

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

    @property
    def x_limits(self):
        return [self.min_x, self.max_x]

    @property
    def y_limits(self):
        return [self.min_y, self.max_y]

    @property
    def z_limits(self):
        return [self.min_z, self.max_z]

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

    def set_ax_limits(self, ax: plt.Axes, plot_mode: evo_plot.PlotMode = evo_plot.PlotMode.xyz):
        from mpl_toolkits.mplot3d import Axes3D

        if  plot_mode == evo_plot.PlotMode.xyz and isinstance(ax, Axes3D):
            ax.set_xlim3d([self.min_x, self.max_x])
            ax.set_ylim3d([self.min_y, self.max_y])
            ax.set_zlim3d([self.min_z, self.max_z])
            set_axes_equal(ax)

        elif plot_mode == evo_plot.PlotMode.xy:
            ax.set_xlim([self.min_x, self.max_x])
            ax.set_ylim([self.min_y, self.max_y])


class ObjectMotionTrajectory(object):

    def __init__(self, pose_trajectory: evo_trajectory.PosePath3D, motion_trajectory: evo_trajectory.PosePath3D):
        from evo.core import sync
        self._pose_trajectory, self._motion_trajectory = sync.associate_trajectories(pose_trajectory, motion_trajectory)

        assert np.array_equal(self._pose_trajectory.timestamps,self._motion_trajectory.timestamps)

    def __repr__(self) -> str:
        return f"Poses {self._pose_trajectory} Motions {self._motion_trajectory}"

    class Iterator(object):

        def __init__(self, skip, timestamps, get_motion_call):
            self._skip = skip
            self._timestamps = timestamps
            self._get_motion_call = get_motion_call

        def __iter__(self):
            self._is_first = True
            self._current_timestamp_iterator = iter(self._timestamps)
            return self

        def __next__(self):
            timestamp = None
            if self._is_first or self._skip == 0:
                timestamp = next(self._current_timestamp_iterator)
                self._is_first = False
            else:
                for _ in range(self._skip):
                    timestamp = next(self._current_timestamp_iterator)

            assert timestamp is not None
            return self._get_motion_call(timestamp)

    @property
    def num_poses(self) -> int:
        assert self._motion_trajectory.num_poses == self._pose_trajectory.num_poses
        return self._motion_trajectory.num_poses

    def get_motion_with_pose_previous_iterator(self, skip: int = 0) -> Iterator:
        return ObjectMotionTrajectory.Iterator(skip, self._pose_trajectory.timestamps, self.get_motion_with_pose_previous)

    def get_motion_with_pose_current_iterator(self, skip: int = 0) -> Iterator:
        return ObjectMotionTrajectory.Iterator(skip, self._pose_trajectory.timestamps, self.get_motion_with_pose_current)

    def get_motion_with_pose_previous(self, k: int):
        return self._get_motion_with_pose(k, k-1)

    def get_motion_with_pose_current(self, k: int):
        return self._get_motion_with_pose(k, k)

    def calculate_velocity(self) -> np.ndarray:
        velocities = []
        for (pose_k_1, motion_k) in self.get_motion_with_pose_previous_iterator():
            if pose_k_1 is None or motion_k is None:
                continue
            I =  evo.core.transformations.identity_matrix()
            R_motion =  evo.core.transformations.identity_matrix()
            # ensure homogenous
            R_motion[0:3, 0:3] = motion_k[0:3, 0:3]

            t_motion = evo.core.transformations.translation_from_matrix(motion_k)
            t_pose = evo.core.transformations.translation_from_matrix(pose_k_1)

            # make homogenous
            t_motion = np.insert(t_motion, 3, 1)
            t_pose = np.insert(t_pose, 3, 1)

            velocity = t_motion - (I - R_motion) @ t_pose
            velocities.append(velocity[0:3])

        return np.array(velocities)



    def _get_motion_with_pose(self, k_motion: int, k_pose: int):
        k_pose_index = np.where(self._pose_trajectory.timestamps == k_pose)[0]
        # then this value is not in the pose timestamps
        if len(k_pose_index) == 0:
            return None, None

        k_motion_index = np.where(self._motion_trajectory.timestamps == k_motion)[0]
         # then this value is not in the motion timestamps
        if len(k_motion_index) == 0:
            return None, None

        # go from np.array to value
        k_pose_index = k_pose_index[0]
        k_motion_index = k_motion_index[0]

        return self._pose_trajectory.poses_se3[k_pose_index], self._motion_trajectory.poses_se3[k_motion_index]


def plot_traj(ax: Axes, plot_mode: evo_plot.PlotMode, traj: evo_trajectory.PosePath3D,
         style: str = '-', color='black', label: str = "", alpha: float = 1.0,
         plot_start_end_markers: bool = False, **kwargs) -> None:
    """
    plot a path/trajectory based on xyz coordinates into an axis
    :param ax: the matplotlib axis
    :param plot_mode: PlotMode
    :param traj: trajectory.PosePath3D or trajectory.PoseTrajectory3D object
    :param style: matplotlib line style
    :param color: matplotlib color
    :param label: label (for legend)
    :param alpha: alpha value for transparency
    :param plot_start_end_markers: Mark the start and end of a trajectory
                                   with a symbol.
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


# fix from here https://github.com/matplotlib/matplotlib/issues/21688
# TODO: clean up
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


def plot_trajectory_error(fig: Figure ,motion_est: evo_trajectory.PoseTrajectory3D, motion_ref: evo_trajectory.PoseTrajectory3D, label:str):
    assert motion_est.num_poses == motion_ref.num_poses

    import evo.core.metrics as metrics
    ape_E = metrics.APE(metrics.PoseRelation.full_transformation)
    data = (motion_ref, motion_est)
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

    rot_error_axes = fig.add_subplot(211)
    trans_error_axes = fig.add_subplot(212)

    rot_error_axes.set_title(f"Rotation Error ({label})", fontweight="bold")
    trans_error_axes.set_title(f"Translation Error ({label})", fontweight="bold")


    set_clean_background(rot_error_axes)
    set_clean_background(trans_error_axes)
    trans_error_axes.margins(0.001)
    rot_error_axes.margins(0.001)


    rot_error_axes.plot(rot_error[:,0], label="rx", color=get_nice_red())
    rot_error_axes.plot(rot_error[:,1], label="ry", color=get_nice_green())
    rot_error_axes.plot(rot_error[:,2], label="rz", color=get_nice_blue())

    trans_error_axes.plot(trans_error[:,0], label="tx", color=get_nice_red())
    trans_error_axes.plot(trans_error[:,1], label="ty", color=get_nice_green())
    trans_error_axes.plot(trans_error[:,2], label="tz", color=get_nice_blue())

    trans_error_axes.set_xlabel("Frame Index [-]")
    rot_error_axes.set_xlabel("Frame Index [-]")

    trans_error_axes.set_ylabel("$E_t$(m)", fontsize=19)
    rot_error_axes.set_ylabel("$E_r$(\N{degree sign})", fontsize=19)

    rot_error_axes.legend(loc='upper left')
    trans_error_axes.legend(loc='upper left')

    fig.tight_layout(pad=0.5)


# def plot_velocity_error(ax: Axes, xyz_velocity_est:np.array, xyz_velocity_ref:np.array, plot_xyz: bool = True):
#     assert xyz_velocity_est.shape == xyz_velocity_ref.shape
#     if plot_xyz:
#         xyz_velocity_error = xyz_velocity_ref - xyz_velocity_est

#         ax.plot(xyz_velocity_error[:,0], label="x", color='r')
#         ax.plot(xyz_velocity_error[:,1], label="y", color='g')
#         ax.plot(xyz_velocity_error[:,2], label="z", color='b')

#         ax.set_xlabel("Frame Index [-]")
#         ax.set_ylabel(r'Error [m/s]')
#     else:
#         xyz_velocity_est_norm = np.linalg.norm(xyz_velocity_est,axis=1)
#         xyz_velocity_ref_norm = np.linalg.norm(xyz_velocity_ref,axis=1)
#         err = xyz_velocity_ref_norm - xyz_velocity_est_norm
#         ax.plot(err, label="velocity", color='r')
#         ax.set_xlabel("Frame Index [-]")
#         ax.set_ylabel(r'Error [m/s]')

#     ax.legend()


def sync_and_align_trajectories(traj_est: Union[evo_trajectory.PosePath3D, evo_trajectory.PoseTrajectory3D],
                                traj_ref: Union[evo_trajectory.PosePath3D, evo_trajectory.PoseTrajectory3D],
                                discard_n_end_poses=-1,
                                max_diff=0.01) -> Union[evo_trajectory.PosePath3D, evo_trajectory.PoseTrajectory3D]:
    import copy
    from .tools import align_trajectory
    # We copy to distinguish another version that may be created
    traj_ref = copy.deepcopy(traj_ref)
    traj_est = copy.deepcopy(traj_est)

    # assume synched and in order!
    if isinstance(traj_est, evo_trajectory.PoseTrajectory3D) and isinstance(traj_ref, evo_trajectory.PoseTrajectory3D):
        traj_ref, traj_est = evo_sync.associate_trajectories(traj_ref, traj_est,max_diff=max_diff)
    traj_est = align_trajectory(traj_est, traj_ref, correct_scale = False,
                                                   n=discard_n_end_poses)
    return traj_est, traj_ref




#TODO: plotmode?
# TODO: outdoor vs indoor presets
# or just the ObjectMotionTrajectory
def plot_velocities(
        ax: Axes,
        object_trajectory: ObjectMotionTrajectory,
        color = 'r'):

    def draw_arrow(ax, xs, ys, zs, color):
        arrow = Arrow3D(xs, ys, zs, arrowstyle='-|>', color=color, mutation_scale=15, lw=3)
        # arrow = Arrow3D(xs, ys, zs, arrowstyle='-|>', color=color, mutation_scale=8, lw=1)
        ax.add_artist(arrow)

    for (pose_k_1, motion_k) in object_trajectory.get_motion_with_pose_previous_iterator(skip=8):
        if pose_k_1 is None or motion_k is None:
            continue
        I =  evo.core.transformations.identity_matrix()
        R_motion =  evo.core.transformations.identity_matrix()
        # ensure homogenous
        R_motion[0:3, 0:3] = motion_k[0:3, 0:3]

        t_motion = evo.core.transformations.translation_from_matrix(motion_k)
        t_pose = evo.core.transformations.translation_from_matrix(pose_k_1)

        # make homogenous
        t_motion = np.insert(t_motion, 3, 1)
        t_pose = np.insert(t_pose, 3, 1)

        # # implement ^o_Ad_B = (I - ^o_AR_B)^o_ot_A + ^o_At_B from Chirikjian
        # velocity = (I - R_motion) @ t_pose + t_motion
        # from VDO-SLAM
        velocity = t_motion - (I - R_motion) @ t_pose

        start = t_pose
        end = t_pose +  20.0 *velocity
        # end = t_pose +  3.0 *velocity
        draw_arrow(ax, (start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color)

        # arrow_length_ratio
        # ax.quiver(t_pose[0], t_pose[1], t_pose[2], velocity[0], velocity[1], velocity[2], length=5.0, normalize=False, color="red", pivot='tail')




# modification of evo.trajectories
# so much going on in this function - refactor and comment!!!!
def plot_object_trajectories(
        fig: Figure,
        obj_trajectories: typing.Dict[str, evo_trajectory.PosePath3D],
        obj_trajectories_ref: Optional[typing.Dict[str, evo_trajectory.PosePath3D]] = None,
        plot_mode=evo_plot.PlotMode.xy,
        title: str = "",
        subplot_arg: int = 111,
        plot_start_end_markers: bool = False,
        length_unit: evo_plot.Unit = evo_plot.Unit.meters,
        **kwargs) -> None:

    from .core.plotting import ObjectTrajectoryPlotter

    return ObjectTrajectoryPlotter().plot(
        fig,
        obj_trajectories,
        obj_trajectories_ref,
        plot_mode,
        title,
        subplot_arg,
        plot_start_end_markers,
        length_unit,
        **kwargs)


def calculate_omd_errors(traj, traj_ref, object_id):
    assert isinstance(traj, evo_trajectory.PoseTrajectory3D) and isinstance(traj_ref, evo_trajectory.PoseTrajectory3D)

    assert len(traj.poses_se3) == len(traj_ref.poses_se3)

    gt_pose_0 = traj_ref.poses_se3[0]
    est_pose_0 = traj.poses_se3[0]

    # the difference in pose of the estimated pose (at s = 0) and the gt pose (at s = 0), w.r.t the gt pose
    # if the poses are properly aligned, this will be identity
    error_0 = np.dot(evo_lie_algebra.se3_inverse(gt_pose_0), est_pose_0)

    t_errors = []
    rx_errors = []
    ry_errors = []
    rz_errors = []


    # TODO: can use object trajectory object now
    for est_pose, gt_pose in zip(traj.poses_se3[1:], traj_ref.poses_se3[1:]):
        # the difference in pose between the ground truth pose (at s = k) and the estimated pose (at s = k) w.r.t the estimated pose
        error_k = evo_lie_algebra.se3_inverse(est_pose) @ gt_pose
        # following equation 29 in https://arxiv.org/pdf/2110.15169
        # after treating all F's as I and j = k and k = 0, as per GE(l, t_k)
        err_se3 = error_k @ error_0
        err_log = gtsam.Pose3.Logmap(gtsam.Pose3(err_se3))
        err_rot = err_log[0:3]
        err_xyz = err_log[3:6]



        rx_errors.append(err_rot[0])
        ry_errors.append(err_rot[1])
        rz_errors.append(err_rot[2])

        #t error is reported separately
        err_t = np.linalg.norm(err_xyz)
        t_errors.append(err_t)

    t_errors = np.array(t_errors)
    # print(f"xyz largest error is {t_errors.max()} for {object_id}")
    # print(f"xyz avg error is {np.mean(t_errors)} for {object_id}")

    rx_errors = np.array(rx_errors) * 180.0/math.pi
    ry_errors = np.array(ry_errors) * 180.0/math.pi
    rz_errors = np.array(rz_errors) * 180.0/math.pi

    # print(f"rx largest error is {rx_errors.max()} for {object_id}")
    # print(f"rx avg error is {np.mean(rx_errors)} for {object_id}")

    # print(f"ry largest error is {ry_errors.max()} for {object_id}")
    # print(f"ry avg error is {np.mean(ry_errors)} for {object_id}")

    # print(f"rz largest error is {rz_errors.max()} for {object_id}")
    # print(f"rz avg error is {np.mean(rz_errors)} for {object_id}")




def reconstruct_trajectory_from_relative(traj, traj_ref):


    starting_pose = traj_ref.poses_se3[0]
    traj_poses = traj.poses_se3

    poses = [starting_pose]
    for traj_pose_k_1, traj_pose_k in zip(traj_poses[:-1], traj_poses[1:]):
        # motion = np.dot(traj_pose_k, evo_lie_algebra.se3_inverse(traj_pose_k_1))
        motion = traj_pose_k @ evo_lie_algebra.se3_inverse(traj_pose_k_1)
        # pose_k = np.dot(motion, poses[-1])
        pose_k = motion @ poses[-1]
        # relative_transform = evo_lie_algebra.relative_se3(traj_pose_k_1, traj_pose_k)
        # pose_k = np.dot(poses[-1], relative_transform)
        poses.append(pose_k)
    # print(poses)

    assert len(poses) == len(traj_poses)

    return evo_trajectory.PoseTrajectory3D(poses_se3=np.array(poses), timestamps=traj.timestamps)




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
