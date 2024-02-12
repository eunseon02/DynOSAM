import csv
import os
from typing import Optional, List, Tuple, Dict
import logging
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from evo.core import lie_algebra
from evo.core import trajectory
import evo.tools.plot as evo_plot

from .tools import so3_from_euler

logger = logging.getLogger("dynosam_eval")

frontend_motion_error_log = "frontend_motion_error_log.csv"
frontend_camera_pose_error_log = "frontend_camera_pose_error_log.csv"
frontend_camera_pose_log = "frontend_camera_pose_log.csv"
frontend_object_pose_log = "frontend_object_pose_log.csv"


def read_csv(csv_file_path:str, expected_header: List[str]):
    assert csv_file_path is not None
    csvfile = open(csv_file_path)
    reader = csv.DictReader(csvfile)

    header = next(reader)
    keys = list(header.keys())
    #TODO: change to exception
    assert keys == expected_header
    return reader


class Evaluator(ABC):

    @abstractmethod
    def get_results_yaml() -> dict:
        pass

    @abstractmethod
    def make_plot() -> Optional[List[Tuple[plt.figure, str]]]:
        pass


def add_eval_plot(plot_collection: evo_plot.PlotCollection, evaluator: Evaluator):
    plots = evaluator.make_plot()

    if plots is None:
        return False

    for fig_name in plots:
        plot_collection.add_figure(
            name = fig_name[1],
            fig=fig_name[0])
    return True

class MotionErrorEvaluator(Evaluator):
    def __init__(self, motion_error_log, object_pose_log) -> None:
        self._motion_error_log = motion_error_log
        self.object_pose_log_path = object_pose_log

        self._motion_error_log_file = read_csv(self._motion_error_log, ["frame_id", "object_id", "t_err", "r_err"])
        self._object_pose_file = read_csv(self.object_pose_log_path, ["frame_id", "object_id", "x", "y", "z", "roll", "pitch", "yaw"])


        # result dictionary in the form
        # [object_id][frame_id]{[t_error, r_error]}
        self._motion_error_dict = {}

        for row in self._motion_error_log_file:
            object_id = int(row["object_id"])
            frame_id = int(row["frame_id"])

            if not object_id in self._motion_error_dict:
                self._motion_error_dict[object_id] = {}

            self._motion_error_dict[object_id][frame_id] = [
                float(row["t_err"]),
                float(row["r_err"])
            ]

        # {object_id: {frame_id : T}} -> T is a homogenous
        self._object_poses_dict = {}

        # assume frames are ordered!!!
        # used to construct the pose trajectires with evo
        object_poses_tmp_dict = {}

        for row in self._object_pose_file:
            frame_id = int(row["frame_id"])
            object_id = int(row["object_id"])

            if object_id not in object_poses_tmp_dict:
                object_poses_tmp_dict[object_id] = []

            if object_id not in self._object_poses_dict:
                self._object_poses_dict[object_id] = {}

            translation = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ])

            rotation = so3_from_euler(np.array([
                float(row["roll"]),
                float(row["pitch"]),
                float(row["yaw"])
            ]),
            order = "xyz",
            degrees=False)

            # frome evo
            T = lie_algebra.se3(rotation, translation)

            #log t in the nested dictionary
            self._object_poses_dict[object_id] = {frame_id: T}
            #log t in tmp object
            object_poses_tmp_dict[object_id].append(T)

        # {object_id: trajectory.PosePath3D}
        self._object_poses_traj = {}
        for object_id, poses in object_poses_tmp_dict.items():
            # This will need the poses to be in order?
            self._object_poses_traj[object_id] = trajectory.PosePath3D(poses_se3=poses)

    @property
    def object_poses_traj(self) -> Dict[int, trajectory.PosePath3D]:
        return self._object_poses_traj

    def get_results_yaml(self) -> dict:
        result_yaml = {}
        # per_frame_dict -> "object": {t_err, r_err}
        for object_id, per_frame_dict in self._motion_error_dict.items():
            t_error_np = np.array([per_frame_dict[frame][0] for frame in per_frame_dict.keys() if per_frame_dict[frame][0] != -1])
            r_error_np = np.array([per_frame_dict[frame][1] for frame in per_frame_dict.keys() if per_frame_dict[frame][1] != -1])

            result_yaml[object_id] = {
                "avg_t_err" : np.average(t_error_np),
                "avg_r" : np.average(r_error_np),
                "n_frame" : len(per_frame_dict)
            }

        return {"motion_error": result_yaml}

    def make_plot(self) -> Optional[List[Tuple[plt.figure, str]]]:
        fig_traj = plt.figure(figsize=(8,8))
        plot_mode = evo_plot.PlotMode.xyz

        ax_traj = evo_plot.prepare_axis(fig_traj, plot_mode)

        for object_id, traj in self._object_poses_traj.items():
            evo_plot.traj(ax_traj, plot_mode, traj, label=str(object_id), plot_start_end_markers=True)
        # evo_plot.trajectories(fig_traj, self._object_poses_traj, plot_mode=evo_plot.PlotMode.xyz)
        ax_traj.autoscale(True)

        return [(fig_traj, "Obj Traj")]



class CameraPoseEvaluator(Evaluator):
    def __init__(self, camera_pose_error_log, camera_pose_log) -> None:
        self._camera_pose_error_log = camera_pose_error_log
        self._camera_pose_log = camera_pose_log

        self._camera_pose_error_file = read_csv(self._camera_pose_error_log, ["frame_id", "t_abs_err", "r_abs_err", "t_rel_err", "r_rel_err"])
        self._camera_pose_file = read_csv(self._camera_pose_log, ["frame_id", "x", "y", "z", "roll", "pitch", "yaw"])

        # in same form as input but now everything is a float
        # key is frame_id and then array
        self._camera_error_dict = {}

        for row in self._camera_pose_error_file:
            frame_id = int(row["frame_id"])
            self._camera_error_dict[frame_id] = [
                float(row["t_abs_err"]),
                float(row["r_abs_err"]),
                float(row["t_rel_err"]),
                float(row["r_rel_err"])
            ]

        # saved as homogenous matrix, key is frame_id
        self._camera_pose_dict = {}

        poses = []

        for row in self._camera_pose_file:
            frame_id = int(row["frame_id"])

            translation = np.array([
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            ])

            rotation = so3_from_euler(np.array([
                float(row["roll"]),
                float(row["pitch"]),
                float(row["yaw"])
            ]),
            order = "xyz",
            degrees=False)

            # frome evo
            T = lie_algebra.se3(rotation, translation)
            self._camera_pose_dict[frame_id] = T
            poses.append(T)

        self._camera_pose_traj = trajectory.PosePath3D(poses_se3=poses)

    @property
    def camera_pose_traj(self) -> trajectory.PosePath3D:
        return self._camera_pose_traj

    def get_results_yaml(self) -> dict:
        result_yaml = {}
        #"frame_id" : {"t_abs_error": val, .... "r_rel_err": val}
        errors = self._camera_error_dict
        # absolute errors
        t_abs_err_np = np.array([errors[frame][0] for frame in errors.keys() if errors[frame][0] != -1])
        r_abs_err_np = np.array([errors[frame][1] for frame in errors.keys() if errors[frame][1] != -1])
        # relative errors
        t_rel_err_np = np.array([errors[frame][2] for frame in errors.keys() if errors[frame][2] != -1])
        r_rel_err_np = np.array([errors[frame][3] for frame in errors.keys() if errors[frame][3] != -1])

        result_yaml = {
            "avg_t_abs_err" : np.average(t_abs_err_np),
            "avg_r_abs_err" : np.average(r_abs_err_np),
            "avg_t_rel_err" : np.average(t_rel_err_np),
            "avg_r_rel_err" : np.average(r_rel_err_np)
        }

        return {"camera_pose": result_yaml}

    def make_plot(self) -> Optional[List[Tuple[plt.figure, str]]]:
        fig_traj = plt.figure(figsize=(8,8))
        plot_mode = evo_plot.PlotMode.xyz

        ax_traj = evo_plot.prepare_axis(fig_traj, plot_mode)
        evo_plot.traj(ax_traj, plot_mode, self._camera_pose_traj)

        return [(fig_traj, "Camera Trajectory")]

class EgoObjectMotionEvaluator(Evaluator):

    def __init__(self, camera_eval: CameraPoseEvaluator, object_eval: MotionErrorEvaluator):
        self._camera_eval = camera_eval
        self._object_eval = object_eval

    def get_results_yaml(self) -> dict:
        return None

    def make_plot(self) -> Optional[List[Tuple[plt.figure, str]]]:
        fig_traj = plt.figure(figsize=(8,8))
        camera_traj = self._camera_eval.camera_pose_traj
        object_trajs = self._object_eval.object_poses_traj

        all_traj = object_trajs
        all_traj["Camera"] = camera_traj

        evo_plot.trajectories(fig_traj, all_traj, plot_mode=evo_plot.PlotMode.xyz)

        return [(fig_traj, "Trajectires")]



class DatasetEvaluator:
    def __init__(self, output_folder_path:str, args) -> None:
        self._output_folder_path = output_folder_path
        self._args = args
        # frontend file paths
        self._frontend_motion_error_log_path = self._create_file_path(frontend_motion_error_log)
        self._frontend_object_pose_log_path = self._create_file_path(frontend_object_pose_log)
        self._frontend_camera_pose_error_log_path = self._create_file_path(frontend_camera_pose_error_log)
        self._frontend_camera_pose_log_path = self._create_file_path(frontend_camera_pose_log)

    def run_analysis(self):
        self._analyse_frontend()

    def _analyse_frontend(self):
        # analyse frontend object motion error
        if self._frontend_motion_error_log_path is None:
            logger.warning("No file {} to analyse", self._frontend_motion_error_log_path)

        frontend_motion_eval = MotionErrorEvaluator(
            self._frontend_motion_error_log_path,
            self._frontend_object_pose_log_path)

        # print(frontend_motion_eval.get_results_yaml())

        frontend_camera_pose_eval = CameraPoseEvaluator(
            self._frontend_camera_pose_error_log_path,
            self._frontend_camera_pose_log_path
        )
        # print(frontend_camera_pose_eval.get_results_yaml())


        self._save_results_file("fontend_results", [frontend_motion_eval, frontend_camera_pose_eval])

        all_eval = EgoObjectMotionEvaluator(frontend_camera_pose_eval, frontend_motion_eval)

        plot_collection = evo_plot.PlotCollection("Frontend")
        add_eval_plot(plot_collection, frontend_camera_pose_eval)
        add_eval_plot(plot_collection, frontend_motion_eval)
        add_eval_plot(plot_collection, all_eval)


        plot_collection.show()


    def _analyse_backend(self):
        pass

    def _save_results_file(self, file_name: str, evaluators: List[Evaluator]):
        output_file_name = os.path.join(self._output_folder_path, file_name) + ".json"

        logger.info("Writing results json to file {}".format(output_file_name))

        output_json = {}

        for evals  in evaluators:
            result = evals.get_results_yaml()
            output_json.update(result)

        import json

        with open(output_file_name, "w") as output_file:
            json.dump(output_json, output_file)


    def _create_file_path(self, file):
        file_path = os.path.join(self._output_folder_path, file)

        if not os.path.exists(file_path):
            return None
        return file_path
