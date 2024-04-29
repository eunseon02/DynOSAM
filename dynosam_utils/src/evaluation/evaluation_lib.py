import csv
import os
import logging
from typing import Optional, List, Tuple, Dict, TypeAlias
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from evo.core import lie_algebra
from evo.core import trajectory
import evo.tools.plot as evo_plot

from .tools import so3_from_euler
from .result_types import (
    MotionErrorDict,
    ObjectPoseDict,
    ObjectTrajDict,
    ObjectPoseErrorDict,
    analyse_motion_error_dict,
    analyse_object_pose_errors
)

logger = logging.getLogger("dynosam_eval")
# ----> console info messages require these lines <----
# create console handler and set level to debug
_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter(
    fmt='[%(levelname)s][%(filename)s:%(lineno)d] %(name)s: %(message)s'))
# add ch to logger
logger.addHandler(_ch)




def read_csv(csv_file_path:str, expected_header: List[str]):
    assert csv_file_path is not None
    csvfile = open(csv_file_path)
    reader = csv.DictReader(csvfile)

    header = next(reader)
    keys = list(header.keys())
    if keys != expected_header:
        raise Exception(
            "Csv file headers were not valid when loading file at path: {}. "
            "Expected header was {} but actual keys were {}".format(
                csv_file_path, expected_header, keys
            )
        )

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
    def __init__(self, motion_error_log:str, object_pose_log:str, object_pose_error_log:str) -> None:
        self._motion_error_log = motion_error_log
        self._object_pose_log = object_pose_log
        self._object_pose_error_log = object_pose_error_log

        self._motion_error_log_file = read_csv(self._motion_error_log, ["frame_id", "object_id", "t_err", "r_err"])
        self._object_pose_log_file = read_csv(self._object_pose_log, ["frame_id", "object_id", "x", "y", "z", "roll", "pitch", "yaw"])
        self._object_pose_error_log_file = read_csv(self._object_pose_error_log, ["frame_id", "object_id", "t_abs_err", "r_abs_err", "t_rel_err", "r_rel_err"])


        # result dictionary in the form
        # [object_id][frame_id]{[t_error, r_error]}
        self._motion_error_dict: MotionErrorDict = MotionErrorEvaluator._construct_motion_error_dict(self._motion_error_log_file)
        # {object_id: {frame_id : T}} -> T is a homogenous, and {object_id: trajectory.PosePath3D}
        self._object_poses_dict: ObjectPoseDict = {}
        self._object_poses_traj: ObjectTrajDict = {}
        self._object_poses_dict, self._object_poses_traj = MotionErrorEvaluator._construction_object_poses(self._object_pose_log_file)

        self._object_pose_error_dict: ObjectPoseErrorDict = MotionErrorEvaluator._construct_pose_error_dict( self._object_pose_error_log_file)



    @property
    def object_poses_traj(self) -> ObjectTrajDict:
        return self._object_poses_traj

    @property
    def object_pose_errors(self) -> ObjectPoseErrorDict:
        return self._object_pose_error_dict

    @property
    def object_motion_errors(self) -> MotionErrorDict:
        return self._motion_error_dict

    def get_results_yaml(self) -> dict:
        return {"object_motion_error": analyse_motion_error_dict(self.object_motion_errors),
                "object_pose_errors": analyse_object_pose_errors(self.object_pose_errors)}

    def make_plot(self) -> Optional[List[Tuple[plt.figure, str]]]:
        fig_traj = plt.figure(figsize=(8,8))
        plot_mode = evo_plot.PlotMode.xyz

        ax_traj = evo_plot.prepare_axis(fig_traj, plot_mode)

        for object_id, traj in self._object_poses_traj.items():
            evo_plot.traj(ax_traj, plot_mode, traj, label=str(object_id), plot_start_end_markers=True)
        # evo_plot.trajectories(fig_traj, self._object_poses_traj, plot_mode=evo_plot.PlotMode.xyz)
        ax_traj.autoscale(True)

        return [(fig_traj, "Obj Traj")]

    @staticmethod
    def _construct_motion_error_dict(motion_error_log_file: csv.DictReader) -> MotionErrorDict:
        """
        Constructs a motion error dict using the error results contained in the motion_error_log_file.
        The motion_error_log_file is expected to have header ["frame_id", "object_id", "t_err", "r_err"]
        The motion_error_dictwill have form
        {
            object_id: {
                frame_id: [t_error: float, r_error: float]
            }
        }

        Args:
            motion_error_log_file (csv.DictReader): Input reader

        Returns:
            MotionErrorDict: Motion error dict
        """
        motion_error_dict: MotionErrorDict = {}

        for row in motion_error_log_file:
            object_id = int(row["object_id"])
            frame_id = int(row["frame_id"])

            if not object_id in motion_error_dict:
                motion_error_dict[object_id] = {}

            motion_error_dict[object_id][frame_id] = [
                float(row["t_err"]),
                float(row["r_err"])
            ]
        return motion_error_dict

    @staticmethod
    def _construction_object_poses(object_poses_log_file: csv.DictReader) -> Tuple[ObjectPoseDict, ObjectTrajDict]:
        """
        Constructs dictionaries representing the object poses from the object_poses_log_file.
        The object_poses_log_file is expected to have a header with the form ["frame_id", "object_id", "x", "y", "z", "roll", "pitch", "yaw"].
        The function returnes two dictionaries with the poses per object in different forms.
        First return is an object pose dictionary in the form
        {
            object_id: { frame_id: T }
        }
        where T is a [4x4] homogenous matrix (np.ndarray).

        Second return is an dictionary of trajectory.PosePath3D (from the evo lib) per object:
        {
            object_id: trajectory.PosePath3D
        }

        Args:
            object_poses_log_file (csv.DictReader[str]): Input reader

        Raises:
            FileNotFoundError: _description_

        Returns:
            Tuple[ObjectPoseDict, ObjectTrajDict]: _description_
        """
        object_poses_dict: ObjectPoseDict = {}

        # assume frames are ordered!!!
        # used to construct the pose trajectires with evo
        object_poses_tmp_dict = {}

        for row in object_poses_log_file:
            frame_id = int(row["frame_id"])
            object_id = int(row["object_id"])

            if object_id not in object_poses_tmp_dict:
                object_poses_tmp_dict[object_id] = []

            if object_id not in object_poses_dict:
                object_poses_dict[object_id] = {}

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
            object_poses_dict[object_id] = {frame_id: T}
            #log t in tmp object
            object_poses_tmp_dict[object_id].append(T)

        # {object_id: trajectory.PosePath3D}
        object_poses_traj: ObjectTrajDict  = {}
        for object_id, poses in object_poses_tmp_dict.items():
            # This will need the poses to be in order?
            object_poses_traj[object_id] = trajectory.PosePath3D(poses_se3=poses)

        return object_poses_dict, object_poses_traj

    @staticmethod
    def _construct_pose_error_dict(object_pose_error_log_file: csv.DictReader) -> ObjectPoseErrorDict:
        """
        Constructs a object pose error dict using the error results contained in the object_pose_error_log_file.
        The object pose error is expected to have header ["frame_id", "object_id", "t_abs_err", "r_abs_err", "t_rel_err", "r_rel_err"].
        The object pose error dict will have form
        {
            object_id: {
                frame_id: [t_abs_err: float, r_abs_err: float, t_rel_err: float, r_rel_err: float]
            }
        }


        Args:
            object_pose_error_log_file (csv.DictReader): _description_

        Raises:
            FileNotFoundError: _description_

        Returns:
            ObjectPoseErrorDict: _description_
        """
        object_pose_error_dict: ObjectPoseErrorDict = {}

        for row in object_pose_error_log_file:
            object_id = int(row["object_id"])
            frame_id = int(row["frame_id"])

            if not object_id in object_pose_error_dict:
                object_pose_error_dict[object_id] = {}

            object_pose_error_dict[object_id][frame_id] = [
                float(row["t_abs_err"]),
                float(row["r_abs_err"]),
                float(row["t_rel_err"]),
                float(row["r_rel_err"])
            ]
        return object_pose_error_dict



class CameraPoseEvaluator(Evaluator):
    def __init__(self, camera_pose_error_log:str, camera_pose_log:str) -> None:
        self._camera_pose_error_log:str = camera_pose_error_log
        self._camera_pose_log:str = camera_pose_log

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


class DataFiles:
    def __init__(self, prefix:str, **kwargs) -> None:
        # These files will match the output logger file names from Logger.cc
        # as the prefix here is the prefix used in the logger
        self.prefix = prefix
        self.results_file_name = kwargs.get("results_file_name", self.prefix + "_results")
        self.plot_collection_name = kwargs.get("plot_collection_name", self.prefix.capitalize())

        self.motion_error_log = prefix + "_object_motion_error_log.csv"
        self.object_pose_log = prefix + "_object_pose_log.csv"
        self.object_pose_error_log = prefix + "_object_pose_errors_log.csv"
        self.camera_pose_error_log = prefix + "_camera_pose_error_log.csv"
        self.camera_pose_log = prefix + "_camera_pose_log.csv"
        self.map_point_log = prefix + "_map_points_log.csv"

    def __str__(self):
        return "DataFiles - prefix: {}, result file name: {}, plot collection name: {}".format(
            self.prefix,
            self.results_file_name,
            self.plot_collection_name
        )


class DatasetEvaluator:
    def __init__(self, output_folder_path:str, args) -> None:
        self._output_folder_path = output_folder_path
        self._args = args

    def run_analysis(self):
        logger.info("Running analysis using files at output path {}".format(self._output_folder_path))

        frontend_files = DataFiles("frontend")
        self.run_single_analysis(frontend_files)

        # backend_files = DataFiles("backend")
        # self.run_single_analysis(backend_files)

        self.run_single_analysis(DataFiles("rgbd_primitive_backend"))

    def run_single_analysis(self, datafiles: DataFiles):
        prefix = datafiles.prefix

        #TODO: this does not work...?
        analysis_logger = logger.getChild(prefix)

        analysis_logger.info("Running analysis with datafile prefix: {}".format(prefix))
        motion_error_log_path = self._create_existing_file_path(datafiles.motion_error_log)
        object_pose_log_path = self._create_existing_file_path(datafiles.object_pose_log)
        object_pose_error_log_path = self._create_existing_file_path(datafiles.object_pose_error_log)
        camera_pose_error_log_path = self._create_existing_file_path(datafiles.camera_pose_error_log)
        camera_pose_log_path = self._create_existing_file_path(datafiles.camera_pose_log)

        evaluators = []
        motion_eval = self._check_and_cosntruct_generic_eval(
            MotionErrorEvaluator,
            motion_error_log_path,
            object_pose_log_path,
            object_pose_error_log_path
        )

        camera_pose_eval = self._check_and_cosntruct_generic_eval(
            CameraPoseEvaluator,
            camera_pose_error_log_path,
            camera_pose_log_path
        )

        if motion_eval:
            analysis_logger.info("Adding motion eval")
            evaluators.append(motion_eval)

        if camera_pose_eval:
            analysis_logger.info("Adding camera pose eval")
            evaluators.append(camera_pose_eval)

        self._save_results_file(datafiles.results_file_name, evaluators)

        if camera_pose_eval and motion_eval:
            analysis_logger.info("Adding EgoObjectMotionEvaluator")
            evaluators.append(EgoObjectMotionEvaluator(camera_pose_eval, motion_eval))

        plot_collection_name = datafiles.plot_collection_name
        analysis_logger.info("Constructing plot collection {}".format(plot_collection_name))
        plot_collection = evo_plot.PlotCollection(plot_collection_name)

        for evals  in evaluators:
            add_eval_plot(plot_collection, evals)


        plot_collection.show()


    def _save_results_file(self, file_name: str, evaluators: List[Evaluator]):
        if len(evaluators) == 0:
            logger.warning("No evaluators provided to save results!")
            return

        output_file_name = self._create_new_file_path(file_name) + ".json"

        logger.info("Writing results json to file {}".format(output_file_name))

        output_json = {}

        for evals  in evaluators:
            result = evals.get_results_yaml()

            if result:
                output_json.update(result)

        import json

        with open(output_file_name, "w") as output_file:
            json.dump(output_json, output_file)

    def _create_new_file_path(self, file:str) -> str:
        return os.path.join(self._output_folder_path, file)

    def _create_existing_file_path(self, file:str) -> Optional[str]:
        """
        Create a full file path at the output folder path using the input file name to an existing file path.
        Checks that a file exists at the path given

        Args:
            file (str): Input file name

        Returns:
            Optional[str]: Absolute file path or None if path does not exist
        """
        file_path = os.path.join(self._output_folder_path, file)

        if not os.path.exists(file_path):
            return None
        return file_path

    def _check_and_cosntruct_generic_eval(self, cls, *args):
        import inspect

        if not inspect.isclass(cls):
            logger.fatal("Argument cls ({}) shoudl be a class type and not an instance".format(cls))

        if not issubclass(cls, Evaluator):
            logger.fatal("Argument cls ({}) must derive from Evaluator!".format(cls.__name__))

        for arg in args:
            assert isinstance(arg, str)
            if arg is None:
                logger.warning("Expected file path is none when constructing evaluator {} with args: {}".format(cls.__name__, args))
                return None

        logger.info("Constructor evaluator: {}".format(cls.__name__))
        return cls(*args)
