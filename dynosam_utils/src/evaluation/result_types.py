from typing import Optional, List, Tuple, Dict, TypeAlias
import numpy as np
from evo.core import trajectory



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
