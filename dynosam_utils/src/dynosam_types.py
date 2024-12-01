
# pip3 install pymongo, NOT bson
from dataclasses import dataclass, field, fields
from typing import List, Optional, Type, TypeVar, Any, Dict, get_type_hints, Tuple, Union
import numpy as np

def so3_from_quat(qw: float, qx: float, qy: float, qz: float):
    from scipy.spatial.transform import Rotation as R
    # from docs order is (x, y, z, w) format
    return R.from_quat([qx, qy, qz, qw]).as_matrix()

T = TypeVar('T', bound='FromDict')

class FromDict:
    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        fieldtypes = get_type_hints(cls)
        return cls(**{k: cls._convert_value(fieldtypes[k], v) for k, v in data.items() if k in fieldtypes})

    @staticmethod
    def _convert_value(fieldtype: Type, value: Any) -> Any:
        origin_type = getattr(fieldtype, '__origin__', None)
        if origin_type is list:
            item_type = fieldtype.__args__[0]
            return [FromDict._convert_value(item_type, v) for v in value]
        elif origin_type is Optional:
            item_type = fieldtype.__args__[0]
            return FromDict._convert_value(item_type, value) if value is not None else None
        elif isinstance(value, dict) and hasattr(fieldtype, 'from_dict'):
            return fieldtype.from_dict(value)
        else:
            return value

@dataclass
class Pose3(FromDict):
    tx: float
    ty: float
    tz: float

    qw: float
    qx: float
    qy: float
    qz: float

    matrix: np.array = field(init=False) #4x4 homogenous matrix - set in the __post__init__

    def __post_init__(self):
        self.matrix = np.eye(4)

        r = so3_from_quat(self.qw, self.qx, self.qy, self.qz)
        t = np.array([self.tx, self.ty, self.tz])
        self.matrix[:3, :3] = r
        self.matrix[:3, 3] = t

    def __repr__(self) -> str:
        return f'{self.matrix}'

@dataclass
class TrackedMeasurement(FromDict):

    frame_id: int # which frame this observation is from
    tracklet_id: int # id, associating tracked feature between frames
    object_id: int # object label which is consistent between frames and indicates which obect the feature is tracked on. 0 if background (i.e static)
    # reference_frame: str
    value: np.array # Nx1 vector measurements in reference_frame frame

@dataclass
class LandmarkStatus(TrackedMeasurement):
    method: str
    reference_frame: str #lmk has a reference frame but keypoint does not (its always the image plane!!)

    def __post_init__(self):
        self.value = np.array(self.value)
        assert self.value.shape == (3, 1)

@dataclass
class KeypointStatus(TrackedMeasurement):
    kp_type: str

    def __post_init__(self):
        self.value = np.array(self.value)
        assert self.value.shape == (2, 1)

@dataclass
class MotionEstimate(FromDict):
    #NOTE: 'estimate' is mis-leanding, it does not have to be an estimate from an optimisation routine, its just any SE3 value
    reference_frame:str # which reference frame the estimate is in
    estimate: Pose3

@dataclass
class ObjectPoseGT(FromDict):
    frame_id: int
    object_id: int

    L_camera: Pose3 # pose of the object in the camera frame
    L_world: Pose3  # pose of the object in the the world frame

    bounding_box: Any
    object_dimensions: Any

    prev_H_current_world: Optional[Pose3] # ^W_{k-1}H_k - motion in world
    prev_H_current_L: Optional[Pose3] # ^L_{k-1}H_k - motion in body frame
    prev_H_current_X: Optional[Pose3] # ^X_{k-1}H_k - motion in camera frame

    motion_info: Any


@dataclass
class GroundTruthInputPacket(FromDict):
    timestamp: float # time ins seoncs
    frame_id: int
    X_world: Pose3 # pose of the camera in the world frame
    objects: List[ObjectPoseGT]


@dataclass
class RGBDInstanceOutputPacket(FromDict):
    frame_id: int
    timestamp: float

    frontend_type: int

    #vector of tracked points/feature in this frame
    # static/dynamic lists should be the same length as the landmarks are just projected
    # from the keypoints in the associated *_keypoints list
    dynamic_landmarks: List[LandmarkStatus] #measurement should be a 3x1 landmark
    static_landmarks: List[LandmarkStatus] #measurement should be a 3x1 landmark
    static_keypoints: List[KeypointStatus] #measurement should be a 2x1 keypoint
    dynamic_keypoints: List[KeypointStatus] #measurement should be a 2x1 keypoint

    # 4x4 homogenous matrix
    T_world_camera: Pose3
    camera_poses: List[Pose3]

    ground_truth: GroundTruthInputPacket


    # all these are updated in the __post_init__
    estimated_motions: Any # after __post_init__ it will be a Dict[objectId(int), MotionEstimate]
    propogated_object_poses: Any # after __post__init will be a map of object ids -> map of frame ids to Pose3

    def __repr__(self) -> str:
        return f' frame id {self.frame_id} T_world_camera {self.T_world_camera} gt {self.ground_truth}'


    def __post_init__(self):
        new_estimated_motions = {}
        for object_id, map in self.estimated_motions:
            new_estimated_motions[object_id] = MotionEstimate.from_dict(map)
        self.estimated_motions = new_estimated_motions

        new_propogated_obejct_poses = {}
        for object_id, map in self.propogated_object_poses:
            new_propogated_obejct_poses[object_id] = {}
            for frame_id, pose in map:
                new_propogated_obejct_poses[object_id][frame_id] = Pose3.from_dict(pose)
        self.propogated_object_poses = new_propogated_obejct_poses


def load_frontend_bson(path:str):
    import bson
    with open(path,'rb') as f:
        content = f.read()
        data = bson.decode_all(content)
        return data

def load_to_frontend_packet(path:str):
    decoded_json = load_frontend_bson(path)

    rgbd_frontend_data_list = []
    # not a map in the python sense
    # actually a list of frontendoutputs, which is a 'list' where the first element is the frame id
    # and the second element is the actual frontend output packet
    frontend_output_packet_map = decoded_json[0]['data']
    for _, rgbd_frontend_data_packet in frontend_output_packet_map:
        rgbd_frontend_data = RGBDInstanceOutputPacket.from_dict(rgbd_frontend_data_packet)
        print(rgbd_frontend_data.static_keypoints)
        # return
        # rgbd_frontend_data_list.append(rgbd_frontend_data)
    return rgbd_frontend_data_list



if __name__ == '__main__':
    # load_to_frontend_packet('/root/results/Dynosam_tro2024/omd_swinging_4_unconstrained_sliding_500/rgbd_frontend_output.bson')
    load_to_frontend_packet('/root/results/DynoSAM/test_small_frontend/rgbd_frontend_output.bson')
