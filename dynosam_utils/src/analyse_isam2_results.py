from dynosam_utils.evaluation.tools import load_bson

results = load_bson("/root/results/DynoSAM/incremental_test/parallel_isam2_results.bson")[0]['data']


for object_id, per_frame_results in results.items():
    print(f"Object id {object_id}")
    for frame, object_isam_result in per_frame_results.items():
        was_smoother_ok = bool(object_isam_result["was_smoother_ok"])
        frame_id = int(object_isam_result["frame_id"])
        full_isam2_result = object_isam_result["isam_result"]
        motion_variable_status = object_isam_result["motion_variable_status"]
        print(f"frame id {frame_id} {motion_variable_status}")
