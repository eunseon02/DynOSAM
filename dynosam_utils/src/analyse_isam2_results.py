from dynosam_utils.evaluation.tools import load_bson
import matplotlib.pyplot as plt

import sys

# results = load_bson("/root/results/Dynosam_ecmr2024/omd_swinging_4_unconstrained/parallel_isam2_results.bson")[0]['data']
results = load_bson("/root/results/Dynosam_ecmr2024/kitti_0000/parallel_isam2_results.bson")[0]['data']


# results = load_bson("/root/results/DynoSAM/incremental_test/parallel_isam2_results.bson")[0]['data']

# per object at frame how many variables were involved

object_map = {}



for object_id, per_frame_results in results.items():
    print(f"Object id {object_id}")
    for frame, object_isam_result in per_frame_results.items():
        was_smoother_ok = bool(object_isam_result["was_smoother_ok"])
        if not was_smoother_ok:
            continue
        frame_id = int(object_isam_result["frame_id"])
        full_isam2_result = object_isam_result["isam_result"]
        timing = float(object_isam_result["timing"])
        # collect_involved_variables(full_isam2_result, object_map, frame_id, object_id)

        if object_id not in object_map:
            object_map[object_id] = {
                "frames": [],
                "variables_reeliminated": [],
                "variables_relinearized": [],
                "timing": [],
                "average_clique_size": [],
                "max_clique_size": []   }

        object_map[object_id]["frames"].append(frame_id)
        object_map[object_id]["variables_reeliminated"].append(int(full_isam2_result["variables_reeliminated"]))
        object_map[object_id]["variables_relinearized"].append(int(full_isam2_result["variables_relinearized"]))
        object_map[object_id]["timing"].append(timing)
        object_map[object_id]["average_clique_size"].append(float(object_isam_result["average_clique_size"]))
        object_map[object_id]["max_clique_size"].append(float(object_isam_result["max_clique_size"]))



        # motion_variable_status = object_isam_result["motion_variable_status"]
        # print(f"frame id {frame_id} {motion_variable_status}")



# Function to plot data
def plot_variable(variable_name, ylabel, title, **kwargs):
    plt.figure()
    for object_id, data in object_map.items():
        frames = data["frames"]
        values = data[variable_name]

        # Sort frames and associated values
        sorted_indices = sorted(range(len(frames)), key=lambda i: frames[i])
        frames = [frames[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        plt.plot(frames, values, linestyle='-', label=f'Object {object_id}', **kwargs)

    plt.xlabel("Frame ID")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Plot each variable
plot_variable("variables_reeliminated", "Variables Reeliminated", "Number of Variables Reeliminated Per Frame Per Object")
plot_variable("variables_relinearized", "Variables Relinearized", "Number of Variables Relinearized Per Frame Per Object")
plot_variable("timing", "Timing", "Timing Per Frame Per Object")
plot_variable("average_clique_size", "Avg. Clique Size", "Avg. Clique Size Per Frame Per Object")
plot_variable("max_clique_size", "Max Clique Size", "Max Clique Size Per Frame Per Object")
