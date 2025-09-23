import csv
import numpy as np
import evo.core.trajectory as traj
import evo.core.transformations as tf
import evo.core.sync as sync
from evo.core.metrics import *
import evo.tools.plot as plot

from evo.main_ape import ape
from evo.main_rpe import rpe
from evo.main_ape_parser import parser

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_csv_to_evo_trajectory(csv_file, delimeter = ","):
    timestamps = []
    positions = []
    orientations = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=delimeter)

        # Peek at first line
        first_row = next(reader)

        # If first element is not numeric, treat as header
        if not is_numeric(first_row[0]):
            print(f"[INFO] Detected header in {csv_file}, skipping first row")
            # Read next row as first data row
            first_row = next(reader)

        # Process first row (whether header was skipped or not)
        while True:
            if len(first_row) < 8:
                break
            timestamp = float(first_row[0])
            tx, ty, tz = float(first_row[1]), float(first_row[2]), float(first_row[3])
            qx, qy, qz, qw = float(first_row[4]), float(first_row[5]), float(first_row[6]), float(first_row[7])

            timestamps.append(timestamp)
            positions.append([tx, ty, tz])
            orientations.append([qw, qx, qy, qz])

            try:
                first_row = next(reader)
            except StopIteration:
                break

    positions_np = np.array(positions)
    orientations_np = np.array(orientations)
    timestamps_np = np.array(timestamps)

    return traj.PoseTrajectory3D(positions_np, orientations_np, timestamps_np)

def load_and_synchronize_trajs(traj_ref, traj_est, max_time_diff=0.001):
    # Load both

    # Synchronize → this returns index pairs
    matches = sync.associate_trajectories(traj_ref, traj_est, max_diff=max_time_diff)

    # # Build synchronized trajectories
    # indices_ref = [i for i, j in matches]
    # indices_est = [j for i, j in matches]

    # traj_ref_sync = traj.PoseTrajectory3D(
    #     traj_ref.positions_xyz[indices_ref],
    #     traj_ref.orientations_quat_wxyz[indices_ref],
    #     traj_ref.timestamps[indices_ref]
    # )

    # traj_est_sync = traj.PoseTrajectory3D(
    #     traj_est.positions_xyz[indices_est],
    #     traj_est.orientations_quat_wxyz[indices_est],
    #     traj_est.timestamps[indices_est]
    # )

    return matches


dyna_vins = load_csv_to_evo_trajectory("/root/results/TRO2025/dyna_vins/city_day_high.csv", delimeter=" ")
ground_truth = load_csv_to_evo_trajectory("/root/data/VIODE/city_day/high/odometry_odom.csv")

print(dyna_vins)
print(ground_truth)


ground_truth_sync, dyna_vins_sync = load_and_synchronize_trajs(ground_truth, dyna_vins)

print(dyna_vins_sync)
print(ground_truth_sync)

# ape_metric = APE(PoseRelation.translation_part)
# ape_metric.process_data((ground_truth_sync, dyna_vins_sync))
# print("APE: ", ape_metric.get_all_statistics())


rpe_t_metric = RPE(PoseRelation.translation_part)
rpe_t_metric.process_data((ground_truth_sync, dyna_vins_sync))
print("RPE t ", rpe_t_metric.get_all_statistics())

rpe_r_metric = RPE(PoseRelation.rotation_angle_deg)
rpe_r_metric.process_data((ground_truth_sync, dyna_vins_sync))
print("RPE r ", rpe_r_metric.get_all_statistics())

est_name = "Dyna Vins"
ape_result = ape(ground_truth_sync, dyna_vins_sync, pose_relation=PoseRelation.translation_part, align=True,est_name=est_name)
print("APE: ",ape_result)

rpe_result_t = rpe(ground_truth_sync, dyna_vins_sync, delta=1, delta_unit=Unit.meters, pose_relation=PoseRelation.translation_part, align=True, est_name=est_name)
print("RPE t ", rpe_result_t)

rpe_result_r = rpe(ground_truth_sync, dyna_vins_sync, delta=1, delta_unit=Unit.degrees, pose_relation=PoseRelation.rotation_angle_deg, align=True,est_name=est_name)
print("RPE r ", rpe_result_r)


trajectories = {}
trajectories["Dyna VINS"] = dyna_vins_sync
trajectories["Ground Truth VO"] = ground_truth_sync

import matplotlib.pyplot as plt


fig_traj = plt.figure(figsize=(8,8))
plot.trajectories(fig_traj, trajectories, plot_mode=plot.PlotMode.xyz, plot_start_end_markers=True)
# plot_result()
plt.show()

# Example: plot
# fig = plot.plot_paths([ground_truth_sync, dyna_vins_sync], pose_relation=plot.PoseRelation.translation_part)
