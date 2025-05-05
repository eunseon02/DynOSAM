import csv

# file = "/root/results/Dynosam_ecmr2024/kitti_0020/object_centric_isam2_timing_inc.csv"
file = "/root/results/Dynosam_ecmr2024/kitti_0020/rgbd_motion_world_isam2_timing_inc.csv"

with open(file) as csvfile:
    reader = csv.reader(csvfile)

    timing = []
    last_frame_id = 0
    for r in reader:
        # awful header form
        # timing (ms), frame, opt values size, graph size
        timing.append(int(r[0]))
        last_frame_id = max(int(r[1]), last_frame_id)

    import numpy as np
    timing = np.array(timing)
    print(f"Timing mean {np.mean(timing)} last frame {last_frame_id}")
