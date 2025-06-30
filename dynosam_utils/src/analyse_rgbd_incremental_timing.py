import csv



def print_timing(file):
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
        print(f"{file} \n\tTiming mean {np.mean(timing)} last frame {last_frame_id}")


# print_timing("/root/results/Dynosam_ecmr2024/cluster_l1/hybrid_isam2_timing_inc.csv")
# print_timing("/root/results/Dynosam_ecmr2024/cluster_l1/wcme_isam2_timing_inc.csv")

sequence = "kitti_00_test"

print_timing(f"/root/results/Dynosam_ecmr2024/{sequence}/hybrid_isam2_timing_inc.csv")
print_timing(f"/root/results/Dynosam_ecmr2024/{sequence}/wcme_isam2_timing_inc.csv")
