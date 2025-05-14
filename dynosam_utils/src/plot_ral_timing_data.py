import matplotlib.ticker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib
from dynosam_utils.evaluation.tools import load_bson
from dynosam_utils.evaluation.core.plotting import startup_plotting



import numpy as np

plt.rcdefaults()

plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    })

startup_plotting(25)
# plt.rcParams['axes.titlesize'] = 25    # Title font size
# plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
# plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
# plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size
# plt.rcParams['legend.fontsize']=18



# collect all ph results
def collect_all_ph_results():
    data = []

    for name in ["omd_swinging_4_unconstrained_long",
                 "kitti_0000",
                 "kitti_0003",
                 "kitti_0004",
                 "kitti_0005",
                 "kitti_0006",
                 "kitti_0018",
                 "kitti_0020"]:
        results = load_bson(f"/root/results/Dynosam_ecmr2024/{name}/parallel_isam2_results.bson")[0]['data']
        frame_map = {}
        for object_id, per_frame_results in results.items():
            for frame, object_isam_result in per_frame_results.items():
                was_smoother_ok = bool(object_isam_result["was_smoother_ok"])
                if not was_smoother_ok:
                    continue
                frame_id = int(object_isam_result["frame_id"])
                timing = float(object_isam_result["timing"])

                if frame_id not in frame_map:
                    frame_map[frame_id] = []

                frame_map[frame_id].append(timing)

        avg_per_frame = []
        for frame_id, timing in frame_map.items():
            timing_mean = np.mean(np.array(timing))
            avg_per_frame.append(timing_mean)
            # data.append(timing_mean)

        data.append(np.mean(np.array(avg_per_frame)))
        # print(data)

    return {"Parallel-Hybrid\n(ours)":data}


def collect_regular_results(suffix):

    import csv
    assert "rgbd_motion_world" == suffix or "object_centric" == suffix
    data = []
    for name in ["omd_swinging_4_unconstrained_long",
                 "kitti_0000",
                 "kitti_0003",
                 "kitti_0004",
                 "kitti_0005",
                 "kitti_0006",
                 "kitti_0018",
                 "kitti_0020"]:
        file = f"/root/results/Dynosam_ecmr2024/{name}/{suffix}_isam2_timing_inc.csv"
        timing_data = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile)

            for r in reader:
                # awful header form
                # timing (ms), frame, opt values size, graph size
                timing_data.append(int(r[0]))
                # data.append(int(r[0]))
    # print(data)
        data.append(np.mean(np.array(timing_data)))

    if suffix == "rgbd_motion_world":
        return {"SOA Comparison":data}
    else:
        return {"Hybrid\n(ours)":data}



# --- Data ---
means = {}
means.update(collect_all_ph_results())
print("object_centric")
means.update(collect_regular_results("object_centric"))
print("rgbd_motion_world")
means.update(collect_regular_results("rgbd_motion_world"))


# # Raw timing data from sequences
# means = {
#     "Parallel-Hybrid": [278, 252, 78, 179, 49, 651, 179, 612],
#     "Full-Hybrid": [5664, 648, 413, 590, 14265, 4249, 4530, 23404],
#     "WCME": [42645, 6014, 2060, 3095, 17399, 967, 22538, 47460]
# }

# # Provided stats
# summary_stats = {
#     "Parallel-Hybrid": {"mean": 284, "std": 213},
#     "Full-Hybrid": {"mean": 6720, "std": 7595},
#     "WCME": {"mean": 17772, "std": 17341}
# }

# # Create DataFrame
# df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in means.items()]))

# # Print min/max/mean
# for series_name, series in df.items():
#     print(f"{series_name}: mean {series.mean():.1f}, min {series.min()}, max {series.max()}")

# # --- Plotting ---

# # Set up axis
# fig, ax = plt.subplots(figsize=(8, 5))

# # Custom props
# boxprops = {'linewidth': 2, 'facecolor': 'w'}
# lineprops = {'linewidth': 2}
# boxplot_kwargs = {
#     'boxprops': boxprops,
#     'medianprops': lineprops,
#     'whiskerprops': lineprops,
#     'capprops': lineprops,
#     'width': 0.75
# }

# # Plot boxplot
# sns.boxplot(
#     data=df,
#     ax=ax,
#     fliersize=0,
#     saturation=0.6,
#     log_scale=False,
#     patch_artist=True,
#     width=0.8
# )

# # Color cycle
# color_cycle = itertools.cycle(matplotlib.rcParams['axes.prop_cycle'].by_key()['color'])

# # Overlay strip plot
# ax = sns.stripplot(data=df, jitter=True, linewidth=0.5, ax=ax, edgecolor=(0, 0, 0, 0))

# # Modify box colors and lines
# face_alpha = 0.5
# for i, (patch, color) in enumerate(zip(ax.patches, color_cycle)):
#     rgba_color = (*matplotlib.colors.to_rgb(color), face_alpha)
#     patch.set_facecolor(rgba_color)
#     patch.set_edgecolor(color)
#     patch.set_linewidth(2)

#     # Each box has 6 associated Line2D objects
#     for j in range(i * 6, i * 6 + 6):
#         line = ax.lines[j]
#         line.set_color(color)
#         line.set_mfc(color)
#         line.set_mec(color)
#         line.set_alpha(0.5)
#         line.set_linewidth(1)

# Overlay provided mean ± std as red error bars
# for i, (method, stats) in enumerate(summary_stats.items()):
#     ax.errorbar(
#         x=i, y=stats["mean"], yerr=stats["std"],
#         fmt='o', color='red', capsize=5, label="Mean ± Std" if i == 0 else ""

labels = list(means.keys())
data = [means[label] for label in labels]
positions = np.arange(len(labels))

# fig, ax = plt.subplots(figsize=(8, 5))
fig = plt.figure()
ax = fig.gca()

color_list = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
face_alpha = 0.5

# Boxplot
box = ax.boxplot(
    data,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(linewidth=2),
    capprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=2),
    medianprops=dict(linewidth=2)
)

# Apply consistent colors for each label
for i, color in enumerate(color_list[:len(labels)]):
    rgba = (*matplotlib.colors.to_rgb(color), face_alpha)

    # Box face and edge
    box['boxes'][i].set_facecolor(rgba)
    box['boxes'][i].set_edgecolor(color)

    # Whiskers: 2 per box
    box['whiskers'][2*i].set_color(color)
    box['whiskers'][2*i+1].set_color(color)

    # Caps: 2 per box
    box['caps'][2*i].set_color(color)
    box['caps'][2*i+1].set_color(color)

    # Median
    box['medians'][i].set_color(color)
# # Strip plot (jittered points)
# for i, (label, values) in enumerate(means.items()):
#     x_vals = np.random.normal(i, 0.04, size=len(values))  # Jitter
#     ax.scatter(x_vals, values, edgecolor='k', linewidth=0.5, alpha=0.7, zorder=3)



# Labels and grid
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Timing (ms)")
# ax.set_title("Per-frame computation using iSAM2")
ax.set_title("Per-frame update")
ax.grid(True, alpha=0.5)
ax.set_yscale("linear")
# ax.yaxis.grid(True)
# ax.xaxis.grid(False)
# ax.legend()
# Set Y-axis to scientific notation
formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # Force scientific notation outside this range

ax.yaxis.set_major_formatter(formatter)
fig.tight_layout()

out_data_path = "/root/results/Dynosam_ecmr2024/icra_poster_timing.pdf"
fig.savefig(out_data_path)
