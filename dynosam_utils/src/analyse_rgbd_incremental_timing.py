import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import dynosam_utils.evaluation.evaluation_lib as eval

from dynosam_utils.evaluation.core.plotting import *


plt.rcdefaults()
startup_plotting(27, line_width=3.0)


def print_timing(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)

        timing = []
        frames = []
        last_frame_id = 0
        for r in reader:
            # awful header form
            # timing (ms), frame, opt values size, graph size
            timing.append(int(r[0]))

            frame = int(r[1])
            frames.append(frame)
            last_frame_id = max(int(r[1]), last_frame_id)

        timing = np.array(timing)
        print(f"{file} \n\tTiming mean {np.mean(timing)} \n\tMedian {np.median(timing)} \n\tStd {np.std(timing)} \n\tLast frame {last_frame_id}")

        return timing, frames, last_frame_id, np.where(timing == last_frame_id)

def load_parallel_hybrid_data(file, variable_name="timing"):
    from dynosam_utils.evaluation.tools import load_bson
    print(f"Loaded ph file {file}")
    ph_results = load_bson(file)[0]['data']

    ph_object_map = {}


    for object_id, per_frame_results in ph_results.items():
        for frame, object_isam_result in per_frame_results.items():
            was_smoother_ok = bool(object_isam_result["was_smoother_ok"])
            if not was_smoother_ok:
                continue
            frame_id = int(object_isam_result["frame_id"])
            full_isam2_result = object_isam_result["isam_result"]
            timing = float(object_isam_result["timing"])
            # collect_involved_variables(full_isam2_result, object_map, frame_id, object_id)

            if object_id not in ph_object_map:
                ph_object_map[object_id] = {
                    "frames": [],
                    "variables_reeliminated": [],
                    "variables_relinearized": [],
                    "timing": [],
                    "average_clique_size": [],
                    "max_clique_size": [],
                    "num_variables": [] ,
                    "num_landmarks_marked": [] ,
                    "num_motions_marked": []   }

            ph_object_map[object_id]["frames"].append(frame_id)
            ph_object_map[object_id]["variables_reeliminated"].append(int(full_isam2_result["variables_reeliminated"]))
            ph_object_map[object_id]["variables_relinearized"].append(int(full_isam2_result["variables_relinearized"]))
            ph_object_map[object_id]["timing"].append(timing)
            ph_object_map[object_id]["average_clique_size"].append(float(object_isam_result["average_clique_size"]))
            ph_object_map[object_id]["max_clique_size"].append(float(object_isam_result["max_clique_size"]))
            ph_object_map[object_id]["num_variables"].append(int(object_isam_result["num_variables"]))
            ph_object_map[object_id]["num_landmarks_marked"].append(int(object_isam_result["num_landmarks_marked"]))
            ph_object_map[object_id]["num_motions_marked"].append(int(object_isam_result["num_motions_marked"]))

    # collect results as per frame
    values_per_frame = {}

    static_values_per_frame = {}
    for object_id, data in ph_object_map.items():


        values = data[variable_name]
        frames = data["frames"]

        sorted_indices = sorted(range(len(frames)), key=lambda i: frames[i])
        frames = [frames[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # for object 0 (ie background) handle differently
        for frame, value in zip(frames, values):
            if object_id == "0":
                if frame not in static_values_per_frame:
                    static_values_per_frame[frame] = []
                static_values_per_frame[frame].append(value)
            else:
                if frame not in values_per_frame:
                    values_per_frame[frame] = []
                values_per_frame[frame].append(value)

    frames = []
    means = []

    for frame, values in sorted(static_values_per_frame.items()):
        assert len(values) == 1
        static_mean = values[0]

        dynamic_mean = 0

        if frame in values_per_frame:
            dynamic_mean = np.mean(np.array(values_per_frame[frame]))
            combined_timing = (static_mean + dynamic_mean)
        else:
            combined_timing = static_mean
        means.append(combined_timing)
        frames.append(frame)

    # for frame, values in sorted(values_per_frame.items()):
    #     mean = np.mean(np.array(values))
    #     means.append(np.mean(np.array(values)))


    #     frames.append(frame)

    print(f"PH {file} \n\tTiming mean {np.mean(means)} \n\tMedian {np.median(means)} \n\tStd {np.std(means)}")
    # print(f"PH Mean value {variable_name} {np.mean(np.array(means))}, std= {np.std(np.array(means))} for var={variable_name}")
    # frames, timing data per frame
    return frames, means


# esults = load_bson(f"/root/results/Dynosam_ecmr2024/{sequence}/parallel_isam2_results.bson")[0]['data']

def draw_objects_frames(ax, motion_error_evaluator = None, result_path=None, data_file_suffix = "parallel_hybrid_backend", **kwargs):
    if not motion_error_evaluator and result_path is None:
        print("Warning: Cannot plot object frames as motion evaluator and result path was none!")
        return

    if motion_error_evaluator is None and result_path:
        dataset_eval = eval.DatasetEvaluator(result_path)
        hybrid_data_files = dataset_eval.make_data_files(data_file_suffix)
        motion_error_evaluator = dataset_eval.create_motion_error_evaluator(hybrid_data_files)

    assert motion_error_evaluator is not None

    alpha = kwargs.get("alpha", 0.2)

    all_frames = motion_error_evaluator.frames

    if motion_error_evaluator is not None:
        x = sorted(all_frames)
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        # Get the number of objects at each frame
        counts = []
        for idx in x:
            objs = motion_error_evaluator.objects_at_frame(idx)
            counts.append(len(objs) if objs is not None else 0)

        max_count = max(counts)
        min_count = min(counts)

        # Define boundaries for discrete bins
        boundaries = np.arange(min_count - 0.5, max_count + 1.5, 1)

        # Get the base colormap
        base_cmap = cm.get_cmap('Greys')

        # # Define how much of the colormap range to use
        # lower_bound = 0.4  # Skip the darkest 40%
        # upper_bound = 1.0  # Use the brightest end

        # # Create a new colormap that is a subset of the original
        # cmap = mcolors.LinearSegmentedColormap.from_list(
        #     'Greys_bright',
        #     base_cmap(np.linspace(lower_bound, upper_bound, 256))
        # )

        # Base colormap (e.g., 'Greys') — reversed to make light = small
        base_cmap = plt.get_cmap('Greys')  # 'Greys_r' is reversed

        # Logarithmic scaling with more resolution in low values
        gamma = 1.2  # Controls the nonlinearity; >1 means more change for small values
        nonlinear_samples = np.linspace(0.2, 1, 256) ** (1 / gamma)

        # Create a new colormap by sampling the reversed base_cmap non-linearly
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'Greys_log_bright_small',
            base_cmap(nonlinear_samples)
        )

        cmap_with_alpha = cmap(np.linspace(0, 1, cmap.N))
        cmap_with_alpha[:, -1] = alpha  # Set alpha channel

        # Rebuild the colormap with alpha
        cmap = mcolors.ListedColormap(cmap_with_alpha)

        # Create the discrete normalization
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # Get current y-limits of the plot
        ymin, ymax = ax.get_ylim()

        # Plot the vertical bands
        for idx, count in zip(x, counts):
            if count > 0:
                color = cmap(norm(count))
                ax.fill_between(
                    [idx - 0.5, idx + 0.5], ymin, ymax,
                    color=color
                )

        # Add a colorbar with discrete ticks
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, ticks=range(min_count, max_count + 1), location='right')
        cbar.set_label('Number of Objects')



def plot_timing(ax, result_path, title, do_ph_evaluation = False, **kwargs):
    hybrid_file = result_path + "hybrid_isam2_timing_inc.csv"
    wcme_file = result_path + "wcme_isam2_timing_inc.csv"


    dataset_eval = eval.DatasetEvaluator(result_path)
    hybrid_data_files = dataset_eval.make_data_files("parallel_hybrid_backend")
    motion_error_evaluator = dataset_eval.create_motion_error_evaluator(hybrid_data_files)

    all_frames = motion_error_evaluator.frames
    last_frame = all_frames[-1]


    h_timing, h_frames, h_last_frame, h_last_index = print_timing(hybrid_file)
    wcme_timing, wcme_frames, wcme_last_frame, wcme_last_index = print_timing(wcme_file)


    ax.plot(h_frames, h_timing, label="Hybrid", color=get_nice_blue())
    ax.plot(wcme_frames ,wcme_timing, label="WCME", color=get_nice_red())

    max_timing = max(np.max(h_timing), np.max(wcme_timing))

    def draw_oom_line(ending_frame, label, line_colour, text_color='red'):
        ymin, ymax = ax.get_ylim()
        # ax.axvline(x=ending_frame, ymin=0, ymax=max_timing, color=text_color, linestyle='--', linewidth=2)
        ax.vlines(x=ending_frame, ymin=0, ymax=max_timing, color=text_color, linestyle='--', linewidth=2)
        # ax.axhline(y=max_timing, color='r', linestyle='-')


        import matplotlib.transforms as mtransforms
        trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=kwargs.get("x_failure_offset", 0.1), y=kwargs.get("y_failure_offset", -4), units='inches')


        ax.text(
            ending_frame, max_timing, f"{label} Failure",
            color=text_color, transform=trans_offset,
             bbox=dict(
                boxstyle="round,pad=0.3",  # Rounded box
                facecolor='white',         # Fill color (opaque background)
                edgecolor='red',           # Box border color
                linewidth=0,
                alpha=0.4
            )
        )


    if motion_error_evaluator:
        draw_objects_frames(ax, motion_error_evaluator=motion_error_evaluator)


    if do_ph_evaluation:
        ph_file = result_path + "parallel_isam2_results.bson"
        ph_frames, ph_timing = load_parallel_hybrid_data(ph_file)

        ax.plot(ph_frames ,ph_timing, label="Parallel-Hybrid", color=get_nice_green())


    # draw last to get on top
    if h_last_frame < last_frame:
        draw_oom_line(h_last_frame, "Hybrid", get_nice_blue())

    if wcme_last_frame < last_frame:
        draw_oom_line(wcme_last_frame, "WCME", get_nice_red())

    ax.set_yscale("log")
    # ax.margins(x=0)

    if title is not None:
        ax.set_title(title)

    ax.set_ylabel("Timing [ms]")
    ax.set_xlabel("Frame")
    ax.legend(loc="lower right")
    ax.grid(True)


fig = plt.figure(figsize=(13,6), constrained_layout=True)
ax = fig.add_subplot(111)

# sequence = "parking_lot_night_mid"
# sequence = "viode_city_night_high"
# sequence = "tas_rc6"
sequence = "kitti_0000"
# sequence = "cluster_l1"
result_path = f"/root/results/Dynosam_ecmr2024/{sequence}/"

ph_frames, ph_timing = load_parallel_hybrid_data(result_path + "parallel_isam2_results_dkp_50.bson", variable_name="num_variables")
ax.plot(ph_frames ,ph_timing, label="Kp=50")

ph_frames, ph_timing = load_parallel_hybrid_data(result_path + "parallel_isam2_results_dkp_150.bson", variable_name="num_variables")
ax.plot(ph_frames ,ph_timing, label="Kp=150")

# ph_frames, ph_timing = load_parallel_hybrid_data(result_path + "parallel_isam2_results.bson")
# ax.plot(ph_frames ,ph_timing, label="Kp=300")


# ph_frames, ph_timing = load_parallel_hybrid_data(result_path + "parallel_isam2_results_dkp_1000.bson")
# ax.plot(ph_frames ,ph_timing, label="Kp=600")

# ph_frames, ph_timing = load_parallel_hybrid_data(result_path + "parallel_isam2_results_dkp_600.bson")
# ax.plot(ph_frames ,ph_timing, label="Kp=1000")

# draw_objects_frames(ax, result_path=result_path, data_file_suffix="parallel_hybrid_dkp_600_backend", alpha=0.1)

# import matplotlib
# formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((0, 3))  # Force scientific notation outside this range

# ax.yaxis.set_major_formatter(formatter)

# ax.set_ylabel("Timing [ms]")
# ax.set_xlabel("Frame Index [-]")
# ax.legend(loc="lower right")
# ax.grid(True)

# ax.set_yscale("log")
ax.legend()

# plot_timing(ax, result_path, None, do_ph_evaluation=True, y_failure_offset=-3, x_failure_offset=0.15)

# # fig.tight_layout()
plt.show()

# fig.savefig(f"/root/results/Dynosam_ecmr2024/{sequence}_timing.pdf", format="pdf")
# fig.savefig(f"/root/results/Dynosam_ecmr2024/{sequence}_dkp_timing.pdf", format="pdf")

# print_timing(f"/root/results/Dynosam_ecmr2024/{sequence}/hybrid_isam2_timing_inc.csv")
# print_timing(f"/root/results/Dynosam_ecmr2024/{sequence}/wcme_isam2_timing_inc.csv")
