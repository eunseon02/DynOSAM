#!/usr/bin/env python

import os
import sys
import argparse
import argcomplete
import csv
import numpy as np

from pathlib import Path

import dynosam_utils.evaluation.filesystem_utils as eval_files
import dynosam_utils.evaluation.formatting_utils as formatting

from ruamel import yaml
import pandas as pd
import seaborn as sns
import itertools

# # This is to avoid using tkinter
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from typing import Optional

# https://cduvallet.github.io/posts/2018/03/boxplots-in-python
plt.rcdefaults()

plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    })

plt.rcParams['axes.titlesize'] = 25    # Title font size
plt.rcParams['axes.labelsize'] = 24    # X/Y label font size
plt.rcParams['xtick.labelsize'] = 19   # X tick label font size
plt.rcParams['ytick.labelsize'] = 20   # Y tick label font size
plt.rcParams['legend.fontsize']=18

def parser():
    basic_desc = "Plot summary of performance results for DynoSAM pipeline."
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    input_options = main_parser.add_argument_group("input options")
    input_options.add_argument(
        "--dynosam_results_path", type=str, help="Path to the folder or parent folder containing a DynoSAM results folder", required=True)
    return main_parser


def get_stats(results_path, stats_keys):
    print(f"Iterating over parent results path {results_path}")
    sub_folders = [os.path.join(results_path, name) for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]
    sub_folders.append(results_path)
    stats_dict = {stats_key : {"mean": [], "stddev": []} for stats_key in stats_keys.keys()}

    keys = list(stats_keys.keys())

    for folder in sub_folders:
        if eval_files.check_if_results_folder(folder):

            #  # ignore the frontend-testing ones
            # if "_P" in Path(folder).name or "_PO" in Path(folder).name:
            #     continue

            for key in keys:
                get_all_stats(folder, stats_dict[key], key)


    return stats_dict


def plot(results_path, details_list):

    num_details = len(details_list)
    fig, axes = plt.subplots(nrows=num_details, ncols=1, constrained_layout=True)

    from string import ascii_lowercase

    alphabet = iter(ascii_lowercase)

    if num_details == 1:
        axes = [axes]

    for ax, details in zip(axes, details_list):
        letter = next(alphabet)
        title = f"({letter})"

        detail_name = details.get("name", None)
        if detail_name is not None:
            title += f" {detail_name}"

        main(results_path, ax, details, title=title)

    fig.tight_layout()
    plt.show()

def main(results_path:str, ax: plt.Axes, details: dict, title: Optional[str] = None):
    stats_keys = details["keys"]
    log_scale= details.get("log_scale", False)
    stats_dict = get_stats(results_path, stats_keys)
    means = {}
    for key, stats in stats_dict.items():

        label = stats_keys[key]["label"]
        data = stats["mean"]

        # label could be in twice if we want to map a stats key to the same label
        if label in means:
            means[label].extend(data)
        else:
            means[label] = data

    # Convert the dictionary into a DataFrame with varying lengths (introducing NaNs)
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in means.items()]))
    for series_name, series in df.items():
        print(f"{series_name}: mean {series.mean()} min {series.min()} max {series.max()}")



    boxprops = { 'linewidth': 2, 'facecolor': 'w'}
    lineprops = {'linewidth': 2}
        # The boxplot kwargs get passed to matplotlib's boxplot function.
    # Note how we can re-use our lineprops dict to make sure all the lines
    # match. You could also edit each line type (e.g. whiskers, caps, etc)
    # separately.
    boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': lineprops,
                    'whiskerprops': lineprops, 'capprops': lineprops,
                    'width': 0.75})

    # boxplot = ax.boxplot(data_for_boxplot, labels=df.columns,patch_artist=True)
    # ax.set_yscale("log")
    sns.boxplot(
        data=df,
        ax=ax,
        fliersize=0,
        saturation=0.6,
        log_scale=log_scale,
        patch_artist=True,
        width=0.8)

    colour_iter = itertools.cycle(formatting.prop_cycle())
    sns.stripplot(data=df,
                jitter=True,linewidth=0.5,ax=ax,edgecolor=(0, 0, 0, 0))
    # Modify the properties of the boxes (accessing them through ax.patches)
    face_alpha = 0.5
    for i,(patch, color) in enumerate(zip(ax.patches, formatting.prop_cycle())):
        rgba_color = (*matplotlib.colors.to_rgb(color), face_alpha)
        patch.set_facecolor(rgba_color)          # Set the patch (fill) color
        patch.set_edgecolor(color)          # Set the edge (outline) color to match
        patch.set_linewidth(2)              # Set the edge width


    # This sets the color for the main box
    # artist.set_edgecolor(col)
        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*6,i*6+6):
            line = ax.lines[j]
            line.set_color(color)
            line.set_mfc(color)
            line.set_mec(color)
            line.set_alpha(0.5)
            line.set_linewidth(1)

    ax.set_ylabel(r"Timing [ms]")
    ax.grid(True, alpha=0.5)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False) # Show the vertical gridlines

    if title is not None:
        ax.set_title(title, loc="left", pad=10)



def get_all_stats(results_folder, stats_dict, stats_key):

    # print(f"Starting results folder {results_folder}")
    # print(f"Stats keys {stats_key}")

    sub_files = [os.path.join(results_folder, name) for name in os.listdir(results_folder)]
    # from all the logging files in the results folder, get those prefixed by stats
    stats_files = list(filter(lambda file: Path(file).name.startswith("stats_") and  Path(file).name.endswith("csv"), sub_files))
    read_keys = set()


    for stats_file in stats_files:


        reader = eval_files.read_csv(stats_file, ["label", "num samples", "log Hz", "mean", "stddev", "min", "max"])

        # find key in rows - bit gross ;)
        for row in reader:
            csv_key = row["label"]
            if stats_key in csv_key:
                stats_dict["mean"].append(float(row["mean"]))
                stats_dict["stddev"].append(float(row["stddev"]))

                # if csv_key not in read_keys:
                #     print(f"Using CSV key {csv_key} with requested key {stats_key}")
                #     read_keys.add(csv_key)





TRACKING_STATS_KEYS = {"name": "Feature Tracking",
                       "log_scale":False,
                       "keys":{"frontend.feature_tracker": {"label":"Feature Tracker"},
                               "tracking_timer": {"label":"Feature Tracker"},
                               "static_feature": {"label": "Static Features"},
                               "dynamic_feature": {"label": "Dynamic Features"}}
                      }


BACKEND_STATS_KEYS = {"name": "Back-end",
                       "log_scale":False,
                       "keys":{
                               "full_batch_opt": {"label":"Feature Tracker"}}
                      }

REFINEMENT_STATS_KEYS = {"name": "Motion Estimation",
                         "log_scale":True,
                       "keys":{"motion_solver.solve_3d2d": {"label":"PnP Solve"},
                               "joint_of_pose [ms]": {"label": "Optical Flow Refinement"},
                               "object_nlo_refinement [ms]": {"label": "Motion Refinement"}}
                      }

SLIDING_WINDOW_STATS = {"name": "Sliding",
                         "log_scale":False,
                       "keys":{"sliding_window_optimise ": {"label":"SW"}}
                      }

# hardcoded key to search for
# TRACKING_STATS_KEYS = ["frontend.feature_tracker", "static_feature", "dynamic_feature"]
# REFINEMENT_STATS_KEYS = ["otion_solver.solve_3d2d","joint_of_pose [ms]", "object_nlo_refinement [ms]"]
# OPT_STATS_KEYS=["batch_opt [ms]", "sliding_window_optimise [ms]"]
# OPT_STATS_KEYS=["batch_opt_num_vars", "sliding_window_optimise_num_vars"]


# INCREMENTAL_STATS = {"name": "Parallel Object Estimation",
#                          "log_scale":False,
#                        "keys":{"parallel_object_sam.optimize": {"label":"P-Opt"}}
#                       }

if __name__ == "__main__":
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if plot(args.dynosam_results_path, [SLIDING_WINDOW_STATS]):
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_IOERR)
