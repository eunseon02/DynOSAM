#!/usr/bin/env python

import os
import sys
import argparse
import argcomplete
import csv
import numpy as np

from pathlib import Path

import evaluation.filesystem_utils as eval_files

from ruamel import yaml

# # This is to avoid using tkinter
# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

# hardcoded key to search for
STATS_KEYS = ["frontend.feature_tracker", "frontend.solve_camera_motion", "frontend.solve_object_motion", "motion_solver.object_nlo_", "motion_solver.object_solve3d"]
# stats_key = "frontend"

def parser():
    basic_desc = "Plot summary of performance results for DynoSAM pipeline."
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    input_options = main_parser.add_argument_group("input options")
    input_options.add_argument(
        "--dynosam_results_path", type=str, help="Path to the folder or parent folder containing a DynoSAM results folder", required=True)
    return main_parser

def main(results_path):
    print(f"Iterating over parent results path {results_path}")
    sub_folders = [os.path.join(results_path, name) for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]

    stats_dict = {stats_key : {"mean": [], "stddev": []} for stats_key in STATS_KEYS}

    for folder in sub_folders:
        if eval_files.check_if_results_folder(folder):
            for stats_keys in STATS_KEYS:
                get_all_stats(folder, stats_dict[stats_keys], stats_keys)

    means = {}
    stddev = {}
    for stats_key, stats in stats_dict.items():
        means[stats_key] = stats["mean"]
        stddev[stats_key] = stats["stddev"]

    print(means)

    # # Create a boxplot
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    ax.boxplot(means.values())
    ax.set_xticklabels(means.keys())

    fig.tight_layout()

    plt.show()
    # ax.boxplot(data, notch=False, patch_artist=True, showmeans=True)

    # # Overlay standard deviation
    # for i, d in enumerate(data, start=1):
    #     mean = np.mean(d)
    #     std_dev = np.std(d)
    #     ax.errorbar(i, mean, yerr=std_dev, fmt='o', color='red', capsize=5)


def get_all_stats(results_folder, stats_dict, stats_key):
    # ignore the frontend-testing ones
    if "_P" in Path(results_folder).name or "_PO" in Path(results_folder).name:
        return

    print(f"Starting results folder {results_folder}")

    sub_files = [os.path.join(results_folder, name) for name in os.listdir(results_folder)]
    # from all the logging files in the results folder, get those prefixed by stats
    stats_files = list(filter(lambda file: Path(file).name.startswith("stats_") and  Path(file).name.endswith("csv"), sub_files))


    for stats_file in stats_files:


        reader = eval_files.read_csv(stats_file, ["label", "num samples", "log Hz", "mean", "stddev", "min", "max"])

        # find key in rows - bit gross ;)
        for row in reader:
            if stats_key in row["label"]:
                stats_dict["mean"].append(float(row["mean"]))
                stats_dict["stddev"].append(float(row["stddev"]))








if __name__ == "__main__":
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if main(args.dynosam_results_path):
        sys.exit(os.EX_OK)
    else:
        sys.exit(os.EX_IOERR)
