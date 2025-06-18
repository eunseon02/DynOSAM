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

def parser():
    basic_desc = "Plot summary of performance results for DynoSAM pipeline."
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    input_options = main_parser.add_argument_group("input options")
    input_options.add_argument(
        "--dynosam_results_path", type=str, help="Path to the folder or parent folder containing a DynoSAM results folder", required=True)
    return main_parser


def get_stats(results_path, stats_keys, result_callback = None):
    print(f"Iterating over parent results path {results_path}")
    sub_folders = [os.path.join(results_path, name) for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]
    sub_folders.append(results_path)
    stats_dict = {stats_key : {"mean": [], "stddev": []} for stats_key in stats_keys.keys()}

    keys = list(stats_keys.keys())


    for folder in sub_folders:
        if eval_files.check_if_results_folder(folder):


            for key in keys:
                get_all_stats(folder, stats_dict[key], key, callback=result_callback)


    return stats_dict

def get_all_stats(results_folder, stats_dict, stats_key, callback=None):

    # ignore = ["viode", "zed", "incremental", "vdo"]
    # if ignore in results_folder:
    #     return

    # # if

    print(f"Starting results folder {results_folder}")
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
            if stats_key in csv_key and callback:
                    mean = float(row["mean"])
                    num_samples = float(row["num samples"])
                    callback(results_folder, stats_key, mean, num_samples)
                # stats_dict["mean"].append(float(row["mean"]))
                # stats_dict["stddev"].append(float(row["stddev"]))

                # if csv_key not in read_keys:
                #     print(f"Using CSV key {csv_key} with requested key {stats_key}")
                #     read_keys.add(csv_key)
def plot(results_path, details, result_callback=None):

    stats_keys = details["keys"]
    log_scale = details.get("log_scale", False)
    stats_dict = get_stats(results_path, stats_keys, result_callback=result_callback)

    print(stats_dict)
    return 1


SLIDING_WINDOW_STATS = {"name": "Sliding",
                        "log_scale":False,
                        "keys":{"sliding_window_optimise ": {"label":"SW"}}
                      }

FB_WINDOW_STATS = {"name": "Sliding",
                        "log_scale":False,
                        "keys":{"full_batch_opt ": {"label":"SW"}}
                      }

sw_averages = []
fb_averages = []

def sw_accumulate(results_folder, stats_key, mean, num_samples):
        print(mean)
        print(num_samples)
        average_by_samples = mean
        sw_averages.append(average_by_samples)

def fb_accumulate(results_folder, stats_key, mean, num_samples):
        print(mean)
        print(num_samples)
        average_by_samples = mean * num_samples
        fb_averages.append(mean)

if __name__ == "__main__":
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    plot(args.dynosam_results_path, FB_WINDOW_STATS, fb_accumulate)
    plot(args.dynosam_results_path, SLIDING_WINDOW_STATS, sw_accumulate)

    print(sw_averages)
    sw_average = np.mean(np.array(sw_averages))
    sw_std = np.std(np.array(sw_averages))
    print(f"SW average: {sw_average}, std {sw_std}")

    print(fb_averages)
    fb_average = np.mean(np.array(fb_averages))
    fb_std = np.std(np.array(fb_averages))
    print(f"FB average: {fb_average}  std {fb_std}")

    #     sys.exit(os.EX_OK)
    # else:
    #     sys.exit(os.EX_IOERR)
