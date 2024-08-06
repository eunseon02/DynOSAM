from typing import Dict
import matplotlib.pyplot as plt

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, MultiColumn, NoEscape, Package
from pylatex.utils import italic

import evo.tools.plot as evo_plot
from cycler import cycler

import math

# Nic Barbara
def startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True):
    """Edited from https://github.com/nackjaylor/formatting_tips-tricks/
    """

    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })

    # Default settings
    plt.rcParams.update({
        "lines.linewidth": line_width,

        "axes.grid" : True,
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7",
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]),

        "errorbar.capsize": 2.5,

        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,

        "legend.framealpha": 0.7,
        "legend.edgecolor": [1,1,1],

        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=0.8*font_size)
    plt.rc('ytick', labelsize=0.8*font_size)
    plt.rc('legend', fontsize=0.8*font_size)

def format_round_4(value: float) -> float:
    return round(value, 4)

def calc_linear_average(values):
    return np.sum(np.array(values))/float(len(values))

def calc_angular_average(values):
    # angles measured in radians
    x_sum = np.sum([math.sin(x) for x in values])
    y_sum = np.sum([math.cos(x) for x in values])
    x_mean = x_sum / float(len(values))
    y_mean = y_sum / float(len(values))
    return np.arctan2(x_mean, y_mean)

class LatexTableFormatter(object):

    def __init__(self) -> None:
        geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
        self._doc = Document(geometry_options=geometry_options)
        self._doc.packages.append(Package("siunitx"))
        self._doc.packages.append(Package("mathtools"))
        self._doc.packages.append(Package("amsmath"))

        # name of collection to dictionary of results
        # results dict we get from DatasetEvaluator._run_single_analysis
        self._results_collection = {}
        # name of collection plot collection
        # plot collection we get from DatasetEvaluator._run_single_analysis
        self._plots_collection = {}

    def save_pdf(self, file_path:str):
        self._write_results_collection_to_doc()
        self._doc.generate_pdf(file_path, clean_tex=False)
        self._doc.generate_tex(file_path)

    def add_results(self, name: str, results: Dict):
        self._results_collection[name] = results

    def add_plots(self, name:str, plot_collection):
        self._results_collection[name] = plot_collection

    def _write_results_collection_to_doc(self):
        for name, results_dict in self._results_collection.items():
            # TODO: bold the best one?
            self._write_single_result_to_doc(name, results_dict)

    def _write_single_result_to_doc(self, name:str, results: Dict):
        """
        high level structure of each results should be
        {
            objects: {
                id: {
                    "poses": {
                        "ape_translation": metrics...,
                        "ape_rotation": metrics...,
                        "rpe_translation": metrics...,
                        "rpe_rotation": metrics...
                    },
                    "motions": {
                        "ape_translation": metrics...,
                        "ape_rotation": metrics...
                    }
                },
                ...
            },
            "vo: {
                "ape_translation": metrics...,
                "ape_rotation": metrics...,
                "rpe_translation": metrics...,
                "rpe_rotation": metrics...
            }
        }

        where metrics is the metrics dictionary as created by evo.Metric.get_all_statistics()
        """
        assert "objects" in results
        assert "vo" in results

        metric_object_map = results["objects"]

        mean_per_column = {
            "ape_translation":[],
            "ape_rotation": [],
            "rpe_translation": [],
            "rpe_rotation": []}

        with self._doc.create(Section(f"{name} Errors")):
            # pose errors
            with self._doc.create(Subsection('Object Pose Errors')):
                with self._doc.create(Tabular('|c|cc|cc|')) as table:
                    table.add_hline()

                    header_row = ("", MultiColumn(2, align='c', data='APE'),MultiColumn(2, align='c|', data='RPE'))
                    table.add_row(header_row)
                    table.add_row((
                        "obj",
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})"),
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
                    table.add_hline()
                    table.add_hline()

                    # populate rows with results
                    for object_id, all_metrics in metric_object_map.items():
                        if "poses" not in all_metrics:
                            print(f"Poses metrics missing from object: {object_id}")
                            continue

                        poses_metric_map = all_metrics["poses"]

                        ape_translation = format_round_4(poses_metric_map["ape_translation"]["mean"])
                        ape_rotation = format_round_4(poses_metric_map["ape_rotation"]["mean"])
                        rpe_translation = format_round_4(poses_metric_map["rpe_translation"]["mean"])
                        rpe_rotation = format_round_4(poses_metric_map["rpe_rotation"]["mean"])


                        # add to datastructure to post-calculate the mean
                        mean_per_column["ape_translation"].append(ape_translation)
                        mean_per_column["ape_rotation"].append(ape_rotation)
                        mean_per_column["rpe_translation"].append(rpe_translation)
                        mean_per_column["rpe_rotation"].append(rpe_rotation)

                        table.add_row(
                            object_id, ape_translation, ape_rotation, rpe_translation, rpe_rotation
                        )


                    # calculate mean
                    table.add_row(
                        "mean",
                        format_round_4(calc_linear_average(mean_per_column["ape_translation"])),
                        format_round_4(calc_angular_average(mean_per_column["ape_rotation"])),
                        format_round_4(calc_linear_average(mean_per_column["rpe_translation"])),
                        format_round_4(calc_angular_average(mean_per_column["rpe_rotation"])),
                    )

                    table.add_hline()

            mean_per_column = {
                "rpe_translation":[],
                "rpe_rotation": []}

            # reconstruction pose errors
            with self._doc.create(Subsection('Reconstructed Object Pose Errors')):
                with self._doc.create(Tabular('|c|cc|')) as table:
                    table.add_hline()

                    header_row = ("", MultiColumn(2, align='c', data='RPE'))
                    table.add_row(header_row)
                    table.add_row((
                        "obj",
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
                    table.add_hline()
                    table.add_hline()

                    # populate rows with results
                    for object_id, all_metrics in metric_object_map.items():
                        if "poses" not in all_metrics:
                            print(f"Poses metrics missing from object: {object_id}")
                            continue

                        poses_metric_map = all_metrics["poses"]

                        rpe_translation = format_round_4(poses_metric_map["rpe_translation_reconstruction"]["mean"])
                        rpe_rotation = format_round_4(poses_metric_map["rpe_translation_reconstruction"]["mean"])

                        # add to datastructure to post-calculate the mean
                        mean_per_column["rpe_translation"].append(rpe_translation)
                        mean_per_column["rpe_rotation"].append(rpe_rotation)

                        table.add_row(
                            object_id, rpe_translation, rpe_rotation
                        )


                    # calculate mean
                    table.add_row(
                        "mean",
                        format_round_4(calc_linear_average(mean_per_column["rpe_translation"])),
                        format_round_4(calc_angular_average(mean_per_column["rpe_rotation"]))
                    )

                    table.add_hline()


            mean_per_column = {
                "ape_translation":[],
                "ape_rotation": []}

            # motion errors
            with self._doc.create(Subsection('Object Motion Errors')):
                with self._doc.create(Tabular('|c|cc|')) as table:
                    table.add_hline()
                    header_row = ("", MultiColumn(2, align='c', data='APE'))
                    table.add_row(header_row)
                    table.add_row((
                        "obj",
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
                    table.add_hline()
                    table.add_hline()

                    # populate rows with results
                    for object_id, all_metrics in metric_object_map.items():
                        if "motions" not in all_metrics:
                            print(f"Motion metrics missing from object: {object_id}")
                            continue

                        motion_metric_map = all_metrics["motions"]

                        ape_translation = format_round_4(motion_metric_map["ape_translation"]["mean"])
                        ape_rotation = format_round_4(motion_metric_map["ape_rotation"]["mean"])

                        # add to datastructure to post-calculate the mean
                        mean_per_column["ape_translation"].append(ape_translation)
                        mean_per_column["ape_rotation"].append(ape_rotation)

                        table.add_row(
                            object_id, ape_translation, ape_rotation
                        )

                     # calculate mean
                    table.add_row(
                        "mean",
                        format_round_4(calc_linear_average(mean_per_column["ape_translation"])),
                        format_round_4(calc_angular_average(mean_per_column["ape_rotation"]))
                    )

                    table.add_hline()

            # vo errors
            with self._doc.create(Subsection('Visual Odometry Errors')):
                with self._doc.create(Tabular('|cc|cc|')) as table:
                    vo_map = results["vo"]

                    table.add_hline()
                    header_row = (MultiColumn(2, align='|c', data='APE'),MultiColumn(2, align='c|', data='RPE'))
                    table.add_row(header_row)
                    table.add_row((
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})"),
                        NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
                    table.add_hline()
                    table.add_hline()

                    # use RMSE for VO APE to match error metrics
                    ape_translation = format_round_4(vo_map["ape_translation"]["rmse"])
                    ape_rotation = format_round_4(vo_map["ape_rotation"]["rmse"])
                    rpe_translation = format_round_4(vo_map["rpe_translation"]["mean"])
                    rpe_rotation = format_round_4(vo_map["rpe_rotation"]["mean"])

                    table.add_row(
                        ape_translation, ape_rotation, rpe_translation, rpe_rotation
                    )
                    table.add_hline()

    # def _write_single_plot_to_doc(self, name: str, plot_collection: evo_plot.PlotCollection):
    #     for fig_name, fig in plot_collection.figures.items():
    #         with self._doc.create(Figure(position='htbp')) as plot:
    #             plot.add_plot(width=NoEscape(r'0.8\textwidth'), dpi=300)
    #             plot.add_caption('I am a caption.')


# def write_metric_stats_to_latex_table(results: Dict) -> str:


#     assert "objects" in results
#     assert "vo" in results

#     print("Writing metric stats!!")

#     def populate_object_pose_doc(results: Dict):

#         metric_poses_map = results["objects"]


#         geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
#         doc = Document(geometry_options=geometry_options)
#         doc.packages.append(Package("siunitx"))
#         doc.packages.append(Package("mathtools"))
#         doc.packages.append(Package("amsmath"))

#         with doc.create(Subsection('Object Pose Errors')):
#             with doc.create(Tabular('c|cc|cc|')) as table:
#                 header_row = ("", MultiColumn(2, align='c', data='APE'),MultiColumn(2, align='c|', data='RPE'))
#                 table.add_row(header_row)
#                 table.add_row((
#                     "obj",
#                     NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})"),
#                     NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
#                 table.add_hline()
#                 table.add_hline()

#                 # populate rows with results
#                 for object_id, all_metrics in metric_poses_map.items():
#                     if "poses" not in all_metrics:
#                         print(f"Poses metrics missing from object: {object_id}")
#                         continue

#                     poses_metric_map = all_metrics["poses"]

#                     ape_translation = poses_metric_map["ape_translation"]["mean"]
#                     ape_rotation = poses_metric_map["ape_rotation"]["mean"]
#                     rpe_translation = poses_metric_map["rpe_translation"]["mean"]
#                     rpe_rotation = poses_metric_map["rpe_rotation"]["mean"]

#                     table.add_row(
#                         object_id, ape_translation, ape_rotation, rpe_translation, rpe_rotation
#                     )
#                 # table.add_row(1, 1, 2, 3, 4)
#                 # table.add_row(2, 1, 2, 3, 4)
#                 table.add_hline()
#         doc.generate_pdf('full', clean_tex=False)


#     populate_object_pose_doc(results)


import numpy as np

import os

if __name__ == '__main__':

    geometry_options = {"tmargin": "1cm", "lmargin": "10cm"}
    doc = Document(geometry_options=geometry_options)
    doc.packages.append(Package("siunitx"))
    doc.packages.append(Package("mathtools"))
    doc.packages.append(Package("amsmath"))

    # doc.data.append(Package("mathtools"))
    # doc.data.append(Package("siunitx"))
    # doc.data.append(Package("amsmath"))

    # ape_t, ape_r, rpe_t, rpe_r

    with doc.create(Subsection('Table of something')):
        with doc.create(Tabular('c|cc|cc|')) as table:
            header_row = ("", MultiColumn(2, align='c', data='APE'),MultiColumn(2, align='c|', data='RPE'))
            table.add_row(header_row)
            table.add_row((
                "obj",
                NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})"),
                NoEscape(r"$E_t$(m)"),NoEscape(r"$E_r$(\si{\degree})")))
            table.add_hline()
            table.add_hline()
            table.add_row(1, 1, 2, 3, 4)
            table.add_row(2, 1, 2, 3, 4)
            table.add_hline()

    # doc.generate_pdf('full', clean_tex=False)


    # with doc.create(Section('The simple stuff')):
    #     doc.append('Some regular text and some')
    #     doc.append(italic('italic text. '))
    #     doc.append('\nAlso some crazy characters: $&#{}')
    #     with doc.create(Subsection('Math that is incorrect')):
    #         doc.append(Math(data=['2*3', '=', 9]))

    #     with doc.create(Subsection('Table of something')):
    #         with doc.create(Tabular('rc|cl')) as table:
    #             table.add_hline()
    #             table.add_row((1, 2, 3, 4))
    #             table.add_hline(1, 2)
    #             table.add_empty_row()
    #             table.add_row((4, 5, 6, 7))

    # a = np.array([[100, 10, 20]]).T
    # M = np.matrix([[2, 3, 4],
    #                [0, 0, 1],
    #                [0, 0, 2]])

    # with doc.create(Section('The fancy stuff')):
    #     with doc.create(Subsection('Correct matrix equations')):
    #         doc.append(Math(data=[Matrix(M), Matrix(a), '=', Matrix(M * a)]))

    #     with doc.create(Subsection('Alignat math environment')):
    #         with doc.create(Alignat(numbering=False, escape=False)) as agn:
    #             agn.append(r'\frac{a}{b} &= 0 \\')
    #             agn.extend([Matrix(M), Matrix(a), '&=', Matrix(M * a)])

    #     with doc.create(Subsection('Beautiful graphs')):
    #         with doc.create(TikZ()):
    #             plot_options = 'height=4cm, width=6cm, grid=major'
    #             with doc.create(Axis(options=plot_options)) as plot:
    #                 plot.append(Plot(name='model', func='-x^5 - 242'))

    #                 coordinates = [
    #                     (-4.77778, 2027.60977),
    #                     (-3.55556, 347.84069),
    #                     (-2.33333, 22.58953),
    #                     (-1.11111, -493.50066),
    #                     (0.11111, 46.66082),
    #                     (1.33333, -205.56286),
    #                     (2.55556, -341.40638),
    #                     (3.77778, -1169.24780),
    #                     (5.00000, -3269.56775),
    #                 ]

    #                 plot.append(Plot(name='estimate', coordinates=coordinates))

        # with doc.create(Subsection('Cute kitten pictures')):
        #     with doc.create(Figure(position='h!')) as kitten_pic:
        #         kitten_pic.add_image(image_filename, width='120px')
        #         kitten_pic.add_caption('Look it\'s on its back')

    doc.generate_pdf('full', clean_tex=False)
