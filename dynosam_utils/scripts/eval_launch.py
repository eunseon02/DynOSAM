#!/usr/bin/python3

#TODO: refactor to experimental yaml file like Kimera?
def parser():
    import argparse
    basic_desc = "TODO:"
    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))

    # input to the dynosam launch file and additional arguements (if any)
    # these will match the LaunchConfiguration (ie. ROS args) arguments in the dynosam launch file
    input_opts = shared_parser.add_argument_group("input options")
    evaluation_opts = shared_parser.add_argument_group("algorithm options")

    #TODO: this needs to match a number of variables in the Yaml files... streamline
    input_opts.add_argument("-d", "--dataset_path", type=str,
                                help="Absolute path to the dataset to run.",
                                default="/root/data/virtual_kitti")

    # By default not required as the dynosam_launch.py file will look for default params folder
    # in the share/ folder of the installed package
    input_opts.add_argument("-p", "--params_path", type=str,
                                help="Absolute path to the params to dun Dynosam with.",
                                required=False)

    input_opts.add_argument("-l", "--launch_file", type=str,
                                help="Which dynosam launch file to run!.",
                                default="dyno_sam_launch.py")


    evaluation_opts.add_argument("-r", "--run_pipeline", action="store_true",
                                 help="Run dyno?")

    evaluation_opts.add_argument("-o", "--output_path",
                                 help="Output folder path to store the logs in.",
                                 default="/root/results/DynoSAM/")

    evaluation_opts.add_argument("-n", "--name",
                                 help="Name of the experiment to run. This will be appended to the output_path file"
                                 " such that the output file path will be output_path/name",
                                 default="")

    evaluation_opts.add_argument("-a", "--run_analysis",
                                 help="Runs analysis on the output files, expected to be found in the output_path/name folder ",
                                 action="store_true")

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser






if __name__ == '__main__':
    from dynosam_utils.evaluation.runner import run
    import argcomplete
    import sys
    ###
    parser = parser()
    argcomplete.autocomplete(parser)
    # args will be the arguments known to the parser and should define options for how to run the
    # pipeline. unknown are cmdline args not registered to the parser and will be additional arguments that
    # the user wants to parse to the LaunchService as real args to be used as GFLAGS.
    args, unknown = parser.parse_known_args()
    args_dictionary = vars(args)

    sys.exit(run(args_dictionary, unknown))
