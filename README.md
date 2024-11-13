<div align="center">
  <a href="https://robotics.sydney.edu.au/our-research/robotic-perception/">
    <img align="center" src="docs/acfr_rpg_logo.png" width="150" alt="acfr-rpg">
  </a>
</div>


# DynoSAM: Dynamic Object Smoothing and Mapping for Dynamic SLAM

DynoSAM is a Stereo/RGB-D Visual Odometry pipeline for Dynamic SLAM and estiamtes camera poses, object motions/poses as well as static background and temporal dynamic object maps.

DynoSAM current provides full-batch and sliding-window optimisation procedures and is integrated with ROS2.


## Related Publication

DynoSAM was build as a culmination of several works:

- J. Morris, Y. Wang, V. Ila.  [**The Importance of Coordinate Frames in Dynamic SLAM**](https://acfr-rpg.github.io/dynamic_slam_coordinates/). IEEE Intl. Conf. on Robotics and Automation (ICRA), 2024
- J. Zhang, M. Henein, R. Mahony, V. Ila [**VDO-SLAM:  A Visual Dynamic Object-aware SLAM System**](https://arxiv.org/abs/2005.11052), ArXiv
- M. Henein, J. Zhang, R. Mahony, V. Ila. [**Dynamic SLAM: The Need for Speed**](https://ieeexplore.ieee.org/abstract/document/9196895).2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.

# 1. Installtion

Tested on Ubuntu 20.04


## Prerequisites
- [ROS2](https://docs.ros.org/en/humble/Installation.html)
    > NOTE: this links to the Humble distribution which used during development. Other distro's will probably work.
- [GTSAM](https://github.com/borglab/gtsam) >= 4.1
- [OpenCV](https://github.com/opencv/opencv) >= 3.4
- [OpenGV](https://github.com/MIT-SPARK/opengv)
    > Note: This links to a forked version of OpenGV which contains an updated fix the CMakeLists.txt due to issues with march=native
- [Glog](http://rpg.ifi.uzh.ch/docs/glog.html), [Gflags](https://gflags.github.io/gflags/)
- [Gtest](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) (installed automagically)
- [config_utilities](https://github.com/MIT-SPARK/config_utilities)

## Dependancies
```
sudo apt install nlohmann-json3-dev libpng++-dev
```
For evaluation we also need various formatting tools
```
pip install pylatex evo &&
sudo apt-get install texlive-pictures texlive-science texlive-latex-extra latexmk
```

## Installation Instructions
DynoSAM is currently built within the ROS2 infrastructure (if there is enough interest I will split out each component into ROS and non-ROS modules.)

We provide a development [Dockerfile](./docker/Dockerfile) that will install all dependancies but expects DynoSAM to be cloned locally. The associated [container creation](./docker/create_container.sh) will then mount the local DynoSAM folder into the container along with local results/dataset folders.

> NOTE: there are some minor issues with the current dockerfile which will be fixed in-time.

The general ROS2 build procedure holds as all relevant subfolders in DynoSAM are built as packages.

# 2. Usage

## Running and Configuration

DynoSAM uses a combination of yaml files and GFLAGS (these are being simplified but GFLAGS allow an easier way to programatically set variables over cmd-line) to configure the system. ROS params are used sparingly and are only used to set input file paths.

### Running with Launch files
All .yaml and .flag files should be placed in the same params folder, which is specified by the command line.
To specify the dataset loader, the GFLAG `--data_provider_type` should be set (see [pipeline.flags](./dynosam/params/pipeline.flags)). Eventually, this will also include the option to take data live from a sensor. For data providers that are specific to each dataset used for evaluation, a dataset path must also be set.

DynoSAM will also log all output configuration to an output-folder specified by `--output_path` (see [pipeline.flags](./dynosam/params/pipeline.flags)). Data will only be logged if this folder exists.

The DynoSAM pipeline can be launched via launch file:
```
ros2 launch dynosam_ros dyno_sam_launch.py params_path:=<value> dataset_path:=<> v:=<>
```
The launch file will load all the GFLAG's from all .flag files found in the params folder.

### Running with complex input
For evaluation and more refined control over the input to the system we also provide a evaluation launch script
```
cd dynosam_utils/src
```
and can be run as
```
python3 eval_launch.py
  --dataset_path <Path to dataset>
  --params_path <Absolute path to the params folder to run dynosam with>
  --launch_file <Wich dynosam launch file to run with!>
  --output_path <Specifies the output path, overwritting the one set by pipeline.flags>
  --name <Name of the experiment to run. This will be appended to the output_path file such that the ouput file path will be output_path/name>
  --run_pipeline <if present, the full visual odometry pipeline will run>
  --run_analysis <if present, the evaluation script will run using the output found in the full output path>
  *args...

```
In addition to these arguments, this script takes all additional cmd-line arguments and parses them to the DynoSAM node, allowing any GFLAGS to be overwritten directly by specifying them in the commandline. e.g the dataset provider type can be specifued as:
```
python3 eval_launch.py --output_path=/path/to/results --name test --run_pipeline --data_provider_type=2
```
This script will also construct the corresponding output folder (e.g. ouput_path/name) and make it, if it does not exist. In the aboce example, the program will make the folder '/path/to/results/test/' and deposit all output logs in that folder.

### Running programtically
All the cmdline functionality can be replicated programtically using python in order to run experiments and evaluations.
See [run_experiments_tro.py](./dynosam_utils/src/run_experiments_tro.py) for examples.


## Datasets

We provide a number of data providers which process datasets into the input format specified by DynoSAM which includes input images for the pipeline and ground truth data for evaluation.

We are currently working on providing datasets to process live data, along-side pre-processing modules which will be available soon.

### i. KITTI Tracking Dataset
We use a modified version of the KITTI tracking dataset which includes ground truth motion data, as well dense optical-flow, depth and segmentation masks.
TODO:

The required dataset loader can be specified by setting `--data_provider_type=0`

### ii. Oxford Multimotion Dataset (OMD)
Raw data can be downloaded from the [project page](https://robotic-esp.com/datasets/omd/).
For our 2024 T-RO paper we used a modified version of the dataset which can be downloaded here. (TODO) which is associated with the current dataloader.

C++ code to parse the raw dataset (although not used) is also provided and the code used to pre-process the dataset (as optical flow and segmentation masks are not provided in the raw dataset) will also be provided soon.

The required dataset loader can be specified by setting `--data_provider_type=3`



### iii. Cluster Dataset
Access [raw dataset](https://huangjh-pub.github.io/page/clusterslam-dataset/) and download the CARLA-* sequences. No pre-processing is needed on this dataset and the raw data can be parsed by DynoSAM directly.

The required dataset loader can be specified by setting `--data_provider_type=2`

### iv. Virtual KITTI 2
Access [raw dataset](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) and extract in a folder. No pre-processing is needed on this dataset and the raw data can be parsed by DynoSAM directly.

The required dataset loader can be specified by setting `--data_provider_type=1`

# 2. Image Pre-processing
DynoSAM requires input image data in the form:
- RGB
- Depth/Stereo
- Dense Optical Flow
- Dense Semantic Instance mask

and can usually be obtained by pre-processing the input RGB image.

All datasets includes already have been pre-processed to include these images.
The code used to do this preprocessing is coming soon...


# 3. Parameters

We use a two different sets of parameters for our pipeline

- YAML files: contain params for the Pipeline and Frontend and some dataset related parameters. We use the wonderful [config_utilies](https://github.com/MIT-SPARK/config_utilities) library to load these files.
- [gflag](https://gflags.github.io/gflags/) for all other params which are found in .flag files.

Structurally DynoSAM expects all params files (.yaml and .flag) to be found in the same folder. By default we use the param files in this repo but you can specify any path as long as the folder follows the required structure:

```
params_folder:
  - FrontendParams.yaml
  - PipelineParams.yaml
 [- DatasetParams.yaml ]
 [- CameraParams.yaml ]
  - *.flags
```

The [dynosam launch file](./dynosam_ros//launch/dyno_sam_launch.py) has some smarts in it so that it will search given the folder at `params path` for any file with the ".flags" suffix and load the provided gflags.

From a design persepctive, we use gflags in addition to yaml files becuase we are able to programmatically change these params as additional arguments to the DynoSAM program. This is used extensively for automated experiments where we re-configure the pipeline for different runs.

To see all params (yaml and gflags) run the eval launch script with the gflag `--show_dyno_args=true`.

> NOTE: We have to manually print the params and there are some minor issues printing all params (e.g. we currently cannot print the camera params with the _config_utilities_ library due to the way we currently load the variables!)

To see all gflag options run:
```
ros2 run dynosam_ros dynosam_node --help
```
> NOTE: there may be some minor discrepeancies between the parameters as some values that are in the config files are then overwritten by gflag values so we can configure them easily.


## System Parameters


## Camera Parameters

Camera Parameters (ie. intrinsics) two possible sources
  - Provided by the loaded data-provider
  - Loaded from [CameraParams.yaml](./dynosam//params/CameraParams.yaml)

If the param `prefer_data_provider_camera_params` is set to True, the parameters provided by the data-loader (if available) will be used. Otherwise, the paramters from the yaml file will be used.

This allows each data-loader to specfy/loaded the camera params from the dataset itself, without having to additionally specify the intrinsics in another file.

# 4. Evaluation

## Output logs
When the DynoSAM pipeline is logged it will output log files in the given output directory. As mentioned, the output directory is specified by


# References to paper (DynoSAM: )
frame id -> k
object id -> j
tracklet id -> i

Where other things are implemented e.g. modules
