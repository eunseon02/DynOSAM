/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/dataprovider/OMDDataProvider.hpp"

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <glog/logging.h>

namespace dyno {

class OMDAllLoader {

public:
    DYNO_POINTER_TYPEDEFS(OMDAllLoader)

    OMDAllLoader(const std::string& file_path)
    :   rgbd_folder_path_(file_path + "/rgbd"),
        instance_masks_folder_(file_path + "/instance_masks"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        vicon_file_path_(file_path + "/vicon.csv")
    {
        throwExceptionIfPathInvalid(rgbd_folder_path_);
        throwExceptionIfPathInvalid(instance_masks_folder_);
        throwExceptionIfPathInvalid(optical_flow_folder_path_);
        throwExceptionIfPathInvalid(vicon_file_path_);

        //first load images and size
        //the size will be used as a refernce for all other loaders
        //size is number of images
        //we use flow to set the size as for this dataset, there will be one less
        //optical flow as the flow ids index t to t+1 (which is gross, I know!!)
        //TODO: code and comments replicated in ClusterSlamDataProvider
        loadFlowImagesAndSize(optical_flow_image_paths_, dataset_size_);

        //load rgb and aligned depth image file paths into rgb_image_paths_ and aligned_depth_image_paths_
        //does some sanity checks
        //as well as timestamps
        loadRGBDImagesAndTime();
    }

private:
    void loadFlowImagesAndSize(std::vector<std::string>& images_paths, size_t& dataset_size) {
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(optical_flow_folder_path_);
        dataset_size = files_in_directory.size();
        CHECK_GT(dataset_size, 0);

        for (const std::string file_path : files_in_directory) {
            throwExceptionIfPathInvalid(file_path);
            images_paths.push_back(file_path);
        }
   }

   void loadRGBDImagesAndTime() {
        // from dataset contains *_aligned_depth.png and *_color.png
        // need to construct both
        auto image_files = getAllFilesInDir(rgbd_folder_path_);

        const auto colour_substring = "color.png";
        const auto depth_substring = "aligned_depth.png";

        for(const std::string file : image_files) {
            if(file.find(colour_substring) != std::string::npos) {
                //is rgb image
                rgb_image_paths_.push_back(file);
                LOG(INFO) << "RGB file " << file;
            }
            else if (file.find(depth_substring) != std::string::npos) {
                //is depth image
                aligned_depth_image_paths_.push_back(file);
            }
            //special case - rgbd.csv file is inside this folder
            //and gives timestamp per frame number
            else if (file.find("rgbd.csv") != std::string::npos) {

            }
            else {
                LOG(FATAL) << "Could not load rgbd image at path " << file;
            }
        }

   }

   void loadTimestampsFromRGBDCsvFile(const std::string file_path) {
    //exepect a csv file with the header [frame_num, time_sec, time_nsec]
    //inside the rgbd folder
    //TODO: for now dont use a csv reader!!
   }


private:
    const std::string rgbd_folder_path_;
    const std::string instance_masks_folder_;
    const std::string optical_flow_folder_path_;
    const std::string vicon_file_path_;

    std::vector<std::string> rgb_image_paths_; //index from 0 to match the naming convention of the dataset
    std::vector<std::string> aligned_depth_image_paths_;
    std::vector<Timestamp> times_; //loaded from rgbd.csv file and associated with rgbd images. Should be the same length as rgb/aligned depth files

    std::vector<std::string> optical_flow_image_paths_;
    std::vector<std::string> instance_masks_image_paths_;


    size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number of files in the /optical_flow folder which is one per frame

};


OMDDataLoader::OMDDataLoader(const fs::path& dataset_path) : OMDDatasetProvider(dataset_path) {
    LOG(INFO) << "Starting OMDDataLoader with path" << dataset_path;

    //this would go out of scope but we capture it in the functional loaders
    auto loader = std::make_shared<OMDAllLoader>(dataset_path);
}

} //dyno
