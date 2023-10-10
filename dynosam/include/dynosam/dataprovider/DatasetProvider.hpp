/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#pragma once

#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/utils/Macros.hpp"
#include "dynosam/utils/Tuple.hpp"
#include "dynosam/frontend/vision/Frame.hpp"

#include <gtsam/geometry/Pose3.h>
#include <opencv4/opencv2/opencv.hpp>

#include <filesystem>
#include <functional>

#include <glog/logging.h>
#include <iostream>
#include <fstream>

namespace dyno {

namespace fs = std::filesystem;


/**
 * @brief Defines a set of datafiles within a folder to be laoded by a dataset (partially inspired by pytorch's dataset class).
 * Each folder exists within a parent folder (defined by the dataset) and contains a number of files that need to be loaded by this class.
 * As a number of different input image types (motion/instance segmenation, with and without depth etc.) exist the class is
 * expected to handle the different types itself and return the loaded type
 *
 * @tparam T
 */
template<typename T = cv::Mat>
class DataFolder {

public:
    DYNO_POINTER_TYPEDEFS(DataFolder)

    using Type = T;
    DataFolder() = default;

    /**
     * @brief Set the Absolute Folder Path object to be used when an item is loaded
     * This needs to be a setter function since we do not know the absolute path at the time of construction.
     *
     * Once set, the virtual onPathInit function is called, which can act like a constructor to the derived classes
     *
     * In the context of a dataset laoder, this function is guaranteed to be called be
     *
     * @param absolute_folder_path
     */
    void setAbsoluteFolderPath(const fs::path& absolute_folder_path) {
        absolute_folder_path_ = absolute_folder_path;
        onPathInit();
    }

    virtual std::string getFolderName() const = 0;
    // virtual size_t size() const = 0;
    virtual T getItem(size_t idx) = 0;

protected:
    virtual void onPathInit() {}

protected:
    fs::path absolute_folder_path_; //! Set when init is called since we need a default constructor
};

class RGBDataFolder : public dyno::DataFolder<cv::Mat> {

public:
    RGBDataFolder() {}

    /**
     * @brief "image_0" as folder name
     *
     * @return std::string
     */
    std::string getFolderName() const override;
    cv::Mat getItem(size_t idx) override;
};

class TimestampFile : public dyno::DataFolder<double> {

public:
    TimestampFile() {}

    /**
     * @brief "times.txt" as file name
     *
     * @return std::string
     */
    std::string getFolderName() const override;
    double getItem(size_t idx) override;

    size_t size() const {
        //only valid after loading
        return times.size();
    }

private:
    /**
     * @brief Setup ifstream and read everything in the data vector
     *
     */
    void onPathInit() override;

private:
    std::vector<double> times;
};



/**
 * @brief Defines the structure, image and type of different datasets (partially inspired by pytorch's dataset class).
 * Each dataset contains a number of datafolders which defines the data for a particular input. Each datafolder is responsible for loading the file
 * and all must contain the same number of input files (or some equivalent "length") so the data can be packaged together.
 *
 * This class intends to just enable access to all the datafolders by defining all the templated tuples
 */

template<typename... DataTypes>
class DataFolderStructure {

public:
    using This = DataFolderStructure<DataTypes...>;
    using DataTypeTuple = std::tuple<DataTypes...>;
    using TypedDataFolderTuple = std::tuple<typename DataFolder<DataTypes>::Ptr...>;

    DYNO_POINTER_TYPEDEFS(This)
    static constexpr size_t N = sizeof...(DataTypes);

    template <size_t I>
    using DataType = std::tuple_element_t<I, DataTypeTuple>;

    template <size_t I>
    using DataFolderType = std::tuple_element_t<I, TypedDataFolderTuple>;

    DataFolderStructure(const fs::path& dataset_path, typename DataFolder<DataTypes>::Ptr... data_folders) : dataset_path_(dataset_path), data_folders_(data_folders...) {}

    template<size_t I>
    auto getDataFolder() const {
        return std::get<I>(data_folders_);
    }

    template<size_t I>
    fs::path getAbsoluteFolderPath() const {
        const auto data_folder = getDataFolder<I>();
        fs::path folder_path = dataset_path_;
        //technically this can also be a file path
        folder_path /= fs::path(data_folder->getFolderName());
        return fs::absolute(folder_path);
    }

    /**
     * @brief Get top level (parent) dataset path.
     *
     * All internal folders should be then found in dataset_path/folder
     *
     * @return const fs::path&
     */
    const fs::path& getDatasetPath() const { return dataset_path_; }


protected:
    const fs::path dataset_path_; //! Path to the dataset directory (ie. the parent folder)
    TypedDataFolderTuple data_folders_;

};


/**
 * @brief GenericDataset extends the DataFolderStructure to include functionality such as validation of
 * each folder and implemets the runtime loading (per index) of each datafolder, updating the internal data structures
 * containing the actual data
 *
 * @tparam DataTypes
 */
template<typename... DataTypes>
class GenericDataset : public DataFolderStructure<DataTypes...> {

public:
    using This = GenericDataset<DataTypes...>;
    using Base = DataFolderStructure<DataTypes...>;


    //! Tuple of vectors containing each datatype
    //! If DataTypes = [double, int], then this would be an alias
    //! for std::tuple<std::vector<double>, std::vector<int>>;
    using DataStorageTuple = std::tuple<std::vector<DataTypes>...>;

    template<size_t I>
    using DataStorageVector = std::vector<typename Base::DataType<I>>;

    DYNO_POINTER_TYPEDEFS(This)


    GenericDataset(const fs::path& dataset_path, typename DataFolder<DataTypes>::Ptr... data_folders)
    : Base(dataset_path, data_folders...)
    {
        //this will also set the absolute folder/file path on each DataFolder
        validateFoldersExist();
    }

    template<size_t I>
    DataStorageVector<I>& getDataVector() {
        return std::get<I>(data_);
    }

    /**
     * @brief Load the data at an index (which we index from 1 so that the same of the vector matches the max index querired)
     * This means that access the DataStorageVector for a loaded index will be index-1
     *
     * @param idx
     * @return true
     * @return false
     */
    bool load(size_t idx) {
        if(idx < 1u) {
            throw std::runtime_error("Failure when loading data in GenericDataset - loading index's start at 1!");
        }
        for (size_t i = 0; i < Base::N; i++) {
            internal::select_apply<Base::N>(i, [&](auto stream_index) {
                internal::select_apply<Base::N>(stream_index, [&](auto I) {
                    using Type = typename Base::DataType<I>;
                    auto data_folder = this->template getDataFolder<I>();
                    //loade at idx-1 since we expect all the indexing and loading to be appropiately zero idnexed
                    Type loaded_data = data_folder->getItem(idx-1);

                    auto& data_vector = this->template getDataVector<I>();

                    // add loaded data to data vector. resize if necessary but we assume in most cases data will be loaded in order
                    if(idx > data_vector.size()) {
                        data_vector.resize(idx);
                    }

                    //index the data storage vector at I-1 to account for the zero indexing
                    data_vector.at(idx-1) = loaded_data;
                });
            });
        }

        return true;//?
    }



private:
    /**
     * @brief For each datafolder, iterate through and check that the folders actually exist for each
     * std::runtime_exception is thrown if a folder is not valid.
     *
     * This function also sets absolute file path for each data folder
     *
     */
    void validateFoldersExist() {
        for (size_t i = 0; i < Base::N; i++) {
            internal::select_apply<Base::N>(i, [&](auto stream_index) {
                internal::select_apply<Base::N>(stream_index, [&](auto I) {
                    const fs::path absolute_file_path = this->template getAbsoluteFolderPath<I>();

                    if(!fs::exists(absolute_file_path)) {
                        throw std::runtime_error("Error loading datafolder - absolute file path " + std::string(absolute_file_path) + " does not exist");
                    }

                    LOG(INFO) << "Validated dataset folder: " << absolute_file_path;
                    auto data_folder = this->template getDataFolder<I>();
                    data_folder->setAbsoluteFolderPath(absolute_file_path);

                });
            });
        }
    }



private:
    DataStorageTuple data_;

};


/**
 * @brief Specalisation of the generic dataset for the expected dynosam dataset structure.
 *
 * Includes some default Datafolders which reflect the minimum requirements for the system (optical flow, rgb, camera and object poses, timestamps etc)
 * which are protected within the dataset class. Additional datafolders must be added on top of this and the class is templated on the extra data folders
 * which also inidicates the total number of datastreams to load for this dataset
 *
 * @tparam DataTypes Types that should be used to construct a frame
 */
template<typename... DataTypes>
class DynoDataset {

public:
    using TypedGenericDataset = GenericDataset<DataTypes...>;
    using DefaultDataset = GenericDataset<cv::Mat, Timestamp>;

    static constexpr size_t RGBFolderIdx = 0u; //! Index in Default Dataset
    static constexpr size_t TimestampFileIdx = RGBFolderIdx + 1u; //!


    //! The type stored at the tuple idnex of the GenericDataset
    template<size_t I>
    using GenericDatasetType = typename TypedGenericDataset::DataType<I>;

    //! The type stored at the tuple idnex of the DefaultDataset
    template<size_t I>
    using DefaultDatasetType = typename DefaultDataset::DataType<I>;

    using This = DynoDataset<DataTypes...>;

    //! Fully defined tuple type containing all the dataset types in order starting with the default types
    using ConcatDataTypes = decltype(std::tuple_cat(DefaultDataset::DataTypeTuple{}, typename TypedGenericDataset::DataTypeTuple{}));


    DynoDataset(const fs::path& dataset_path, typename DataFolder<DataTypes>::Ptr... data_folders)
    {

        //This one will give us the number of files to expect
        auto timestamps = std::make_shared<TimestampFile>();

        default_dataset_ = std::make_unique<DefaultDataset>(
            dataset_path,
            std::make_shared<RGBDataFolder>(),
            timestamps
        );

        dataset_ = std::make_unique<TypedGenericDataset>(dataset_path, data_folders...);

        dataset_size_ = timestamps->size();

        CHECK(dataset_size_ > 0);
        //TODO: error for now
        LOG(ERROR) << "DynoDataset prepared with an expected size of " << dataset_size_
            << " - loaded from timestamps file found at " << default_dataset_->getAbsoluteFolderPath<TimestampFileIdx>();

    }

    //TODO: set start and end idx?
    void process() {
        for (size_t loading_idx = 1; loading_idx <= dataset_size_; loading_idx++) {
            const size_t frame_id = loading_idx - 1;
            CHECK(default_dataset_->load(loading_idx));
            CHECK(dataset_->load(loading_idx));

            //get default data
            ConcatDataTypes loaded_data;
            constexpr size_t loadded_data_size = std::tuple_size<ConcatDataTypes>::value;

            //iterate over default dataset portion of the ConcatDataTypes
            for(size_t data_idx = 0; data_idx < loadded_data_size; data_idx++) {
                internal::select_apply<loadded_data_size>(data_idx, [&](auto stream_index) {
                    internal::select_apply<loadded_data_size>(stream_index, [&](auto I) {
                        //I should be data_idx -> access the data vector we just loaded
                        auto per_folder_data_vector = default_dataset_->getDataVector<I>();

                        //the data idx should also correspond with the index in the ConcatDataTypes tuple
                        //update data
                        std::get<I>(loaded_data) = per_folder_data_vector.at(frame_id);


                    });
                });
            }


            // auto construct_input_func = [&](auto&&... args) { return this->constructFrame(frame_id, args...); };
            // std::apply(construct_input_func, loaded_data);
            this->constructFrame(frame_id, loaded_data);

        }
    }

protected:
    virtual Frame::UniquePtr constructFrame(size_t frame_id, const ConcatDataTypes& data) {
        double timestamp = std::get<TimestampFileIdx>(data);
        LOG(ERROR) << "timestamp " << timestamp;
        return nullptr;
    };

private:
    /**
     * @brief
     *
     * @tparam I the datastream to access (ie which folder), where I is the index in the DefaultDataset tuple
     * @param idx The iteration (frame) in the dataset to access
     * @return DefaultDatasetType<I> The type to load
     */
    template<size_t I>
    DefaultDatasetType<I> loadAndGetFromDefault(size_t idx) {
        CHECK(default_dataset_->load(idx));
        return this->template getDataVector<I>().at(idx);
    }

private:
    typename TypedGenericDataset::UniquePtr dataset_;

    typename DefaultDataset::UniquePtr default_dataset_;

    size_t dataset_size_{0};

};

// /**
//  * @brief This loads data from a given source (path to sequence) using a set of given
//  * PerFolderDataLoaders which know how to load image data from a subfolder in the sequence
//  *
//  *
//  */
// class DatasetProvider : public DataProvider {

// public:
//     struct ImageLoaderDetails {
//         std::string folder_name_;
//         std::string file_suffix_;
//     };
//     //! A pair containing a ImageLoaderDetails and a function that should load an image from within the function
//     // The folder name (.first) only makes sense in the context of the DatasetProvider::path_to_dataset which should
//     // be the parent folder. The fully qualified name for the loading function is constructed from the ImageLoaderDetails
//     // and the dataset path
//     using ImageLoader = std::pair<ImageLoaderDetails, LoadImageFunction>;

//     DatasetProvider(const std::string& path_to_dataset, std::initializer_list<ImageLoader> image_loaders = {});

// protected:
//     virtual gtsam::Pose3 parsePose(const std::vector<double>& data) const = 0;

// protected:
//   const std::string path_to_sequence_;
//   const std::vector<ImageLoader> image_loaders_;

//   std::vector<gtsam::Pose3> camera_pose_gt_;
//   std::vector<Timestamp> timestamps_;

// private:
//   // should correspond with the frame_id of the data
//   size_t index_ = 0;



// };

} //dyno
