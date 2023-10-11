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

#include "dynosam/dataprovider/DatasetLoader.hpp"
#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/utils/Macros.hpp"
#include "dynosam/utils/Tuple.hpp"


#include <filesystem>
#include <functional>

#include <glog/logging.h>
#include <iostream>
#include <fstream>

namespace dyno {

template<typename... DynoDataTypes>
struct _DynoDatasetConstructor {

    using DynoDataTypesTuple = std::tuple<DynoDataTypes...>; //! All the dyno datypes in one definition (default at the start and then the optional ones)

    virtual ~_DynoDatasetConstructor() {}
    virtual bool constructFrame(size_t frame_id, const DynoDataTypes... data) = 0;
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
class DynoDataset : public _DynoDatasetConstructor<cv::Mat, Timestamp, DataTypes...> {

public:
    using TypedGenericDataset = GenericDataset<DataTypes...>;
    using DefaultDataset = GenericDataset<cv::Mat, Timestamp>;

    using Base = _DynoDatasetConstructor<cv::Mat, Timestamp, DataTypes...>;
    using DynoDataTypesTuple = typename Base::DynoDataTypesTuple; //! All the dataypes declared in one tuple (cv::Mat, Timestamp, ....)

    //TODO: check that the type created by the _DynoDatasetConstructor is the same as the concat of the
    // DefaultDataset and TypedGenericDataset

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
    // using ConcatDataTypes = decltype(std::tuple_cat(DefaultDataset::DataTypeTuple{}, typename TypedGenericDataset::DataTypeTuple{}));

    //uses timestamp file to set the size
    DynoDataset(const fs::path& dataset_path, typename DataFolder<DataTypes>::Ptr... data_folders)
    {

        //This one will give us the number of files (frames) to expect
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

    virtual ~DynoDataset() {}

    size_t getDatasetSize() const { return dataset_size_; }

    bool processSingle(size_t frame_id) {
        if(frame_id >= dataset_size_) {
            throw std::runtime_error("Failure when processing a single frame_id!");
        }

        const size_t loading_idx = frame_id + 1; //odd indexing... oops
        CHECK(default_dataset_->load(loading_idx));
        CHECK(dataset_->load(loading_idx));

        DefaultDataset::DataTypeTuple default_data;
        typename TypedGenericDataset::DataTypeTuple extra_data;
        constexpr size_t default_data_size = std::tuple_size<DefaultDataset::DataTypeTuple>::value;
        constexpr size_t extra_data_size = std::tuple_size<typename TypedGenericDataset::DataTypeTuple>::value;

        //iterate over default dataset portion of the ConcatDataTypes
        for(size_t data_idx = 0; data_idx < default_data_size; data_idx++) {
            internal::select_apply<default_data_size>(data_idx, [&](auto stream_index) {
                internal::select_apply<default_data_size>(stream_index, [&](auto I) {
                    //I should be data_idx -> access the data vector we just loaded
                    auto per_folder_data_vector = default_dataset_->getDataVector<I>();

                    //the data idx should also correspond with the index in the DynoDataTypesTuple tuple
                    //update data
                    std::get<I>(default_data) = per_folder_data_vector.at(frame_id);

                });
            });
        }


        // //iterate over extra dataset portion of the ConcatDataTypes
        for(size_t data_idx = 0; data_idx < extra_data_size; data_idx++) {
            internal::select_apply<extra_data_size>(data_idx, [&](auto stream_index) {

                internal::select_apply<extra_data_size>(stream_index, [&](auto I) {
                    //I should be data_idx -> access the data vector we just loaded
                    auto per_folder_data_vector = dataset_->template getDataVector<I>();

                    //the data idx should also correspond with the index in the DynoDataTypesTuple tuple
                    //update data
                    std::get<I>(extra_data) = per_folder_data_vector.at(frame_id);

                });
            });
        }

        //concat the default and extra data into the full tuple
        //this is a bit annoting as we will have to copy everything over
        //get default data
        DynoDataTypesTuple loaded_data = std::tuple_cat(default_data, extra_data);


        auto construct_input_func = [&](auto&&... args) { return this->constructFrame(frame_id, args...); };
        return std::apply(construct_input_func, loaded_data);
        return true;
    }

    //TODO: set start and end idx?
    void processRange() {
        for (size_t frame_id = 0; frame_id < dataset_size_; frame_id++) {
            processSingle(frame_id);
        }
    }

private:
    typename TypedGenericDataset::UniquePtr dataset_;
    typename DefaultDataset::UniquePtr default_dataset_;

    size_t dataset_size_{0};

};

template<typename... DataTypes>
class DynoDatasetProvider :  public DynoDataset<DataTypes...>, public DataProvider  {

public:
    using BaseDynoDataset = DynoDataset<DataTypes...>;

    //does not accept DataProviderModule* for dataprovider
    DynoDatasetProvider(const fs::path& dataset_path, typename DataFolder<DataTypes>::Ptr... data_folders)
        : BaseDynoDataset(dataset_path, data_folders...), DataProvider() {}

    virtual ~DynoDatasetProvider() {}

    bool spin() override {
        const size_t dataset_size = BaseDynoDataset::getDatasetSize();

        for (size_t frame_id = 0; frame_id < dataset_size; frame_id++) {
            if(!BaseDynoDataset::processSingle(frame_id)) return false;
        }

        return true;

    }
};




} //dyno
