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

#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"

#include "dynosam/frontend/vision/VisionTools.hpp" //for getObjectLabels

#include <glog/logging.h>
#include <fstream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

namespace dyno {

/**
 * @brief Loading class for the Virtual Kitti 2 dataset: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
 *
 */

class GenericVirtualKittiImageLoader : public dyno::DataFolder<cv::Mat> {
public:
    GenericVirtualKittiImageLoader(const std::string& full_path) : full_path_(full_path) {}

    std::string getFolderName() const override { return ""; }
    const std::string full_path_; //! Full path as set by the full path constructor argument
};

class VirtualKittiRGBDataFolder : public GenericVirtualKittiImageLoader {

public:
    VirtualKittiRGBDataFolder(const std::string& path) : GenericVirtualKittiImageLoader(path) {}
    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << idx;
        const std::string file_path = full_path_  + "/rgb_" + ss.str() + ".jpg";
        throwExceptionIfPathInvalid(file_path);

        cv::Mat rgb;
        loadRGB(file_path, rgb);

        return rgb;
    }
};

class VirtualKittiForwardFlowDataFolder : public GenericVirtualKittiImageLoader  {

public:
    VirtualKittiForwardFlowDataFolder(const std::string& path) : GenericVirtualKittiImageLoader(path) {}

    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << idx;
        const std::string file_path = full_path_  + "/flow_" + ss.str() + ".png";
        throwExceptionIfPathInvalid(file_path);
        return vKittiPngToFlow(file_path);
    }

private:
    //as per example code in https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
    // In R, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    // In G, flow along x-axis normalized by image width and quantized to [0;2^16 – 1]
    // B = 0 for invalid flow (e.g., sky pixels)
    cv::Mat vKittiPngToFlow(const std::string& path) const {
        // cv::Mat bgr = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::Mat bgr;
        loadRGB(path, bgr);

        CHECK(!bgr.empty());

        int h = bgr.rows;
        int w = bgr.cols;
        int c = bgr.channels();

        CHECK_EQ(bgr.type(), CV_16UC3) << "type was " << utils::cvTypeToString(bgr);
        CHECK_EQ(c, 3);

        cv::Mat bgr_channels[3];
        cv::split(bgr, bgr_channels);
        cv::Mat b_channel = bgr_channels[0];

        cv::Mat bgr_float;
        //float 32
        bgr.convertTo(bgr_float, CV_32F);

        cv::Mat unscaled_out_flow = 2.0 / (std::pow(2, 16) - 1.0) * bgr_float - 1;

        cv::Mat channels[2];
        //now only take g and r channels
        cv::split(unscaled_out_flow, channels);
        cv::Mat& g = channels[1];
        cv::Mat& r = channels[2];
        g *= w -1.0;
        r *= h -1.0;

        cv::Mat out_flow;
        cv::merge(channels, 2, out_flow);

        //b == invalid flow flag == 0 for sky or other invalid flow
        cv::Mat invalid = (b_channel == 0);
        CHECK(!invalid.empty());
        out_flow.setTo(0,invalid);

        out_flow.convertTo(out_flow, CV_32F);

        return out_flow;

    }
};

class VirtualKittiDepthDataFolder : public GenericVirtualKittiImageLoader  {

public:
    VirtualKittiDepthDataFolder(const std::string& path) : GenericVirtualKittiImageLoader(path) {}

    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') <<  std::setw(5) << idx;
        const std::string file_path = full_path_  + "/depth_" + ss.str() + ".png";
        throwExceptionIfPathInvalid(file_path);

        //TODO: for now? When to apply preprocessing?
        cv::Mat depth;
        loadDepth(file_path, depth);
        return depth;
    }
};

struct VirtualKittiTimestampLoader : public TimestampBaseLoader {

    VirtualKittiTimestampLoader(size_t total_num_frames) : total_num_frames_(total_num_frames) {}

    std::string getFolderName() const override { return ""; }
    double getItem(size_t idx) override {
        //we dont have a time? for now just make idx
        return static_cast<double>(idx);
    }

    size_t size() const override {
        //only valid after loading
        //we go up to -1 becuase the optical flow has ONE LESS image
        return total_num_frames_-1u;
    }

    const size_t total_num_frames_;

};


class VirtualKittiTextGtLoader {

public:
    DYNO_POINTER_TYPEDEFS(VirtualKittiTextGtLoader)

    /**
     * @brief Construct a new Virtual Kitti Text Gt Laoder object
     *
     * File path is full file path to the textgt folder containing bbox.txt, colors.txt, ... pose.txt
     *
     * @param file_path
     */
    VirtualKittiTextGtLoader(const std::string& file_path)
    :  bbox_path(file_path + "/bbox.txt"),
       colors_path(file_path + "/colors.txt"),
       extrinsic_path(file_path + "/extrinsic.txt"),
       info_path(file_path + "/info.txt"),
       intrinsic_path(file_path + "/intrinsic.txt"),
       pose_path(file_path + "/pose.txt")  {

        throwExceptionIfPathInvalid(bbox_path);
        throwExceptionIfPathInvalid(colors_path);
        throwExceptionIfPathInvalid(extrinsic_path);
        throwExceptionIfPathInvalid(info_path);
        throwExceptionIfPathInvalid(intrinsic_path);
        throwExceptionIfPathInvalid(pose_path);

        loadBBoxMetaData(bbox_data_);
        loadPoseTxt(object_poses_, bbox_data_);
        loadCameraPoses(camera_poses_);

        gt_packets_ = constructGTPackets(object_poses_, camera_poses_);

    }

    const GroundTruthInputPacket& getGTPacket(size_t frame) const {
        return gt_packets_.at(frame);
    }

    size_t size() const {
        return gt_packets_.size();
    }

    /**
     * @brief Queries if an object is moving in the current frame
     *
     * Unlike in Virtual kitti which indicates if movement is between frame t and t+1,
     * we need to know if the object moved between t-1 and t (as we estimate for t-1 to t)
     *
     * The frame argument is considered to be frame t and we requery the isMoving variable of the BBoxMetaData at t-1.
     *
     * If object does not exist at frame return false?
     *
     * @param frame
     * @param object_id
     * @return true
     * @return false
     */
    bool isMoving(FrameId frame, ObjectId object_id) const {
        //check current frame and prev frame
        if(checkExists(frame, object_id, bbox_data_) && checkExists(frame-1, object_id, bbox_data_)) {
            return bbox_data_.at(frame - 1).at(object_id).is_moving;
        }
        return false;
    }


private:
    /**
     * @brief Nested map mapping frame id to a map of object ids to a type T
     * Allows fast access of object per frame
     *
     * @tparam T
     */
    template<typename T>
    using FrameObjectIDMap = std::map<FrameId, std::map<ObjectId, T>>;


    template<typename T>
    bool checkExists(FrameId frame_id, ObjectId object_id, const FrameObjectIDMap<T>& map) const {
        auto it = map.find(frame_id);
        if(it == map.end()) return false;

        const auto& inner_map = it->second;
        auto it_inner = inner_map.find(object_id);

        if(it_inner == inner_map.end()) return false;

        return true;
    }

    template<typename T>
    void addToFrameObjectMap(FrameId frame_id, ObjectId object_id, T t, FrameObjectIDMap<T>& map) {
        if(map.find(frame_id) == map.end()) {
            map[frame_id] = std::map<ObjectId, T>();
        }

        map[frame_id][object_id] = t;
    }

    /// @brief Storing metadata from the bbox.txt
    struct BBoxMetaData {
        FrameId frame;
        ObjectId track_id; //!to be treated as object id
        cv::Rect bbox;
        bool is_moving; //!flag to indicate whether the object is really moving between this frame and the next one
    };


private:
    void loadBBoxMetaData(FrameObjectIDMap<BBoxMetaData>& bbox_metadata) {
        std::ifstream fstream;
        fstream.open(bbox_path, std::ios::in);
        if(!fstream) {
            throw std::runtime_error("Could not open file " + bbox_path + " when trying to load BBox metadata for Virtual Kitti dataset!");
        }

        LOG(INFO) << "Loading BBox metadta for virtual kitti datasetat path: " << bbox_path;

        static const std::map<std::string, int> header_map = {
            {"frame", 0},
            {"cameraID", 1},
            {"trackID", 2},
            {"left", 3},
            {"right", 4},
            {"top", 5},
            {"bottom", 6},
            {"number_pixels", 7},
            {"truncation_ratio", 8},
            {"occupancy_ratio", 9},
            {"isMoving", 10},
        };

        auto convert_string_bool = [](const std::string& string_bool) -> bool {
            if(string_bool == "True") return true;
            else if(string_bool == "False") return false;
            else throw std::runtime_error("Unknown string bool " + string_bool + " when attempting to covnert to bool type");
        };

        bool is_first = true; //to access header
        while (!fstream.eof()) {
            std::vector<std::string> split_line;
            if(!getLine(fstream, split_line)) continue;

            CHECK_EQ(split_line.size(), header_map.size()) << "Header map and the line present in the bbox.txt file do not have the same length - " << container_to_string(split_line);

            if(is_first) {
                //check header and validate
                for(const auto& [header_value, index] : header_map) {
                    const std::string value = split_line.at(index);
                    if(value != header_value) {
                        throw std::runtime_error("Header value mismatch - " + value + " != " + header_value);
                    }
                }
                is_first = false;
            }
            else {
                const int frame =  std::stoi(split_line.at(header_map.at("frame")));
                const int camera_id =  std::stoi(split_line.at(header_map.at("cameraID")));
                const int track_id =  std::stoi(split_line.at(header_map.at("trackID")));
                const int left =  std::stoi(split_line.at(header_map.at("left")));
                const int right =  std::stoi(split_line.at(header_map.at("right")));
                const int top =  std::stoi(split_line.at(header_map.at("top")));
                const int bottom =  std::stoi(split_line.at(header_map.at("bottom")));
                // const int number_pixels =  std::stoi(split_line.at(header_map.at("number_pixels")));
                // const double truncation_ratio =  std::stod(split_line.at(header_map.at("truncation_ratio")));
                // const double occupancy_ratio =  std::stod(split_line.at(header_map.at("occupancy_ratio")));
                const bool is_moving =  convert_string_bool(split_line.at(header_map.at("isMoving")));

                if(camera_id != 0) {
                    continue; //hardcoded for only one camera atm
                }

                BBoxMetaData bbox_data;
                bbox_data.frame = frame;
                bbox_data.track_id = track_id;
                bbox_data.is_moving = is_moving;

                //convert to cv::Rect where image starts top right
                cv::Point tl(left, top);
                cv::Point br(right, bottom);
                bbox_data.bbox = cv::Rect(tl, br);

                addToFrameObjectMap(frame, track_id, bbox_data, bbox_metadata);
            }
        }
    }


    void loadPoseTxt(FrameObjectIDMap<ObjectPoseGT>& object_poses, const FrameObjectIDMap<BBoxMetaData>& bbox_metadata) {
        std::ifstream fstream;
        fstream.open(pose_path, std::ios::in);
        if(!fstream) {
            throw std::runtime_error("Could not open file " + pose_path + " when trying to load Object pose metadata for Virtual Kitti dataset!");
        }

        LOG(INFO) << "Loading Object pose metadta for virtual kitti datasetat path: " << pose_path;

        static const std::map<std::string, int> header_map = {
            {"frame", 0},
            {"cameraID", 1},
            {"trackID", 2},
            {"alpha", 3},
            {"width", 4},
            {"height", 5},
            {"length", 6},
            {"world_space_X", 7},
            {"world_space_Y", 8},
            {"world_space_Z", 9},
            {"rotation_world_space_y", 10},
            {"rotation_world_space_x", 11},
            {"rotation_world_space_z", 12},
            {"camera_space_X", 13},
            {"camera_space_Y", 14},
            {"camera_space_Z", 15},
            {"rotation_camera_space_y", 16},
            {"rotation_camera_space_x", 17},
            {"rotation_camera_space_z", 18},
        };

        bool is_first = true; //to access header
        while (!fstream.eof()) {
            std::vector<std::string> split_line;
            if(!getLine(fstream, split_line)) continue;


            CHECK_EQ(split_line.size(), header_map.size()) << "Header map and the line present in the pose.txt file do not have the same length";

             if(is_first) {
                //check header and validate
                for(const auto& [header_value, index] : header_map) {
                    CHECK_EQ(split_line.at(index), header_value);
                }
                is_first = false;
            }
            else {

                const int frame =  std::stoi(split_line.at(header_map.at("frame")));
                const int camera_id =  std::stoi(split_line.at(header_map.at("cameraID")));
                const int track_id =  std::stoi(split_line.at(header_map.at("trackID")));

                if(camera_id != 0) {
                    continue; //hardcoded for only one camera atm
                }

                // const double alpha = std::stod(split_line.at(header_map.at("alpha")));
                const double width = std::stod(split_line.at(header_map.at("width")));
                const double height = std::stod(split_line.at(header_map.at("height")));
                const double length = std::stod(split_line.at(header_map.at("length")));
                // const double world_space_X =  std::stod(split_line.at(header_map.at("world_space_X")));
                // const double world_space_Y =  std::stod(split_line.at(header_map.at("world_space_Y")));
                // const double world_space_Z =  std::stod(split_line.at(header_map.at("world_space_Z")));
                // const double rotation_world_space_y =  std::stod(split_line.at(header_map.at("rotation_world_space_y")));
                // const double rotation_world_space_x =  std::stod(split_line.at(header_map.at("rotation_world_space_x")));
                // const double rotation_world_space_z =  std::stod(split_line.at(header_map.at("rotation_world_space_z")));
                const double camera_space_X =  std::stod(split_line.at(header_map.at("camera_space_X")));
                const double camera_space_Y =  std::stod(split_line.at(header_map.at("camera_space_Y")));
                const double camera_space_Z =  std::stod(split_line.at(header_map.at("camera_space_Z")));
                const double rotation_camera_space_y =  std::stod(split_line.at(header_map.at("rotation_camera_space_y")));
                const double rotation_camera_space_x =  std::stod(split_line.at(header_map.at("rotation_camera_space_x")));
                const double rotation_camera_space_z =  std::stod(split_line.at(header_map.at("rotation_camera_space_z")));

                ObjectPoseGT object_pose_gt;
                object_pose_gt.frame_id_ = frame;
                object_pose_gt.object_id_ = track_id;
                object_pose_gt.object_dimensions_ = gtsam::Vector3(width, height, length);

                {
                    // assign r vector
                    double y = rotation_camera_space_y + (3.1415926 / 2);  // +(3.1415926/2)
                    double x = rotation_camera_space_x;
                    double z = rotation_camera_space_z;

                    // the angles are in radians.
                    double cy = cos(y);
                    double sy = sin(y);
                    double cx = cos(x);
                    double sx = sin(x);
                    double cz = cos(z);
                    double sz = sin(z);

                    double m00, m01, m02, m10, m11, m12, m20, m21, m22;

                    m00 = cy * cz + sy * sx * sz;
                    m01 = -cy * sz + sy * sx * cz;
                    m02 = sy * cx;
                    m10 = cx * sz;
                    m11 = cx * cz;
                    m12 = -sx;
                    m20 = -sy * cz + cy * sx * sz;
                    m21 = sy * sz + cy * sx * cz;
                    m22 = cy * cx;

                    gtsam::Matrix33 rot;
                    rot(0, 0) = m00;
                    rot(0, 1) = m01;
                    rot(0, 2) = m02;
                    rot(1, 0) = m10;
                    rot(1, 1) = m11;
                    rot(1, 2) = m12;
                    rot(2, 0) = m20;
                    rot(2, 1) = m21;
                    rot(2, 2) = m22;

                    gtsam::Vector3 translation(camera_space_X, camera_space_Y, camera_space_Z);

                    object_pose_gt.L_camera_ = gtsam::Pose3(gtsam::Rot3(rot), translation);
                }

                CHECK(checkExists(frame, track_id, bbox_metadata));

                //check asociated bboxdata has the right info for this object
                const BBoxMetaData& bbox_data = bbox_metadata.at(frame).at(track_id);
                CHECK_EQ(bbox_data.frame, frame);
                CHECK_EQ(bbox_data.track_id, track_id);

                object_pose_gt.bounding_box_ = bbox_data.bbox;
                addToFrameObjectMap(frame, track_id, object_pose_gt, object_poses);

            }

        }
    }

    void loadCameraPoses(std::vector<gtsam::Pose3>& camera_poses) {
        std::ifstream fstream;
        fstream.open(extrinsic_path, std::ios::in);
        if(!fstream) {
            throw std::runtime_error("Could not open file " + extrinsic_path + " when trying to load camera exterinsics for Virtual Kitti dataset!");
        }

        LOG(INFO) << "Camera pose data for virtual kitti datasetat path: " << extrinsic_path;

        bool is_first = true;

        while (!fstream.eof()) {
            std::vector<std::string> split_line;
            if(!getLine(fstream, split_line)) continue;

            //frame camera_id, 3x4 camera pose (r11, r12, r13, t1....)
            //header: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
            constexpr static size_t kExtrinsicFileLength = 18u;
            CHECK_EQ(split_line.size(), kExtrinsicFileLength);

            if(is_first){

                is_first = false;
            }
            else {
                const int frame =  std::stoi(split_line.at(0));
                const int camera_id =  std::stoi(split_line.at(1));

                if(camera_id != 0) {
                    continue; //hardcoded for only one camera atm
                }

                //sanity check to ensure things are added in order and at the right index (frame is zero indexed)
                //check only after we check the camera id
                CHECK_EQ(frame, camera_poses.size());

                const double r11 =  std::stod(split_line.at(2));
                const double r12 =  std::stod(split_line.at(3));
                const double r13 =  std::stod(split_line.at(4));
                const double t1 =   std::stod(split_line.at(5));
                const double r21 =  std::stod(split_line.at(6));
                const double r22 =  std::stod(split_line.at(7));
                const double r23 =  std::stod(split_line.at(8));
                const double t2 =   std::stod(split_line.at(9));
                const double r31 =  std::stod(split_line.at(10));
                const double r32 =  std::stod(split_line.at(11));
                const double r33 =  std::stod(split_line.at(12));
                const double t3 =   std::stod(split_line.at(13));

                gtsam::Matrix33 rot;
                rot(0, 0) = r11;
                rot(0, 1) = r12;
                rot(0, 2) = r13;
                rot(1, 0) = r21;
                rot(1, 1) = r22;
                rot(1, 2) = r23;
                rot(2, 0) = r31;
                rot(2, 1) = r32;
                rot(2, 2) = r33;

                //last 4 values should be [0 0 0 1] for homogenous matrix form so we just check
                CHECK_EQ(std::stod(split_line.at(14)), 0);
                CHECK_EQ(std::stod(split_line.at(15)), 0);
                CHECK_EQ(std::stod(split_line.at(16)), 0);
                CHECK_EQ(std::stod(split_line.at(17)), 1);


                camera_poses.push_back(gtsam::Pose3(gtsam::Rot3(rot), gtsam::Point3(t1, t2, t3)));
            }

        }

    }

    //only include objects that are moving - we will then use the returned vector as the input
    //to the mask loader to remove masks for stationary objects
    std::vector<GroundTruthInputPacket> constructGTPackets(
        const FrameObjectIDMap<ObjectPoseGT>& object_poses,
        const std::vector<gtsam::Pose3>& camera_poses
    ) const
    {

        std::vector<GroundTruthInputPacket> gt_packets;
        for(size_t frame_id = 0; frame_id < camera_poses.size(); frame_id++) {
            GroundTruthInputPacket gt_packet;
            gt_packet.frame_id_ = frame_id;
            gt_packet.X_world_ = camera_poses.at(frame_id);

            const auto& it = object_poses.find(frame_id);
            if(it != object_poses.end()) {
                const auto& object_id_pose_map = it->second;
                for(const auto& [object_id, object_pose_gt] : object_id_pose_map) {
                    //query for moving
                    // if(isMoving(frame_id, object_id)) {
                        // LOG(INFO) << "Added moving object " << frame_id << " " << object_id;
                        gt_packet.object_poses_.push_back(object_pose_gt);
                    // }
                }
            }

            gt_packets.push_back(gt_packet);
        }

        return gt_packets;
    }

    std::vector<std::string> trimAndSplit(const std::string& input) const {
        std::string trim_input = boost::algorithm::trim_right_copy(input);
        std::vector<std::string> split_line;
        boost::algorithm::split(split_line, trim_input, boost::is_any_of(" "));
        return split_line;
    }


    bool getLine(std::ifstream& fstream, std::vector<std::string>& split_lines) const {
        std::string line;
        getline(fstream, line);

        split_lines.clear();

        if(line.empty()) return false;

        split_lines = trimAndSplit(line);
        return true;
    }



private:
    const std::string bbox_path;
    const std::string colors_path;
    const std::string extrinsic_path;
    const std::string info_path;
    const std::string intrinsic_path;
    const std::string pose_path;

    FrameObjectIDMap<ObjectPoseGT> object_poses_; //! all object poses
    FrameObjectIDMap<BBoxMetaData> bbox_data_; //! all object poses

    std::vector<gtsam::Pose3> camera_poses_; //! camera poses in the world frame
    std::vector<GroundTruthInputPacket> gt_packets_; //!

};

//to be inherited either for MOTION segmentation or normal instance seg
class VirtualKittiGenericInstanceSegFolder : public GenericVirtualKittiImageLoader {
public:
    DYNO_POINTER_TYPEDEFS(VirtualKittiGenericInstanceSegFolder)

    VirtualKittiGenericInstanceSegFolder(const std::string& path, VirtualKittiTextGtLoader::Ptr gt_loader) : GenericVirtualKittiImageLoader(path), gt_loader_(CHECK_NOTNULL(gt_loader)) {}
    virtual ~VirtualKittiGenericInstanceSegFolder() = default;

    cv::Mat getItem(size_t idx) override {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(5) << idx;
        const std::string file_path = full_path_  + "/instancegt_" + ss.str() + ".png";
        throwExceptionIfPathInvalid(file_path);

        const GroundTruthInputPacket& gt_packet = gt_loader_->getGTPacket(idx);
        return loadImage(file_path, idx, gt_packet);
    }

protected:
    virtual cv::Mat loadImage(const std::string& file_path, FrameId frame, const GroundTruthInputPacket& gt_packet) = 0;

    /**
     * @brief Loads instance semantic mask from the file path. Does not change pixel values but converts to ImageType::SemanticMask::OpenCVType
     *
     * @param file_path
     * @return cv::Mat
     */
    cv::Mat loadSemanticInstanceUnchanged(const std::string& file_path) const {
        cv::Mat semantic;
        loadRGB(file_path, semantic);


        LOG(INFO) << "Semantic channels " << utils::cvTypeToString(semantic.type());

        // semantic.convertTo(semantic, ImageType::SemanticMask::OpenCVType);
        return semantic;
    }

protected:
    VirtualKittiTextGtLoader::Ptr gt_loader_;

};


//this one we will have to remove semgentation mask depending on whats in the gt folder
class VirtualKittiMotionSegFolder : public VirtualKittiGenericInstanceSegFolder {

public:
    VirtualKittiMotionSegFolder(const std::string& path, VirtualKittiTextGtLoader::Ptr gt_loader): VirtualKittiGenericInstanceSegFolder(path, gt_loader) {}

    cv::Mat loadImage(const std::string& file_path, FrameId frame, const GroundTruthInputPacket& gt_packet) override {
        //remove object in semantic mask
        cv::Mat instance_semantic_mask = loadSemanticInstanceUnchanged(file_path);

        // cv::cvtColor(instance_semantic_mask, instance_semantic_mask, cv::COLOR_RGB2GRAY);
        ObjectIds object_ids = vision_tools::getObjectLabels(instance_semantic_mask);

        std::stringstream ss;
        for(const ObjectPoseGT& object_pose_gt : gt_packet.object_poses_) {
            ss << object_pose_gt.object_id_ << " ";
        }

        CHECK_EQ(frame, gt_packet.frame_id_);

        for(int row = 0; row < instance_semantic_mask.rows; row++) {
            for(int col = 0; col < instance_semantic_mask.cols; col++) {
                // const uchar label = instance_semantic_mask.at<uchar>(row, col);
                const cv::Vec3b pixels = instance_semantic_mask.at<cv::Vec3b>(row, col);
                // LOG(INFO) << ss.str();
                // LOG(INFO) << (int)label;

                // // LOG(INFO) << container_to_string(object_ids);
                LOG(INFO) << static_cast<int>(pixels[0]) <<" " <<static_cast<int>(pixels[1]) << " " <<static_cast<int>(pixels[2]) ;

            }

        }
        //iterate over each object and if not moving remove
        for(const ObjectPoseGT& object_pose_gt : gt_packet.object_poses_) {
            const ObjectId object_id = object_pose_gt.object_id_;



            // cv::Mat idx_mask = (instance_semantic_mask == object_id+1);
            // cv::Mat roi = instance_semantic_mask(idx_mask);

            // const cv::Vec3d pixels = roi.at<cv::Vec3d>(0, 0);

            // //just a sanity (debug) check
            // const auto& it = std::find(object_ids.begin(), object_ids.end(), object_id);
            // CHECK(it != object_ids.end()) << "Object id " << object_id << " appears in gt packet but"
            //     "is not in mask when loading motion mask for Virtual Kitti at frame " << frame;

            // if(!gt_loader_->isMoving(frame, object_id)) {
            //     cv::Mat idx_mask = (instance_semantic_mask == object_id);
            //     CHECK(!idx_mask.empty());
            //     instance_semantic_mask.setTo(0,idx_mask);
            // }
        }
        return instance_semantic_mask;
    }


};

//instance segmentation for all objects regardless of motion
class VirtualKittiInstanceSegFolder : public VirtualKittiGenericInstanceSegFolder {

public:
    VirtualKittiInstanceSegFolder(const std::string& path, VirtualKittiTextGtLoader::Ptr gt_loader) : VirtualKittiGenericInstanceSegFolder(path, gt_loader) {}

    cv::Mat loadImage(const std::string& file_path, FrameId, const GroundTruthInputPacket&) override {
        return loadSemanticInstanceUnchanged(file_path);
    }

};

class VirtualKittiGTFolder : public DataFolder<GroundTruthInputPacket> {

public:
    VirtualKittiGTFolder(VirtualKittiTextGtLoader::Ptr gt_loader): gt_loader_(CHECK_NOTNULL(gt_loader)) {}

    std::string getFolderName() const override { return ""; }

    GroundTruthInputPacket getItem(size_t idx) override {
       return gt_loader_->getGTPacket(idx);
    }

    VirtualKittiTextGtLoader::Ptr gt_loader_;

};


//everything current assumes camera0!! This includes loading things like extrinsics...
VirtualKittiDataLoader::VirtualKittiDataLoader(const fs::path& dataset_path,  const std::string& scene, const std::string& scene_type, MaskType mask_type) : VirtualKittiDatasetProvider(dataset_path) {

    const std::string path = dataset_path;

    LOG(INFO) << "Starting VirtualKittiDataLoader with path " << path << " requested scene " << scene << " and scene type " << scene_type;
    const std::string depth_folder = path + "/" + v_depth_folder + "/" + scene + "/" + scene_type + "/frames/depth/Camera_0";
    const std::string forward_flow_folder = path + "/" + v_forward_flow_folder + "/" + scene + "/" + scene_type + "/frames/forwardFlow/Camera_0";
    const std::string forward_scene_flow_folder = path + "/" + v_forward_scene_flow_folder + "/" + scene + "/" + scene_type + "/frames/forwardsceneFlow/Camera_0";
    const std::string instance_segmentation_folder = path + "/" + v_instance_segmentation_folder + "/" + scene + "/" + scene_type + "/frames/instanceSegmentation/Camera_0";
    const std::string rgb_folder = path + "/" + v_rgb_folder + "/" + scene + "/" + scene_type + "/frames/rgb/Camera_0";
    const std::string text_gt_folder = path + "/" + v_text_gt_folder + "/" + scene + "/" + scene_type;

    throwExceptionIfPathInvalid(depth_folder);
    throwExceptionIfPathInvalid(forward_flow_folder);
    throwExceptionIfPathInvalid(forward_scene_flow_folder);
    throwExceptionIfPathInvalid(instance_segmentation_folder);
    throwExceptionIfPathInvalid(rgb_folder);
    throwExceptionIfPathInvalid(text_gt_folder);

    auto gt_loader = std::make_shared<VirtualKittiTextGtLoader>(text_gt_folder);

    auto timestamp_folder = std::make_shared<VirtualKittiTimestampLoader>(gt_loader->size());
    auto rgb_loader = std::make_shared<VirtualKittiRGBDataFolder>(rgb_folder);
    auto optical_flow_loader = std::make_shared<VirtualKittiForwardFlowDataFolder>(forward_flow_folder);
    auto depth_loader = std::make_shared<VirtualKittiDepthDataFolder>(depth_folder);

    VirtualKittiGenericInstanceSegFolder::Ptr motion_mask_loader = nullptr;
    if(mask_type == MaskType::MOTION) {
        LOG(INFO) << "Using MaskType::MOTION for loading mask";
        motion_mask_loader =  std::make_shared<VirtualKittiMotionSegFolder>(instance_segmentation_folder, gt_loader);
    }
    else if(mask_type == MaskType::SEMANTIC_INSTANCE) {
        LOG(INFO) << "Using MaskType::SEMANTIC_INSTANCE for loading mask";
        motion_mask_loader = std::make_shared<VirtualKittiInstanceSegFolder>(instance_segmentation_folder, gt_loader);
    }
    else {
        LOG(FATAL) << "Unknown MaskType for KittiDataLoader";
    }
    CHECK_NOTNULL(motion_mask_loader);

    auto gt_loader_folder = std::make_shared<VirtualKittiGTFolder>(gt_loader);

    this->setLoaders(
        timestamp_folder,
        rgb_loader,
        optical_flow_loader,
        depth_loader,
        motion_mask_loader,
        gt_loader_folder
    );


}

// bool VirtualKittiDataLoader::spin() {
//     return false;
// }

} //dyno
