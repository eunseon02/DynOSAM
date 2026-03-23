#include "dynosam/frontend/vision/ClipFeatureClient.hpp"

#include <glog/logging.h>
#include <zmq.hpp>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace dyno {

ClipFeatureClient::ClipFeatureClient(const std::string& endpoint, int timeout_ms)
    : endpoint_(endpoint), timeout_ms_(timeout_ms)
{
    ctx_ = std::make_unique<zmq::context_t>(1);
    reconnect();
}

ClipFeatureClient::~ClipFeatureClient() {
    if (sock_) sock_->close();
    if (ctx_)  ctx_->close();
}

void ClipFeatureClient::reconnect() {
    try {
        sock_ = std::make_unique<zmq::socket_t>(*ctx_, zmq::socket_type::req);
        sock_->set(zmq::sockopt::sndtimeo, timeout_ms_);
        sock_->set(zmq::sockopt::rcvtimeo, timeout_ms_);
        sock_->set(zmq::sockopt::linger, 0);
        sock_->connect(endpoint_);
        VLOG(2) << "[ClipClient] Connected to " << endpoint_;
    } catch (const zmq::error_t& e) {
        LOG(WARNING) << "[ClipClient] Failed to connect: " << e.what();
        sock_.reset();
    }
}

std::vector<float> ClipFeatureClient::extractFeature(const cv::Mat& bgr_crop) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!sock_ || bgr_crop.empty()) return {};

    // Ensure continuous BGR
    cv::Mat img = bgr_crop.isContinuous() ? bgr_crop : bgr_crop.clone();
    if (img.type() != CV_8UC3) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    // Build message: [rows(4)][cols(4)][type(4)][pixel_data]
    const int rows = img.rows;
    const int cols = img.cols;
    const int cv_type = img.type();  // CV_8UC3 = 16
    const size_t pixel_bytes = static_cast<size_t>(rows) * cols * img.elemSize();
    const size_t msg_size = 12 + pixel_bytes;

    zmq::message_t request(msg_size);
    auto* ptr = static_cast<char*>(request.data());
    std::memcpy(ptr + 0, &rows, 4);
    std::memcpy(ptr + 4, &cols, 4);
    std::memcpy(ptr + 8, &cv_type, 4);
    std::memcpy(ptr + 12, img.data, pixel_bytes);

    try {
        auto send_result = sock_->send(request, zmq::send_flags::none);
        if (!send_result.has_value()) {
            connected_ = false;
            reconnect();
            return {};
        }

        zmq::message_t reply;
        auto recv_result = sock_->recv(reply, zmq::recv_flags::none);
        if (!recv_result.has_value() || reply.size() < 4) {
            connected_ = false;
            reconnect();
            return {};
        }

        // Check for error response
        if (reply.size() == 3 && std::memcmp(reply.data(), "ERR", 3) == 0) {
            connected_ = false;
            return {};
        }

        // Parse: [dim(4)][dim * float32]
        const auto* rptr = static_cast<const char*>(reply.data());
        int dim = 0;
        std::memcpy(&dim, rptr, 4);
        if (dim <= 0 || static_cast<size_t>(4 + dim * 4) > reply.size()) {
            return {};
        }

        std::vector<float> feat(dim);
        std::memcpy(feat.data(), rptr + 4, dim * sizeof(float));
        connected_ = true;
        return feat;

    } catch (const zmq::error_t& e) {
        VLOG(3) << "[ClipClient] ZMQ error: " << e.what();
        connected_ = false;
        // Recreate socket on error (REQ socket state machine may be broken)
        reconnect();
        return {};
    }
}

}  // namespace dyno
