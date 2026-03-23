#ifndef CLIP_FEATURE_CLIENT_HPP
#define CLIP_FEATURE_CLIENT_HPP

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>

// Forward declare zmq types to avoid exposing zmq.hpp in header
namespace zmq { class context_t; class socket_t; }

namespace dyno {

/**
 * @brief ZMQ client that sends bbox-cropped images to the Python CLIP server
 *        and receives L2-normalized feature vectors.
 *
 * Thread-safe: multiple threads can call extractFeature() concurrently.
 * If the server is not running, extractFeature() returns an empty vector
 * (non-blocking after timeout).
 */
class ClipFeatureClient {
public:
    /**
     * @param endpoint  ZMQ endpoint, e.g. "tcp://localhost:5555"
     * @param timeout_ms  Send/recv timeout in milliseconds (0 = infinite)
     */
    explicit ClipFeatureClient(const std::string& endpoint = "tcp://localhost:5555",
                               int timeout_ms = 200);
    ~ClipFeatureClient();

    /**
     * @brief Extract CLIP feature from a BGR image crop.
     * @param bgr_crop  CV_8UC3 image (bbox crop of the detection).
     * @return L2-normalized feature vector (e.g. 512-d).  Empty if server unavailable.
     */
    std::vector<float> extractFeature(const cv::Mat& bgr_crop);

    /// Check if the client is connected (last request succeeded).
    bool isConnected() const { return connected_; }

private:
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t>  sock_;
    std::mutex mutex_;
    bool connected_ = false;
    std::string endpoint_;
    int timeout_ms_;

    void reconnect();
};

}  // namespace dyno

#endif  // CLIP_FEATURE_CLIENT_HPP
