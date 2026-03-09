#include "dynosam/backend/edge_map/ElementEdge.hpp"
#include <opencv2/core/core.hpp>
#include <functional>

using namespace dyno;
// 静态counter 初始化
unsigned int elementEdge::id_counter = 0;

std::mt19937 elementEdgeCluster::rng;
std::uniform_int_distribution<unsigned char> elementEdgeCluster::dist(0, 255);

// Generate deterministic color from cluster_id
// This ensures the same cluster_id always gets the same color, even after sliding window updates
// Uses HSV color space to ensure different cluster_ids get visually distinct colors
cv::Vec3b elementEdgeCluster::generateColorFromClusterId(unsigned int cluster_id) {
    // Use HSV color space for better color distribution
    // Hue: distributed evenly based on cluster_id (0-360 degrees)
    // Saturation: fixed at high value (0.8) for vibrant colors
    // Value: fixed at high value (0.9) for bright colors
    
    // Map cluster_id to hue (0-360 degrees)
    // Use modulo to wrap around, ensuring different IDs get different hues
    double hue = static_cast<double>(cluster_id % 360);
    
    // Fixed saturation and value for consistent appearance
    double saturation = 0.8;
    double value = 0.9;
    
    // Convert HSV to RGB
    double c = value * saturation;
    double x = c * (1.0 - std::abs(std::fmod(hue / 60.0, 2.0) - 1.0));
    double m = value - c;
    
    double r_d, g_d, b_d;
    if (hue < 60) {
        r_d = c; g_d = x; b_d = 0;
    } else if (hue < 120) {
        r_d = x; g_d = c; b_d = 0;
    } else if (hue < 180) {
        r_d = 0; g_d = c; b_d = x;
    } else if (hue < 240) {
        r_d = 0; g_d = x; b_d = c;
    } else if (hue < 300) {
        r_d = x; g_d = 0; b_d = c;
    } else {
        r_d = c; g_d = 0; b_d = x;
    }
    
    unsigned char r = static_cast<unsigned char>((r_d + m) * 255);
    unsigned char g = static_cast<unsigned char>((g_d + m) * 255);
    unsigned char b = static_cast<unsigned char>((b_d + m) * 255);
    
    return cv::Vec3b(b, g, r); // OpenCV uses BGR format
}