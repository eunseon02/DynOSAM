#include "dynosam/backend/edge_map/ElementEdge.hpp"

using namespace dyno;
// 静态counter 初始化
unsigned int elementEdge::id_counter = 0;

std::mt19937 elementEdgeCluster::rng;
std::uniform_int_distribution<unsigned char> elementEdgeCluster::dist(0, 255);