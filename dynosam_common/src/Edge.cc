#include "dynosam_common/Edge.hpp"
#include <stdexcept>

Edge::Edge(int ID, std::vector<orderedEdgePoint> list){
    edge_ID = ID;
    for(int i = 0; i < list.size(); ++i){
        list[i].frame_edge_ID = ID;
        this->push_back(list[i]);
    }
}

void Edge::push_back(orderedEdgePoint pt){
    mvPoints.push_back(pt);
}

void Edge::samplingEdgeUniform(int bias){
    mvSampledEdgeIndex.clear();

    //-- 进行采样
    for(int i = 0; i < mvPoints.size(); ++i){
        if(i % bias == 0){
            mvSampledEdgeIndex.push_back(i);
        }
    }
}

bool Edge::isAssociatedWith(Edge query_edge){
    return true;
}

//-- 根据所有边缘点的可见性分数计算边缘本身的可见性分数
void Edge::calcEdgeScoreViz()
{
    //-- 由于大量随机噪声的存在，用中位数比较合适
    std::vector<double> score;
    double total_score;
    for(int i = 0; i < mvPoints.size(); ++i){
        score.push_back(mvPoints[i].score_visible);
        total_score += mvPoints[i].score_visible;
    }
    std::sort(score.begin(), score.end());
    double medianValue;
    if(score.size()%2==0){
        medianValue = 1.0/2.0 * (score[score.size()/2] + score[score.size()/2 - 1]);
    }else{
        medianValue = score[score.size()/2];
    }
    //-- 平均值作为score
    //-- mVisScore = total_score / mvPoints.size();
    //-- 中位值作为score
    mVisScore = medianValue;
}

// EdgeContainer implementation
EdgeContainer::EdgeContainer() : edge_map_() {}

EdgeContainer::EdgeContainer(const std::vector<Edge>& edges) {
  for (const auto& edge : edges) {
    add(edge);
  }
}

void EdgeContainer::add(const Edge& edge) {
  if (exists(edge.edge_ID)) {
    throw std::runtime_error("Edge with ID " + std::to_string(edge.edge_ID) +
                             " already exists in container");
  }
  edge_map_[edge.edge_ID] = edge;
}

void EdgeContainer::remove(EdgeId edge_id) {
  if (!exists(edge_id)) {
    throw std::runtime_error("Cannot remove edge with ID " +
                             std::to_string(edge_id) +
                             " as edge does not exist!");
  }
  edge_map_.erase(edge_id);
}

void EdgeContainer::clear() {
  edge_map_.clear();
}

size_t EdgeContainer::size() const {
  return edge_map_.size();
}

Edge* EdgeContainer::getByEdgeId(EdgeId edge_id) {
  auto it = edge_map_.find(edge_id);
  if (it == edge_map_.end()) {
    return nullptr;
  }
  return &(it->second);
}

const Edge* EdgeContainer::getByEdgeId(EdgeId edge_id) const {
  auto it = edge_map_.find(edge_id);
  if (it == edge_map_.end()) {
    return nullptr;
  }
  return &(it->second);
}

bool EdgeContainer::exists(EdgeId edge_id) const {
  return edge_map_.find(edge_id) != edge_map_.end();
}

EdgeIds EdgeContainer::collectEdgeIds() const {
  EdgeIds edge_ids;
  edge_ids.reserve(edge_map_.size());
  for (const auto& [edge_id, edge] : edge_map_) {
    edge_ids.push_back(edge_id);
  }
  return edge_ids;
}

std::vector<cv::Point2f> EdgeContainer::toOpenCV(EdgeIds* edge_ids) const {
  if (edge_ids) edge_ids->clear();

  std::vector<cv::Point2f> points;
  for (const auto& [edge_id, edge] : edge_map_) {
    for (const auto& pt : edge.mvPoints) {
      points.push_back(cv::Point2f(static_cast<float>(pt.x),
                                   static_cast<float>(pt.y)));
      if (edge_ids) {
        edge_ids->push_back(edge_id);
      }
    }
  }
  return points;
}

std::vector<std::vector<cv::Point2f>> EdgeContainer::toOpenCVByEdge() const {
  std::vector<std::vector<cv::Point2f>> edges_points;
  edges_points.reserve(edge_map_.size());

  for (const auto& [edge_id, edge] : edge_map_) {
    std::vector<cv::Point2f> edge_points;
    edge_points.reserve(edge.mvPoints.size());
    for (const auto& pt : edge.mvPoints) {
      edge_points.push_back(cv::Point2f(static_cast<float>(pt.x),
                                        static_cast<float>(pt.y)));
    }
    edges_points.push_back(edge_points);
  }
  return edges_points;
}