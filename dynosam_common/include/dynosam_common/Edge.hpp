#ifndef EDGE_H
#define EDGE_H

#include <iostream>
#include <fstream>
#include <queue>
#include <sys/types.h>
#include <dirent.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <chrono>

/**
 * @brief 有序的边缘点，是直接参与到slam过程中的边缘点类型
 * @details 不再存储与相邻有关的信息，仅存储当前位置的图像属性与结构属性
*/
class orderedEdgePoint{
public:
    //########################### 
    //--      2D相关成员
    //########################### 

    //-- 像素坐标
    double x;
    double y;
    //-- RGBD给的深度
    float depth;
    //-- 梯度角度
    float imgGradAngle;

    //########################### 
    //--      3D相关成员
    //########################### 
    double x_3d;
    double y_3d;
    double z_3d;
    float score_depth;
    float score_visible;
    

    //########################### 
    //--      索引相关成员
    //###########################
    int frame_edge_ID;
    int frame_point_index;

    //########################### 
    //--      关联相关成员
    //###########################
    bool mbAssociated;
    //-- depth continuity flag: true if this point is part of a depth-continuous
    //   segment (set by Frame::edgeCullingContinuity after depth assignment).
    bool is_depth_continuous = true;

    //-- 半径内搜索得到的最近邻列表
    std::vector<int> mvAssoFrameEdgeIDs;
    std::vector<int> mvAssoFramePointIndices;
    int asso_edge_ID; //-- 与当前点关联的参考帧点的edge ID
    int asso_point_index; //-- 与当前点关联的参考帧点的point index

    orderedEdgePoint(double _x, double _y, float _imgGradAngle){
        x = _x;
        y = _y;
        imgGradAngle = _imgGradAngle;
        mbAssociated = false;
    }
};

/**
 * @brief 自组织之后的有序边缘，存储有序的orderedEdgePoint, 其索引就代表其顺序
 * @details 本类直接用于slam过程中，直接参与特征的匹配、关联以及配准优化
*/
class Edge{
public:
    int edge_ID;

    //-- 每个边缘的点列表，可以用push_back进行操作
    std::vector<orderedEdgePoint> mvPoints;
    float mVisScore;

    //-- 采样之后的索引，在mvPoints中检索，与edgeCloud一一对应
    std::vector<int> mvSampledEdgeIndex;

  // Optional per-edge color for visualization (typically set from Object::GetColor()).
  // Defaults to a neutral gray until associated with an object.
  cv::Scalar color = cv::Scalar(150, 150, 150);

  // Optional object association (e.g., from detection / segmentation)
  // -1 means no associated object.
  int object_id = -1;

    void samplingEdgeUniform(int bias);

    Edge(){}

    Edge(int ID){ edge_ID = ID;}

    Edge(int ID, std::vector<orderedEdgePoint> list);

    void push_back(orderedEdgePoint pt);

    //-- 检查两个边缘是否可以建立关联，如果可以建立关联则返回true, 不行则返回false
    bool isAssociatedWith(Edge query_edge);

    void calcEdgeScoreViz();

};

/**
 * @brief Edge ID type for EdgeContainer
 */
using EdgeId = int;
using EdgeIds = std::vector<EdgeId>;

/**
 * @brief Container for managing Edge objects by edge_ID
 * 
 * Similar to FeatureContainer, but designed specifically for Edge objects.
 * Provides efficient lookup by edge_ID and convenient iteration.
 */
class EdgeContainer {
 public:
  using EdgeIdToEdgeMap = std::unordered_map<EdgeId, Edge>;

  /**
   * @brief Internal iterator type allowing iteration over the edges directly
   * e.g for(Edge& edge : container)
   */
  template <typename MapIterator>
  struct vector_iterator_base {
    using iterator_type = MapIterator;

    using value_type = Edge;
    using reference = Edge&;
    using pointer = Edge*;

    iterator_type it_;
    vector_iterator_base(iterator_type it) : it_(it) {}

    reference operator*() { return it_->second; }
    pointer operator->() { return &(it_->second); }

    bool operator==(const vector_iterator_base& other) const {
      return it_ == other.it_;
    }
    bool operator!=(const vector_iterator_base& other) const {
      return it_ != other.it_;
    }

    bool operator==(const iterator_type& other) const { return it_ == other; }
    bool operator!=(const iterator_type& other) const { return it_ != other; }

    vector_iterator_base& operator++() {
      ++it_;
      return *this;
    }
  };

  /// @brief Vector-style iterator definition
  using vector_iterator =
      vector_iterator_base<EdgeIdToEdgeMap::iterator>;
  /// @brief Vector-style const iterator definition
  using const_vector_iterator =
      vector_iterator_base<EdgeIdToEdgeMap::const_iterator>;

  /// @brief Internal typedefs to allow EdgeContainer to satisfy the
  /// definitions of a std::iterator
  using iterator = vector_iterator;
  using pointer = Edge*;
  using const_iterator = const_vector_iterator;
  using const_pointer = const Edge*;
  using value_type = Edge;
  using reference = Edge&;
  using const_reference = const Edge&;
  using difference_type = std::ptrdiff_t;

  EdgeContainer();
  EdgeContainer(const std::vector<Edge>& edges);

  /**
   * @brief Adds a new edge to the container.
   * Uses edge.edge_ID to set the edge key.
   *
   * @param edge const Edge&
   */
  void add(const Edge& edge);

  /**
   * @brief Removes an edge by edge_ID
   *
   * @param edge_id EdgeId
   */
  void remove(EdgeId edge_id);

  /**
   * @brief Clears the entire container
   */
  void clear();

  /**
   * @brief If the container is empty.
   *
   * @return true
   * @return false
   */
  inline bool empty() const { return size() == 0u; }

  /**
   * @brief Returns the number of edges in the container.
   *
   * @return size_t
   */
  size_t size() const;

  /**
   * @brief Gets an edge given its edge_ID.
   * If the edge does not exist, returns nullptr.
   *
   * @param edge_id EdgeId
   * @return Edge* (nullptr if not found)
   */
  Edge* getByEdgeId(EdgeId edge_id);

  /**
   * @brief Gets an edge given its edge_ID (const version).
   * If the edge does not exist, returns nullptr.
   *
   * @param edge_id EdgeId
   * @return const Edge* (nullptr if not found)
   */
  const Edge* getByEdgeId(EdgeId edge_id) const;

  /**
   * @brief Returns true if an edge with the given edge_ID exists.
   *
   * @param edge_id EdgeId
   * @return true
   * @return false
   */
  bool exists(EdgeId edge_id) const;

  /**
   * @brief Collects all edge IDs in the container.
   *
   * @return EdgeIds
   */
  EdgeIds collectEdgeIds() const;

  EdgeContainer& operator+=(const EdgeContainer& other) {
    edge_map_.insert(other.edge_map_.begin(), other.edge_map_.end());
    return *this;
  }

  // vector begin
  inline vector_iterator begin() {
    return vector_iterator(edge_map_.begin());
  }
  inline const_vector_iterator begin() const {
    return const_vector_iterator(edge_map_.cbegin());
  }

  // vector end
  inline vector_iterator end() { return vector_iterator(edge_map_.end()); }
  inline const_vector_iterator end() const {
    return const_vector_iterator(edge_map_.cend());
  }

  /**
   * @brief Converts all edge points to OpenCV Point2f representation.
   * This makes them compatible with OpenCV functions.
   *
   * @param edge_ids EdgeIds*. If provided, will be filled with edge IDs
   * @return std::vector<cv::Point2f> All points from all edges
   */
  std::vector<cv::Point2f> toOpenCV(EdgeIds* edge_ids = nullptr) const;

  /**
   * @brief Converts edges to OpenCV format, organized by edge.
   * Each edge's points are stored as a separate vector.
   *
   * @return std::vector<std::vector<cv::Point2f>> Vector of edge point vectors
   */
  std::vector<std::vector<cv::Point2f>> toOpenCVByEdge() const;

 private:
  EdgeIdToEdgeMap edge_map_;
};

#endif