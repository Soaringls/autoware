#pragma once

#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <deque>
#include <fstream>

#include "pointmatcher_macro.h"
#include "yaml-cpp/yaml.h"

namespace autobot {
namespace cyberverse {
namespace mapping {

class AlignPointMatcher {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  typedef std::shared_ptr<AlignPointMatcher> Ptr;
  typedef std::shared_ptr<const AlignPointMatcher> ConstPtr;

 public:
  AlignPointMatcher() { icp.setDefault(); };

  explicit AlignPointMatcher(const std::string& yaml_file);

  ~AlignPointMatcher() = default;

  void SetIcpSetupByYamlFile(const std::string& yaml_file);

  bool Align(const DP& reference_cloud, const DP& reading_cloud,
             Eigen::Matrix4d& trans_matrix, double& score,
             std::string info = "",
             const Eigen::Matrix4d& init_tf = Eigen::Matrix4d::Identity());

  template <typename PointT>
  bool Align(const typename pcl::PointCloud<PointT>::ConstPtr reference_cloud,
             const typename pcl::PointCloud<PointT>::ConstPtr reading_cloud,
             Eigen::Matrix4d& trans_matrix, double& score);

  DP GetAlignDP() const { return align_pts_; }

 private:
  bool Init(const std::string& config_file);

  template <typename PointT>
  void ConverToDataPoints(
      const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP& cur_pts);

  void CheckInitTransformation(const Eigen::Matrix4d& init_matrix,
                               PM::TransformationParameters& init_trans);

  void CalRobustScore(const DP& ref, const DP& reading_pts,
                      const PM::TransformationParameters& T, double& score);
  void CalHaussdorffScore(const DP& ref, const DP& reading,
                          const PM::TransformationParameters& T, double& score);
  void CalOdomRobustScore(double& score, std::string info);

 private:
  PM::ICP icp;
  // check init_matrix
  std::shared_ptr<PM::Transformation> rigid_trans_;

  // evaluate score:haussdorff distance
  std::shared_ptr<PM::Matcher> matcher_hausdorff_;

  DP align_pts_;
  DP tf_reading_pts_;
};

}  // namespace mapping
}  // namespace cyberverse
}  // namespace autobot