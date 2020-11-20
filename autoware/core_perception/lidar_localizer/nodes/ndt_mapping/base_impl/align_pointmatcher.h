#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pointmatcher/Parametrizable.h>
#include <pointmatcher/PointMatcher.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <deque>
#include <fstream>

// #include "ranger_base/util.hpp"
// #include "ranger_base/utility.hpp"
#include "yaml-cpp/yaml.h"

namespace common {
const float DEG_TO_RAD = M_PI / 180.0;
const float RAD_TO_DEG = 180 / M_PI;

const double DEG_TO_RAD_D = M_PI / 180.0;
const double RAD_TO_DEG_D = 180 / M_PI;

}  // namespace common

class AlignPointMatcher {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  typedef std::shared_ptr<AlignPointMatcher> Ptr;
  typedef std::shared_ptr<const AlignPointMatcher> ConstPtr;

  typedef PointMatcher<double> PM;
  typedef PM::DataPoints DP;
  typedef PM::Parameters Parameters;

 public:
  AlignPointMatcher(){
    icp.setDefault();
    icp_sequence_.setDefault();
  }
  AlignPointMatcher(const std::string& yaml_file);
  ~AlignPointMatcher() = default;

  template <typename PointT>
  bool Align(const typename pcl::PointCloud<PointT>::ConstPtr reference_cloud,
             const typename pcl::PointCloud<PointT>::ConstPtr reading_cloud,
             Eigen::Matrix4d& trans_matrix, double& score);
  bool setmap(const std::string& map_file);
  template <typename PointT>
  bool SetMap(const typename pcl::PointCloud<PointT>::ConstPtr reference_coud);
  template <typename PointT>
  bool AlignWithMap(const typename pcl::PointCloud<PointT>::ConstPtr reading_cloud,
            Eigen::Matrix4d& trans_matrix, double& score); 
  
  bool AlignTest(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr reading_cloud,
                 const pcl::PointCloud<pcl::PointXYZI>::ConstPtr ref_cloud,
                 Eigen::Matrix4d& tf, double& score);
  struct Config {
    double max_trans;
    double max_angle;
    double min_trans;
  };

 private:
  bool Init(const std::string& config_file);
  template <typename PointT>
  void ConverToDataPoints(
      const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP& cur_pts);

  void CheckInitTransformation(const Eigen::Matrix4d& init_matrix,
                               PM::TransformationParameters& init_trans);

  void CalOdomRobustScore(double& score);
  void CalMapRobustScore(double& score);
  void CalHaussdorffScore(const DP& ref, const DP& reading,
                          const PM::TransformationParameters& T, double& score);

 private:
  Eigen::Matrix4d prev_trans_;
  Eigen::Matrix4d odom_pose_;
  Config config_;
  PM::ICP icp;
  PM::ICPSequence icp_sequence_;
  // check init_matrix
  std::shared_ptr<PM::Transformation> rigid_trans_;

  // evaluate score:haussdorff distance
  std::shared_ptr<PM::Matcher> matcher_hausdorff_;
};
