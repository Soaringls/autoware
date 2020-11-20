#pragma once
#include <glog/logging.h>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <float.h>
#include <boost/circular_buffer.hpp>
#include "config_loader.h"

namespace lidar_alignment{

struct SegmentedCloudMsg {
  //record start and end segmented point index of each line
  std::vector<int32_t> startRingIndex;
  std::vector<int32_t> endRingIndex;
  //lidar scan satrt and end angle
  float startOrientation;
  float endOrientation;
  float orientationDiff;
  //true-is ground point, false-is other points
  std::vector<bool> segmentedCloudGroundFlag;
  //relative column index of segmented point in range image
  std::vector<uint32_t> segmentedCloudColInd;
  //relative index of segmented point in full point cloud
  std::vector<uint32_t> indexInFullPointCloud;
  //segmented point range
  std::vector<float> segmentedCloudRange;
};

class LidarSegmentation{
 public:
  typedef pcl::PointXYZI PointT;
  typedef pcl::PointCloud<PointT> PointCloudI;
  typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
  typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

  LidarSegmentation(const std::string& lidar_config_file);
  ~LidarSegmentation() = default;

 
  void CloudMsgHandler(PointCloudPtr cloud_in);
  PointCloudPtr GetSegmentedCloud() { return segmented_cloud_with_ground_;}
  PointCloudPtr GetSegmentedCloudPure() { return segmented_cloud_pure_;}
  PointCloudPtr GetGroundCloud() { return ground_cloud_;}

 private:
  void resetParameters();
  void findStartEndAngle();
  void projectPointCloud();
  void groundRemoval();
  void cloudClusters();
  void labelComponents(int row, int col);

  float angle_resolution_x_ = (M_PI * 2) / 1800;
  float angle_resolution_y_ = common::DEG_TO_RAD_D * 2;
  //record the number of label
  int label_count_ = 0;

  PointCloudPtr lidar_cloud_in_;

  PointCloudPtr segmented_cloud_with_ground_;
  PointCloudPtr segmented_cloud_pure_;  
  PointCloudPtr ground_cloud_;

  PointCloudPtr full_cloud_;
  PointCloudPtr full_cloud_with_range_;
  PointCloudPtr outlier_cloud_;


  SegmentedCloudMsg seg_msg_;
  //point cloud range, size is vertical*horizontal nums
  std::vector<float> point_cloud_range_;


  Eigen::MatrixXf range_mat_; //range image
  Eigen::MatrixXi label_mat_; //label matrix for segmentation marking
  
  // ground_mat: ground matrix for ground cloud marking
  // -1, no valid info to check if ground of not
  //  0, initial value, after validation, means not ground
  //  1, ground
  Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> ground_mat_;
};
typedef std::shared_ptr<LidarSegmentation> LidarSegmentationPtr;
typedef std::shared_ptr<const LidarSegmentation> LidarSegmentationConstPtr;

}