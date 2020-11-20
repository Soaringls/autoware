#pragma once
#include <glog/logging.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pointmatcher/Parametrizable.h>
#include <pointmatcher/PointMatcher.h>

#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <unordered_map>

#include "align_pointmatcher.h"
#include "align_util.h"
#include "lidar_alignment/lidar_segmentation.h"
#include "timer.h"

using namespace lidar_alignment;
using namespace autobot::cyberverse::mapping;
using namespace std;
using namespace PointMatcherSupport;
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

struct MappingConfig {
  // init
  std::string seg_config_yaml = "";
  // preprocess cloud
  double lidar_yaw_calib_degree = 90;
  // filter cloud
  double distance_near_thresh = 2;
  double distance_far_thresh = 50;
  double voxel_filter_size = 0.12;
  double octree_filter_resol = 0.1;

  double time_synchronize_threshold = 0.02;
  double submap_length = 5.0;       // m
  double submap_frame_resol = 0.7;  // m
  double input_frame_dist = 0.1;

  // config pointmatcher
  std::string config_yaml =
      "/autoware/workspace/data/params/libpointmatcher_config.p2plane.yaml";
};

class AlignSLAM {
 public:
  explicit AlignSLAM(const MappingConfig &config)
      : config_(config),
        odo_align_(new AlignPointMatcher),
        map_align_(new AlignPointMatcher) {
    bool init_ = false;
    current_tf_ = Eigen::Affine3d::Identity();
    last_tf_ = Eigen::Affine3d::Identity();
    lidar_segmentation_ptr.reset(
        new LidarSegmentation(config_.seg_config_yaml));
  }

  void SetIntermediateAlignDir(const std::string intermediate_align_dir) {
    intermediate_align_dir_ = intermediate_align_dir;
  }

  bool InScan(const PointCloudPtr cloud_in, int id);
  bool InScan(const DP &scan, int id);
  void SaveMap(const std::string map_filename,
               const std::string &method = "pointmatcher");

 private:
  PointCloudPtr SegmentedCloud(const PointCloudPtr cloud) {
    auto header = cloud->header;
    lidar_segmentation_ptr->CloudMsgHandler(cloud);
    // auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloud();
    auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloudPure();
    seg_cloud->header = header;
    return seg_cloud;
  }
  PointCloudPtr TransformCloud(const PointCloudPtr cloud,
                               const Eigen::Affine3d &pose);
  //   PointCloudPtr VoxelFilter(const PointCloudPtr cloud, const double size);
  PointCloudPtr FilterByOctree(const PointCloudPtr cloud_ptr,
                               double resolution = 0.05);
  template <typename PointT>
  void ConverToDataPoints(
      const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP &cur_pts);
  bool Odometry(DP &scan, int id);
  bool Scan2Map(const DP &scan, int id);

  std::vector<int> SearchNearbyScan(int id);
  DP ConcatenateNearbyMap(const std::vector<int> &nearbors);

 private:
  MappingConfig config_;
  //   pcl::VoxelGrid<PointT> voxel_filter_;
  LidarSegmentationPtr lidar_segmentation_ptr;
  bool init_ = false;
  int init_id_ = 0;
  Eigen::Affine3d current_tf_, last_tf_;
  DP last_scan_;

  AlignPointMatcher::Ptr odo_align_;
  AlignPointMatcher::Ptr map_align_;

  std::unordered_map<int, Eigen::Affine3d, hash<int>, std::equal_to<int>,
                     Eigen::aligned_allocator<Eigen::Affine3d>>
      poses_;
  std::unordered_map<int, DP> scans_;
  std::unordered_map<int, DP> raw_scans_;
  std::unordered_map<int, PointCloudPtr> raw_frames_;
  std::string intermediate_align_dir_ = "./tmp/";
};
typedef std::shared_ptr<AlignSLAM> AlignSLAMPtr;