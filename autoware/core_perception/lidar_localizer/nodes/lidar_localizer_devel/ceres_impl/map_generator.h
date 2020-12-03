#include <autoware_config_msgs/ConfigNDTMapping.h>
#include <autoware_config_msgs/ConfigNDTMappingOutput.h>
#include <glog/logging.h>
#include <lidar_alignment/timer.h>
#include <nav_msgs/Odometry.h>
#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <time.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;

struct Pose {
  double stamp = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  Pose& operator=(const Pose& rhs) {
    if (this == &rhs) {
      return *this;
    }
    this->stamp = rhs.stamp;
    this->x = rhs.x;
    this->y = rhs.y;
    this->z = rhs.z;
    this->roll = rhs.roll;
    this->pitch = rhs.pitch;
    this->yaw = rhs.yaw;
    return *this;
  }
};

struct MappingConfig {
  double time_threshold = 0.02;
  double scan_calibration_angle = 0;

  // filter params
  double min_scan_range = 5;
  double max_scan_range = 60;
  double voxel_leaf_size = 1;  // autoware 2.0

  // ndt
  double ndt_res = 1;
  double step_size = 0.1;
  double trans_eps = 0.01;  // Transformation epsilon
  double max_iter = 30;     // Maximum iterations

  // extend map
  double min_add_scan_shift = 1;  // 0.15 0.2 0.5 original

  // back-end optimization params
  CeresConfig ceres_config;

  // mapping
  std::string map_init_filename = "";
  double map_voxel_filter_size = 0.2;
  double keyframe_delta_trans = 8;
  double map_cloud_update_interval = 10.0;
};

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointCloudPtr cloud;
  Eigen::Isometry3d pose;
  double stamp = 0.0;
  KeyFrame() {}
  KeyFrame(const double& stamp, const Eigen::Isometry3d& pose,
           const PointCloudPtr& cloud)
      : stamp(stamp), pose(pose), cloud(cloud) {}
};
typedef std::shared_ptr<KeyFrame> KeyFramePtr;
typedef std::shared_ptr<const KeyFrame> KeyFrameConstPtr;
std::mutex keyframe_mutex;

void PublishPose(ros::Publisher& pub, const Pose pose, std::string frame_id);
void InputParams(ros::NodeHandle& private_nh, MappingConfig& config);
void OutputParams(const MappingConfig& config);
Pose Matrix2Pose(const Eigen::Matrix4f matrix, const double stamp);
void ClearQueue(std::queue<geometry_msgs::PoseStamped::ConstPtr>& queue);
void FilterByDist(const PointCloud& raw_scan, PointCloudPtr& output);
void DumpPose(const Eigen::Affine3d pose, const std::string filename);
Eigen::Matrix4f TransformPoseToEigenMatrix4f(const Pose& ref);
double calcDiffForRadian(const double lhs_rad, const double rhs_rad);

void AddKeyFrame(const double& time, const Eigen::Matrix4d& tf,
                 const Eigen::Matrix4d& pose, const PointCloudPtr cloud,
                 const Eigen::Affine3d& map_init, const MappingConfig& config);