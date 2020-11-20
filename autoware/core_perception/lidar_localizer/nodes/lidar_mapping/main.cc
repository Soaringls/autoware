#include <autoware_config_msgs/ConfigNDTMapping.h>
#include <autoware_config_msgs/ConfigNDTMappingOutput.h>
#include <geometry_msgs/PoseStamped.h>
#include <glog/logging.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <fstream>
#include <iostream>
#include <queue>

#include "slam.h"

// using namespace lidar_mapping;
// global variables
MappingConfig config;
std::mutex gps_mutex;
std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;

// AlignSLAM slam(config);
// AlignSLAM slam;
AlignSLAMPtr align_slam_ptr;

bool FindCorrespondGpsMsg(const double pts_stamp, Eigen::Affine3d& ins_pose) {
  bool is_find(false);
  gps_mutex.lock();
  while (!gps_msgs.empty()) {
    auto stamp_diff = gps_msgs.front()->header.stamp.toSec() - pts_stamp;
    if (std::abs(stamp_diff) <= config.time_synchronize_threshold) {
      auto gps_msg = gps_msgs.front();
      auto pos = Eigen::Translation3d(gps_msg->pose.position.x,
                                      gps_msg->pose.position.y,
                                      gps_msg->pose.position.z);
      auto q = Eigen::Quaterniond(
          gps_msg->pose.orientation.w, gps_msg->pose.orientation.x,
          gps_msg->pose.orientation.y, gps_msg->pose.orientation.z);
      ins_pose = pos * q.normalized();
      gps_msgs.pop();
      is_find = true;
      break;
    } else if (stamp_diff < -config.time_synchronize_threshold) {
      gps_msgs.pop();
    } else if (stamp_diff > config.time_synchronize_threshold) {
      LOG(INFO) << "(gps_time - pts_time = " << stamp_diff
                << ") lidar msgs is delayed! ";
      break;
    }
  }
  gps_mutex.unlock();
  return is_find;
}

PointCloudPtr RemoveNanAndFilterBydist(const PointCloudPtr& cloud) {
  PointCloudPtr filtered(new PointCloud);
  for (auto pt : cloud->points) {
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
        std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z))
      continue;
    double dist = pt.getVector3fMap().norm();
    if (dist < config.distance_near_thresh || dist > config.distance_far_thresh)
      continue;
    filtered->push_back(pt);
  }
  filtered->header = cloud->header;
  return filtered;
}
// PointCloudPtr VoxelFilter(const PointCloudPtr cloud, const double size) {
//   PointCloudPtr filtered(new PointCloud);
//   pcl::VoxelGrid<PointT> voxel_filter_;
//   voxel_filter_.setLeafSize(size, size, size);
//   voxel_filter_.setInputCloud(cloud);
//   voxel_filter_.filter(*filtered);
//   filtered->header = cloud->header;
//   return filtered;
// }

void PointsCallback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  double timestamp = input->header.stamp.toSec();

  static bool init(false);
  static Eigen::Matrix4d frame_pose;
  if (!init) {
    Eigen::Affine3d map_init;
    if (!FindCorrespondGpsMsg(timestamp, map_init)) {
      LOG(INFO) << "failed to find the gps msgs for pointscallback init";
      return;
    }
    frame_pose = map_init.matrix();
    init = true;
    return;
  }

  Eigen::Affine3d ins_pose;
  if (!FindCorrespondGpsMsg(timestamp, ins_pose)) {
    LOG(INFO) << "failed to search ins_pose at:" << std::fixed
              << std::setprecision(6) << timestamp;
    return;
  }
  auto trans = frame_pose.inverse() * ins_pose;
  auto dist = trans.matrix().block<3, 1>(0, 3).norm();
  if (dist < config.input_frame_dist) {
    LOG(INFO) << "the distance to last frame is :" << dist << " too small!!!";
    return;
  }

  LOG(INFO) << "-------------------new lidar-msg------------------------";
  Timer t;
  PointCloudPtr point_msg(new PointCloud);
  pcl::fromROSMsg(*input, *point_msg);
  LOG(INFO) << "pts before filter:" << point_msg->size();
  point_msg = RemoveNanAndFilterBydist(point_msg);
  //   point_msg = VoxelFilter(point_msg, config.voxel_filter_size);
  LOG(INFO) << "pts after  filter:" << point_msg->size();
  if (point_msg->size() == 0) return;
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.rotate(Eigen::AngleAxisd(config.lidar_yaw_calib_degree * M_PI / 180,
                                     Eigen::Vector3d::UnitZ()));
  pcl::transformPointCloud(*point_msg, *point_msg, transform);

  static std::size_t frame_cnt(1);
  align_slam_ptr->InScan(point_msg, frame_cnt++);

  LOG(INFO) << "processor elapsed:" << t.end() << " [ms]";
}

void InsCallback(const geometry_msgs::PoseStamped::ConstPtr& msgs) {
  gps_mutex.lock();
  gps_msgs.push(msgs);
  gps_mutex.unlock();
}

void output_callback(
    const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input) {
  Timer t;
  LOG(INFO) << "================begin to save map====================";
  std::string map_filename = input->filename;
  align_slam_ptr->SaveMap(map_filename, "pcl");
  LOG(INFO) << "Saved map to " << map_filename << " success!!!";
}

void InitParams(ros::NodeHandle& nh) {
  nh.getParam("gps_lidar_time_threshold", config.time_synchronize_threshold);
  // params filter
  nh.getParam("seg_config_yaml", config.seg_config_yaml);
  nh.getParam("lidar_yaw_calib_degree", config.lidar_yaw_calib_degree);
  nh.getParam("time_synchronize_threshold", config.time_synchronize_threshold);
  nh.getParam("distance_near_thresh", config.distance_near_thresh);
  nh.getParam("distance_far_thresh", config.distance_far_thresh);
  nh.getParam("voxel_filter_size", config.voxel_filter_size);
  nh.getParam("input_frame_dist", config.input_frame_dist);
  nh.getParam("submap_length", config.submap_length);
  nh.getParam("submap_frame_resol", config.submap_frame_resol);
  align_slam_ptr.reset(new AlignSLAM(config));
  LOG(INFO) << "init success.";
}
int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_mapping");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  InitParams(private_nh);

  LOG(INFO) << "begin to subscribe topic...";
  ros::Subscriber points_sub =
      nh.subscribe("/points_raw", 10000, PointsCallback);
  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 30000, InsCallback);
  ros::Subscriber output_sub =
      nh.subscribe("config/ndt_mapping_output", 10, output_callback);
  ros::spin();
  return 0;
}