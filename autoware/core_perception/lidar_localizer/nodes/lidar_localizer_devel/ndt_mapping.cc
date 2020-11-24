/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 Localization and mapping program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#define OUTPUT  // If you want to output "position_log.txt", "#define OUTPUT".

#include <glog/logging.h>
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
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt.h>
#endif

#include <autoware_config_msgs/ConfigNDTMapping.h>
#include <autoware_config_msgs/ConfigNDTMappingOutput.h>
#include <lidar_alignment/timer.h>
#include <time.h>

using namespace lidar_alignment;
struct pose {
  double stamp = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  pose& operator=(const pose& rhs) {
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

enum class MethodType {
  PCL_GENERIC = 0,
  PCL_ANH = 1,
  PCL_ANH_GPU = 2,
  PCL_OPENMP = 3,
};

double config_time_threshold = 0.02;
std::string map_init_pos_file = "";
std::mutex gps_mutex;
std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;
static MethodType _method_type = MethodType::PCL_GENERIC;

// global variables
static pose previous_pose, guess_pose, guess_pose_imu, guess_pose_odom,
    guess_pose_imu_odom, current_pose, current_pose_imu, current_pose_odom,
    current_pose_imu_odom, ndt_pose, added_pose;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0,
              diff_yaw;  // current_pose - previous_pose
static double offset_imu_x, offset_imu_y, offset_imu_z, offset_imu_roll,
    offset_imu_pitch, offset_imu_yaw;
static double offset_odom_x, offset_odom_y, offset_odom_z, offset_odom_roll,
    offset_odom_pitch, offset_odom_yaw;
static double offset_imu_odom_x, offset_imu_odom_y, offset_imu_odom_z,
    offset_imu_odom_roll, offset_imu_odom_pitch, offset_imu_odom_yaw;

static double current_velocity_x = 0.0;
static double current_velocity_y = 0.0;
static double current_velocity_z = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;

static pcl::PointCloud<pcl::PointXYZI> map;

static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
static cpu::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>
    anh_ndt;

#ifdef USE_PCL_OPENMP
static pcl_omp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>
    omp_ndt;
#endif

// Default values
static int max_iter = 30;             // Maximum iterations
static float ndt_res = 1.0;           // Resolution
static double step_size = 0.1;        // Step size
static double trans_eps = 0.01;       // Transformation epsilon
static double voxel_leaf_size = 1.0;  // 2.0;

static ros::Time callback_start, callback_end, t1_start, t1_end, t2_start,
    t2_end, t3_start, t3_end, t4_start, t4_end, t5_start, t5_end;
static ros::Duration d_callback, d1, d2, d3, d4, d5;

static ros::Publisher ndt_map_pub;
static ros::Publisher current_pose_pub;
static ros::Publisher guess_pose_linaer_pub;
static geometry_msgs::PoseStamped current_pose_msg, guess_pose_msg;

static ros::Publisher ndt_stat_pub;
static std_msgs::Bool ndt_stat_msg;

static int initial_scan_loaded = 0;

static Eigen::Matrix4f gnss_transform = Eigen::Matrix4f::Identity();

static double min_scan_range = 5.0;
static double max_scan_range = 60.0;
static double min_add_scan_shift = 1.0;

static Eigen::Matrix4f tf_btol, tf_ltob;

static bool _use_imu = false;
static bool _use_odom = false;
static bool _imu_upside_down = false;

static bool _incremental_voxel_update = false;

static std::string _imu_topic = "/imu_raw";

static double fitness_score;
static bool has_converged;
static int final_num_iteration;
static double transformation_probability;

static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

static std::ofstream ofs;
static std::string filename;

void PublishPose(ros::Publisher& pub, const pose pose, std::string frame_id) {
  tf::Quaternion q;
  q.setRPY(pose.roll, pose.pitch, pose.yaw);

  geometry_msgs::PoseStamped msg;
  msg.header.frame_id = frame_id;
  msg.header.stamp = ros::Time(pose.stamp);
  msg.pose.position.x = pose.x;
  msg.pose.position.y = pose.y;
  msg.pose.position.z = pose.z;

  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();
  msg.pose.orientation.w = q.w();

  pub.publish(msg);
}

pose Matrix2Pose(const Eigen::Matrix4f matrix, const double stamp) {
  tf::Matrix3x3 mat_tf;
  mat_tf.setValue(
      static_cast<double>(matrix(0, 0)), static_cast<double>(matrix(0, 1)),
      static_cast<double>(matrix(0, 2)), static_cast<double>(matrix(1, 0)),
      static_cast<double>(matrix(1, 1)), static_cast<double>(matrix(1, 2)),
      static_cast<double>(matrix(2, 0)), static_cast<double>(matrix(2, 1)),
      static_cast<double>(matrix(2, 2)));
  pose result;
  result.stamp = stamp;
  result.x = matrix(0, 3);
  result.y = matrix(1, 3);
  result.z = matrix(2, 3);
  mat_tf.getRPY(result.roll, result.pitch, result.yaw, 1);

  return result;
}

static void param_callback(
    const autoware_config_msgs::ConfigNDTMapping::ConstPtr& input) {
  ndt_res = input->resolution;                     // 1
  step_size = input->step_size;                    // 0.1
  trans_eps = input->trans_epsilon;                // 0.01
  max_iter = input->max_iterations;                // 30
  voxel_leaf_size = input->leaf_size;              // 1
  min_scan_range = input->min_scan_range;          // 5
  max_scan_range = input->max_scan_range;          // 60
  min_add_scan_shift = input->min_add_scan_shift;  // 1

  LOG(INFO) << "set param_param_callback";
  LOG(INFO) << "set param_ndt_res: " << ndt_res;
  LOG(INFO) << "set param_step_size: " << step_size;
  LOG(INFO) << "set param_trans_epsilon: " << trans_eps;
  LOG(INFO) << "set param_max_iter: " << max_iter;
  LOG(INFO) << "set param_voxel_leaf_size: " << voxel_leaf_size;
  LOG(INFO) << "set param_min_scan_range: " << min_scan_range;
  LOG(INFO) << "set param_max_scan_range: " << max_scan_range;
  LOG(INFO) << "set param_min_add_scan_shift: " << min_add_scan_shift;
}

static void output_callback(
    const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input) {
  if (map.empty()) {
    LOG(WARNING) << "map is empty!";
    return;
  }

  double filter_res = input->filter_res;
  std::string filename = input->filename;
  std::cout << "output_callback" << std::endl;
  std::cout << "filter_res: " << filter_res << std::endl;
  std::cout << "filename: " << filename << std::endl;

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(
      new pcl::PointCloud<pcl::PointXYZI>(map));
  pcl::PointCloud<pcl::PointXYZI>::Ptr map_filtered(
      new pcl::PointCloud<pcl::PointXYZI>());
  map_ptr->header.frame_id = "map";
  map_filtered->header.frame_id = "map";
  sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
  for (int i = 0; i < 200; ++i) {
    LOG(INFO) << std::fixed << std::setprecision(6)
              << "map-point:" << map_ptr->points[i].x << ", "
              << map_ptr->points[i].y << ", " << map_ptr->points[i].z;
  }
  // Apply voxelgrid filter
  if (filter_res == 0.0) {
    std::cout << "Original: " << map_ptr->points.size() << " points."
              << std::endl;
    // pcl::toROSMsg(*map_ptr, *map_msg_ptr);
  } else {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(filter_res, filter_res, filter_res);
    voxel_grid_filter.setInputCloud(map_ptr);
    voxel_grid_filter.filter(*map_filtered);
    std::cout << "Original: " << map_ptr->points.size() << " points."
              << std::endl;
    std::cout << "Filtered: " << map_filtered->points.size() << " points."
              << std::endl;
    // pcl::toROSMsg(*map_filtered, *map_msg_ptr);
  }

  // ndt_map_pub.publish(*map_msg_ptr);

  // Writing Point Cloud data to PCD file
  if (filter_res == 0.0) {
    pcl::io::savePCDFileBinary(filename, *map_ptr);
    LOG(INFO) << "Saved " << map_ptr->points.size() << " data origin points to "
              << filename;
  } else {
    pcl::io::savePCDFileBinary(filename, *map_filtered);
    LOG(INFO) << "Saved " << map_filtered->points.size()
              << " data filtered's size[" << filter_res << "] points to "
              << filename;
  }
  LOG(INFO) << "map_points save finished.";
}

static void imu_odom_calc(ros::Time current_time, pose& pose_imu_odom) {
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_imu_roll = imu.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu.angular_velocity.z * diff_time;

  pose_imu_odom.roll += diff_imu_roll;
  pose_imu_odom.pitch += diff_imu_pitch;
  pose_imu_odom.yaw += diff_imu_yaw;

  double diff_distance = odom.twist.twist.linear.x * diff_time;
  offset_imu_odom_x +=
      diff_distance * cos(-pose_imu_odom.pitch) * cos(pose_imu_odom.yaw);
  offset_imu_odom_y +=
      diff_distance * cos(-pose_imu_odom.pitch) * sin(pose_imu_odom.yaw);
  offset_imu_odom_z += diff_distance * sin(-pose_imu_odom.pitch);

  offset_imu_odom_roll += diff_imu_roll;
  offset_imu_odom_pitch += diff_imu_pitch;
  offset_imu_odom_yaw += diff_imu_yaw;

  guess_pose_imu_odom.x = previous_pose.x + offset_imu_odom_x;
  guess_pose_imu_odom.y = previous_pose.y + offset_imu_odom_y;
  guess_pose_imu_odom.z = previous_pose.z + offset_imu_odom_z;
  guess_pose_imu_odom.roll = previous_pose.roll + offset_imu_odom_roll;
  guess_pose_imu_odom.pitch = previous_pose.pitch + offset_imu_odom_pitch;
  guess_pose_imu_odom.yaw = previous_pose.yaw + offset_imu_odom_yaw;

  previous_time = current_time;
}

static void odom_calc(ros::Time current_time, pose& pose_odom) {
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_odom_roll = odom.twist.twist.angular.x * diff_time;
  double diff_odom_pitch = odom.twist.twist.angular.y * diff_time;
  double diff_odom_yaw = odom.twist.twist.angular.z * diff_time;

  pose_odom.roll += diff_odom_roll;
  pose_odom.pitch += diff_odom_pitch;
  pose_odom.yaw += diff_odom_yaw;

  double diff_distance = odom.twist.twist.linear.x * diff_time;
  offset_odom_x += diff_distance * cos(-pose_odom.pitch) * cos(pose_odom.yaw);
  offset_odom_y += diff_distance * cos(-pose_odom.pitch) * sin(pose_odom.yaw);
  offset_odom_z += diff_distance * sin(-pose_odom.pitch);

  offset_odom_roll += diff_odom_roll;
  offset_odom_pitch += diff_odom_pitch;
  offset_odom_yaw += diff_odom_yaw;

  guess_pose_odom.x = previous_pose.x + offset_odom_x;
  guess_pose_odom.y = previous_pose.y + offset_odom_y;
  guess_pose_odom.z = previous_pose.z + offset_odom_z;
  guess_pose_odom.roll = previous_pose.roll + offset_odom_roll;
  guess_pose_odom.pitch = previous_pose.pitch + offset_odom_pitch;
  guess_pose_odom.yaw = previous_pose.yaw + offset_odom_yaw;

  previous_time = current_time;
}

static void imu_calc(ros::Time current_time, pose& pose_imu) {
  static ros::Time previous_time = current_time;
  double diff_time = (current_time - previous_time).toSec();

  double diff_imu_roll = imu.angular_velocity.x * diff_time;
  double diff_imu_pitch = imu.angular_velocity.y * diff_time;
  double diff_imu_yaw = imu.angular_velocity.z * diff_time;

  pose_imu.roll += diff_imu_roll;
  pose_imu.pitch += diff_imu_pitch;
  pose_imu.yaw += diff_imu_yaw;

  double accX1 = imu.linear_acceleration.x;
  double accY1 = std::cos(pose_imu.roll) * imu.linear_acceleration.y -
                 std::sin(pose_imu.roll) * imu.linear_acceleration.z;
  double accZ1 = std::sin(pose_imu.roll) * imu.linear_acceleration.y +
                 std::cos(pose_imu.roll) * imu.linear_acceleration.z;

  double accX2 =
      std::sin(pose_imu.pitch) * accZ1 + std::cos(pose_imu.pitch) * accX1;
  double accY2 = accY1;
  double accZ2 =
      std::cos(pose_imu.pitch) * accZ1 - std::sin(pose_imu.pitch) * accX1;

  double accX = std::cos(pose_imu.yaw) * accX2 - std::sin(pose_imu.yaw) * accY2;
  double accY = std::sin(pose_imu.yaw) * accX2 + std::cos(pose_imu.yaw) * accY2;
  double accZ = accZ2;

  offset_imu_x +=
      current_velocity_imu_x * diff_time + accX * diff_time * diff_time / 2.0;
  offset_imu_y +=
      current_velocity_imu_y * diff_time + accY * diff_time * diff_time / 2.0;
  offset_imu_z +=
      current_velocity_imu_z * diff_time + accZ * diff_time * diff_time / 2.0;

  current_velocity_imu_x += accX * diff_time;
  current_velocity_imu_y += accY * diff_time;
  current_velocity_imu_z += accZ * diff_time;

  offset_imu_roll += diff_imu_roll;
  offset_imu_pitch += diff_imu_pitch;
  offset_imu_yaw += diff_imu_yaw;

  guess_pose_imu.x = previous_pose.x + offset_imu_x;
  guess_pose_imu.y = previous_pose.y + offset_imu_y;
  guess_pose_imu.z = previous_pose.z + offset_imu_z;
  guess_pose_imu.roll = previous_pose.roll + offset_imu_roll;
  guess_pose_imu.pitch = previous_pose.pitch + offset_imu_pitch;
  guess_pose_imu.yaw = previous_pose.yaw + offset_imu_yaw;

  previous_time = current_time;
}

static double wrapToPm(double a_num, const double a_max) {
  if (a_num >= a_max) {
    a_num -= 2.0 * a_max;
  }
  return a_num;
}

static double wrapToPmPi(double a_angle_rad) {
  return wrapToPm(a_angle_rad, M_PI);
}

static double calcDiffForRadian(const double lhs_rad, const double rhs_rad) {
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}
static void odom_callback(const nav_msgs::Odometry::ConstPtr& input) {
  // std::cout << __func__ << std::endl;

  odom = *input;
  odom_calc(input->header.stamp, current_pose_odom);
}

static void imuUpsideDown(const sensor_msgs::Imu::Ptr input) {
  double input_roll, input_pitch, input_yaw;

  tf::Quaternion input_orientation;
  tf::quaternionMsgToTF(input->orientation, input_orientation);
  tf::Matrix3x3(input_orientation).getRPY(input_roll, input_pitch, input_yaw);

  input->angular_velocity.x *= -1;
  input->angular_velocity.y *= -1;
  input->angular_velocity.z *= -1;

  input->linear_acceleration.x *= -1;
  input->linear_acceleration.y *= -1;
  input->linear_acceleration.z *= -1;

  input_roll *= -1;
  input_pitch *= -1;
  input_yaw *= -1;

  input->orientation = tf::createQuaternionMsgFromRollPitchYaw(
      input_roll, input_pitch, input_yaw);
}

static void imu_callback(const sensor_msgs::Imu::Ptr& input) {
  // std::cout << __func__ << std::endl;

  if (_imu_upside_down) imuUpsideDown(input);

  const ros::Time current_time = input->header.stamp;
  static ros::Time previous_time = current_time;
  const double diff_time = (current_time - previous_time).toSec();

  double imu_roll, imu_pitch, imu_yaw;
  tf::Quaternion imu_orientation;
  tf::quaternionMsgToTF(input->orientation, imu_orientation);
  tf::Matrix3x3(imu_orientation).getRPY(imu_roll, imu_pitch, imu_yaw);

  imu_roll = wrapToPmPi(imu_roll);
  imu_pitch = wrapToPmPi(imu_pitch);
  imu_yaw = wrapToPmPi(imu_yaw);

  static double previous_imu_roll = imu_roll, previous_imu_pitch = imu_pitch,
                previous_imu_yaw = imu_yaw;
  const double diff_imu_roll = calcDiffForRadian(imu_roll, previous_imu_roll);
  const double diff_imu_pitch =
      calcDiffForRadian(imu_pitch, previous_imu_pitch);
  const double diff_imu_yaw = calcDiffForRadian(imu_yaw, previous_imu_yaw);

  imu.header = input->header;
  imu.linear_acceleration.x = input->linear_acceleration.x;
  // imu.linear_acceleration.y = input->linear_acceleration.y;
  // imu.linear_acceleration.z = input->linear_acceleration.z;
  imu.linear_acceleration.y = 0;
  imu.linear_acceleration.z = 0;

  if (diff_time != 0) {
    imu.angular_velocity.x = diff_imu_roll / diff_time;
    imu.angular_velocity.y = diff_imu_pitch / diff_time;
    imu.angular_velocity.z = diff_imu_yaw / diff_time;
  } else {
    imu.angular_velocity.x = 0;
    imu.angular_velocity.y = 0;
    imu.angular_velocity.z = 0;
  }

  imu_calc(input->header.stamp, current_pose_imu);

  previous_time = current_time;
  previous_imu_roll = imu_roll;
  previous_imu_pitch = imu_pitch;
  previous_imu_yaw = imu_yaw;
}

Eigen::Matrix4f TransformPoseToEigenMatrix4f(const pose& ref) {
  Eigen::AngleAxisf x_rotation_vec(ref.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf y_rotation_vec(ref.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf z_rotation_vec(ref.yaw, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f t(ref.x, ref.y, ref.z);
  return (t * z_rotation_vec * y_rotation_vec * x_rotation_vec).matrix();
}

bool FindCorrespondGpsMsg(const double pts_stamp, Eigen::Affine3d& ins_pose) {
  bool is_find(false);
  gps_mutex.lock();
  while (!gps_msgs.empty()) {
    auto stamp_diff = gps_msgs.front()->header.stamp.toSec() - pts_stamp;
    if (std::abs(stamp_diff) <= config_time_threshold) {
      auto gps_msg = gps_msgs.front();
      ins_pose = Eigen::Translation3d(gps_msg->pose.position.x,
                                      gps_msg->pose.position.y,
                                      gps_msg->pose.position.z) *
                 Eigen::Quaterniond(
                     gps_msg->pose.orientation.w, gps_msg->pose.orientation.x,
                     gps_msg->pose.orientation.y, gps_msg->pose.orientation.z);
      gps_msgs.pop();
      is_find = true;
      break;
    } else if (stamp_diff < -config_time_threshold) {
      gps_msgs.pop();
    } else if (stamp_diff > config_time_threshold) {
      LOG(INFO) << "(gps_time - pts_time = " << stamp_diff
                << ") lidar msgs is delayed! ";
      break;
    }
  }
  gps_mutex.unlock();
  return is_find;
}

void DumpPose(const Eigen::Affine3d pose, const std::string filename) {
  std::ofstream fo;
  fo.open(filename.c_str(), std::ofstream::out);
  if (fo.is_open()) {
    fo.setf(std::ios::fixed, std::ios::floatfield);
    fo.precision(6);
    Eigen::Quaterniond q(pose.linear());
    Eigen::Translation3d t(pose.translation());
    double qx = q.x();
    double qy = q.y();
    double qz = q.z();
    double qw = q.w();
    fo << t.x() << " " << t.y() << " " << t.z() << " " << qw << " " << qx << " "
       << qy << " " << qz << "\n";

    fo.close();
  } else {
    LOG(WARNING) << "failed to dump pose optimized to the file:" << filename;
  }
}
static bool init_flag = false;
static Eigen::Affine3d init_pose = Eigen::Affine3d::Identity();
static void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& input) {
  gps_mutex.lock();
  gps_msgs.push(input);
  gps_mutex.unlock();

  if (!init_flag) {
    init_pose =
        Eigen::Translation3d(input->pose.position.x, input->pose.position.y,
                             input->pose.position.z) *
        Eigen::Quaterniond(input->pose.orientation.w, input->pose.orientation.x,
                           input->pose.orientation.y,
                           input->pose.orientation.z);
    DumpPose(init_pose, map_init_pos_file);
    init_flag = true;
    LOG(INFO) << std::fixed << std::setprecision(6)
              << "init gnss(global):" << input->pose.position.x << ", "
              << input->pose.position.y << ", " << input->pose.position.z;
  }
  LOG(INFO) << std::fixed << std::setprecision(6)
            << "real-time gnss(global):" << input->pose.position.x << ", "
            << input->pose.position.y << ", " << input->pose.position.z;

  Eigen::Affine3d cur_pose(
      Eigen::Translation3d(input->pose.position.x, input->pose.position.y,
                           input->pose.position.z) *
      Eigen::Quaterniond(input->pose.orientation.w, input->pose.orientation.x,
                         input->pose.orientation.y, input->pose.orientation.z));
  cur_pose = init_pose.inverse() * cur_pose;

  Eigen::Quaterniond q(cur_pose.linear());
  tf::Matrix3x3 matrix_q(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
  matrix_q.getRPY(guess_pose_odom.roll, guess_pose_odom.pitch,
                  guess_pose_odom.yaw);
  guess_pose_odom.x = cur_pose.translation().x();
  guess_pose_odom.y = cur_pose.translation().y();
  guess_pose_odom.z = cur_pose.translation().z();
  LOG(INFO) << std::fixed << std::setprecision(6)
            << "real-time gnss(local):" << guess_pose_odom.x << ", "
            << guess_pose_odom.y << ", " << guess_pose_odom.z;
  // static nav_msgs::Path path;
  // AddPoseToNavPath(path, guess_pose_odom);
  // gps_pose_pub.publish(path);
}

static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  if (!init_flag) return;
  double r;
  pcl::PointXYZI p;
  pcl::PointCloud<pcl::PointXYZI> tmp, scan;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(
      new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(
      new pcl::PointCloud<pcl::PointXYZI>());

  static tf::TransformBroadcaster br;
  tf::Transform transform;

  current_scan_time = input->header.stamp;
  double timestamp = input->header.stamp.toSec();

  pcl::fromROSMsg(*input, tmp);
  LOG(INFO) << "original scan-msg:" << tmp.size();
  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = tmp.begin();
       item != tmp.end(); item++) {
    if (std::isnan(item->x) || std::isnan(item->y) || std::isnan(item->z) ||
        std::isinf(item->x) || std::isinf(item->y) || std::isinf(item->z))
      continue;
    p.x = (double)item->x;
    p.y = (double)item->y;
    p.z = (double)item->z;
    p.intensity = (double)item->intensity;

    r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
    if (min_scan_range < r && r < max_scan_range) {  // 5-60m
      scan.push_back(p);
    }
  }
  // test
  static int frame_cnt(1);
  LOG(INFO) << std::fixed << std::setprecision(6) << "map:" << map.size()
            << ", distance filtered:" << scan.size() << ", time:" << timestamp
            << " --cnt:" << frame_cnt++;
  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(
      new pcl::PointCloud<pcl::PointXYZI>(scan));

  // Add initial point cloud to velodyne_map
  if (initial_scan_loaded == 0) {
    Eigen::Affine3d lio_start = Eigen::Affine3d::Identity();
    if (!FindCorrespondGpsMsg(timestamp, lio_start)) {
      LOG(INFO) << "PointCloud Callback init failed!!!";
      return;
    }
    auto init_pose_matrix = init_pose.inverse() * lio_start;  //*tf_btol;
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr,
                             init_pose_matrix);
    map += *transformed_scan_ptr;
    initial_scan_loaded = 1;
  }

  // Apply voxelgrid filter
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size,
                                voxel_leaf_size);  // 1.0m
  voxel_grid_filter.setInputCloud(scan_ptr);
  voxel_grid_filter.filter(*filtered_scan_ptr);

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(
      new pcl::PointCloud<pcl::PointXYZI>(map));
  Timer elapsed_t;
  if (_method_type == MethodType::PCL_GENERIC) {
    ndt.setTransformationEpsilon(trans_eps);  // 0.01
    ndt.setStepSize(step_size);               // 0.1
    ndt.setResolution(ndt_res);               // 1
    ndt.setMaximumIterations(max_iter);       // 30
    ndt.setInputSource(filtered_scan_ptr);
  }
#ifdef USE_PCL_OPENMP
  else if (_method_type == MethodType::PCL_OPENMP) {
    omp_ndt.setTransformationEpsilon(trans_eps);
    omp_ndt.setStepSize(step_size);
    omp_ndt.setResolution(ndt_res);
    omp_ndt.setMaximumIterations(max_iter);
    omp_ndt.setInputSource(filtered_scan_ptr);
  }
#endif

  static bool is_first_map = true;
  if (is_first_map == true) {
    if (_method_type == MethodType::PCL_GENERIC)
      ndt.setInputTarget(map_ptr);
    else if (_method_type == MethodType::PCL_ANH)
      anh_ndt.setInputTarget(map_ptr);

#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP)
      omp_ndt.setInputTarget(map_ptr);
#endif
    is_first_map = false;
  }

  guess_pose.x = previous_pose.x + diff_x;
  guess_pose.y = previous_pose.y + diff_y;
  guess_pose.z = previous_pose.z + diff_z;
  guess_pose.roll = previous_pose.roll;
  guess_pose.pitch = previous_pose.pitch;
  guess_pose.yaw = previous_pose.yaw + diff_yaw;

  if (_use_imu == true && _use_odom == true)
    imu_odom_calc(current_scan_time, current_pose_imu_odom);
  if (_use_imu == true && _use_odom == false)
    imu_calc(current_scan_time, current_pose_imu);
  // this way
  // if (_use_imu == false && _use_odom == true)
  //   odom_calc(current_scan_time,current_pose_odom);

  pose guess_pose_for_ndt;
  if (_use_imu == true && _use_odom == true)
    guess_pose_for_ndt = guess_pose_imu_odom;
  else if (_use_imu == true && _use_odom == false)
    guess_pose_for_ndt = guess_pose_imu;
  // this way
  else if (_use_imu == false && _use_odom == true)
    guess_pose_for_ndt = guess_pose_odom;
  else
    guess_pose_for_ndt = guess_pose;

  Eigen::AngleAxisf init_rotation_x(guess_pose_for_ndt.roll,
                                    Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(guess_pose_for_ndt.pitch,
                                    Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(guess_pose_for_ndt.yaw,
                                    Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(
      guess_pose_for_ndt.x, guess_pose_for_ndt.y, guess_pose_for_ndt.z);

  // Eigen::Matrix4f init_guess =
  //     (init_translation * init_rotation_z * init_rotation_y *
  //     init_rotation_x).matrix() * tf_btol;

  // test
  Eigen::Affine3d real_ins = Eigen::Affine3d::Identity();
  if (!FindCorrespondGpsMsg(timestamp, real_ins)) {
    LOG(INFO) << "failed to search ins_pose at:" << std::fixed
              << std::setprecision(6) << current_scan_time;
    return;
  }
  Eigen::Matrix4f init_guess =
      (init_pose.inverse() * real_ins).matrix().cast<float>();

  t3_end = ros::Time::now();
  d3 = t3_end - t3_start;

  t4_start = ros::Time::now();

  pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  Eigen::Matrix4f t_localizer(Eigen::Matrix4f::Identity());
  if (_method_type == MethodType::PCL_GENERIC) {
    LOG(INFO) << std::fixed << std::setprecision(6)
              << "init_guess:" << init_guess(0, 3) << " " << init_guess(1, 3)
              << " " << init_guess(2, 3);
    ndt.align(*output_cloud, init_guess);
    fitness_score = ndt.getFitnessScore();
    t_localizer = ndt.getFinalTransformation();
    has_converged = ndt.hasConverged();
    final_num_iteration = ndt.getFinalNumIteration();
    transformation_probability = ndt.getTransformationProbability();

    static int cnt(1);
    LOG(INFO) << std::fixed << std::setprecision(6) << "ndt_regis map->frame:("
              << map_ptr->size() << " -> " << filtered_scan_ptr->size()
              << "),score:" << fitness_score
              << ", has_converged:" << has_converged << " ,cnt:" << cnt++;
  }

  pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_localizer);

  current_pose = Matrix2Pose(t_localizer, timestamp);
  PublishPose(current_pose_pub, current_pose, std::string("map"));

  scan_duration = current_scan_time - previous_scan_time;
  double secs = scan_duration.toSec();

  // Calculate the offset (curren_pos - previous_pos)
  diff_x = current_pose.x - previous_pose.x;
  diff_y = current_pose.y - previous_pose.y;
  diff_z = current_pose.z - previous_pose.z;
  //保证在正负不超过180
  diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
  diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

  current_velocity_x = diff_x / secs;
  current_velocity_y = diff_y / secs;
  current_velocity_z = diff_z / secs;
  {
    // useless when don't use imu
    current_pose_imu = current_pose;
    // useless when dont't use odom
    current_pose_odom = current_pose;
    // only useful when both imu and odom topic
    current_pose_imu_odom = current_pose;

    current_velocity_imu_x = current_velocity_x;
    current_velocity_imu_y = current_velocity_y;
    current_velocity_imu_z = current_velocity_z;

    offset_imu_x = 0.0;
    offset_imu_y = 0.0;
    offset_imu_z = 0.0;
    offset_imu_roll = 0.0;
    offset_imu_pitch = 0.0;
    offset_imu_yaw = 0.0;

    offset_odom_x = 0.0;
    offset_odom_y = 0.0;
    offset_odom_z = 0.0;
    offset_odom_roll = 0.0;
    offset_odom_pitch = 0.0;
    offset_odom_yaw = 0.0;

    offset_imu_odom_x = 0.0;
    offset_imu_odom_y = 0.0;
    offset_imu_odom_z = 0.0;
    offset_imu_odom_roll = 0.0;
    offset_imu_odom_pitch = 0.0;
    offset_imu_odom_yaw = 0.0;
  }

  // Update position and posture. current_pos -> previous_pos
  previous_pose = current_pose;

  previous_scan_time.sec = current_scan_time.sec;
  previous_scan_time.nsec = current_scan_time.nsec;

  // Calculate the shift between added_pos and current_pos
  double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) +
                      pow(current_pose.y - added_pose.y, 2.0));
  if (shift >= min_add_scan_shift) {  // default 1.0
    map += *transformed_scan_ptr;     //扩大地图
    added_pose = current_pose;
    if (_method_type == MethodType::PCL_GENERIC) ndt.setInputTarget(map_ptr);

#ifdef USE_PCL_OPENMP
    else if (_method_type == MethodType::PCL_OPENMP)
      omp_ndt.setInputTarget(map_ptr);
#endif
  }

  sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*map_ptr, *map_msg_ptr);
  ndt_map_pub.publish(*map_msg_ptr);
  LOG(INFO) << "elapsed time:" << elapsed_t.end() << " [ms]";

  // Write log
  {
    if (!ofs) {
      std::cerr << "Could not open " << filename << "." << std::endl;
      exit(1);
    }

    ofs << input->header.seq << "," << input->header.stamp << ","
        << input->header.frame_id << "," << scan_ptr->size() << ","
        << filtered_scan_ptr->size() << "," << std::fixed
        << std::setprecision(5) << current_pose.x << "," << std::fixed
        << std::setprecision(5) << current_pose.y << "," << std::fixed
        << std::setprecision(5) << current_pose.z << "," << current_pose.roll
        << "," << current_pose.pitch << "," << current_pose.yaw << ","
        << final_num_iteration << "," << fitness_score << "," << ndt_res << ","
        << step_size << "," << trans_eps << "," << max_iter << ","
        << voxel_leaf_size << "," << min_scan_range << "," << max_scan_range
        << "," << min_add_scan_shift << std::endl;

    std::cout
        << "-----------------------------------------------------------------"
        << std::endl;
    std::cout << "Sequence number: " << input->header.seq << std::endl;
    std::cout << "Number of scan points: " << scan_ptr->size() << " points."
              << std::endl;
    std::cout << "Number of filtered scan points: " << filtered_scan_ptr->size()
              << " points." << std::endl;
    std::cout << "transformed_scan_ptr: " << transformed_scan_ptr->points.size()
              << " points." << std::endl;
    std::cout << "map: " << map.points.size() << " points." << std::endl;
    std::cout << "NDT has converged: " << has_converged << std::endl;
    std::cout << "Fitness score: " << fitness_score << std::endl;
    std::cout << "Number of iteration: " << final_num_iteration << std::endl;
    std::cout << "(x,y,z,roll,pitch,yaw):" << std::endl;
    std::cout << "(" << current_pose.x << ", " << current_pose.y << ", "
              << current_pose.z << ", " << current_pose.roll << ", "
              << current_pose.pitch << ", " << current_pose.yaw << ")"
              << std::endl;
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << t_localizer << std::endl;
    std::cout << "shift: " << shift << std::endl;
    std::cout
        << "-----------------------------------------------------------------"
        << std::endl;
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_mapping_ndt");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Set log file name.
  char buffer[80];
  std::time_t now = std::time(NULL);
  std::tm* pnow = std::localtime(&now);
  std::strftime(buffer, 80, "%Y%m%d_%H%M%S", pnow);
  filename = "ndt_mapping_" + std::string(buffer) + ".csv";
  ofs.open(filename.c_str(), std::ios::app);

  // write header for log file
  {
    if (!ofs) {
      std::cerr << "Could not open " << filename << "." << std::endl;
      exit(1);
    }

    ofs << "input->header.seq"
        << ","
        << "input->header.stamp"
        << ","
        << "input->header.frame_id"
        << ","
        << "scan_ptr->size()"
        << ","
        << "filtered_scan_ptr->size()"
        << ","
        << "current_pose.x"
        << ","
        << "current_pose.y"
        << ","
        << "current_pose.z"
        << ","
        << "current_pose.roll"
        << ","
        << "current_pose.pitch"
        << ","
        << "current_pose.yaw"
        << ","
        << "final_num_iteration"
        << ","
        << "fitness_score"
        << ","
        << "ndt_res"
        << ","
        << "step_size"
        << ","
        << "trans_eps"
        << ","
        << "max_iter"
        << ","
        << "voxel_leaf_size"
        << ","
        << "min_scan_range"
        << ","
        << "max_scan_range"
        << ","
        << "min_add_scan_shift" << std::endl;

    // setting parameters
    int method_type_tmp = 0;
    private_nh.getParam("method_type", method_type_tmp);
    _method_type = static_cast<MethodType>(method_type_tmp);
    private_nh.getParam("use_odom", _use_odom);
    private_nh.getParam("use_imu", _use_imu);
    private_nh.getParam("imu_upside_down", _imu_upside_down);
    private_nh.getParam("imu_topic", _imu_topic);
    private_nh.getParam("incremental_voxel_update", _incremental_voxel_update);
    private_nh.getParam("config_time_threshold", config_time_threshold);
    private_nh.getParam("map_init_pos_file", map_init_pos_file);

    std::cout << "method_type: " << static_cast<int>(_method_type) << std::endl;
    std::cout << "use_odom: " << _use_odom << std::endl;
    std::cout << "use_imu: " << _use_imu << std::endl;
    std::cout << "imu_upside_down: " << _imu_upside_down << std::endl;
    std::cout << "imu_topic: " << _imu_topic << std::endl;
    std::cout << "incremental_voxel_update: " << _incremental_voxel_update
              << std::endl;
    std::cout << "config_time_threshold: " << config_time_threshold
              << std::endl;

    //   Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);  // tl: translation
    //   Eigen::AngleAxisf rot_x_btol(_tf_roll,
    //                                Eigen::Vector3f::UnitX());  // rot:
    //                                rotation
    //   Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
    //   Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
    //   tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
    //   tf_ltob = tf_btol.inverse();
  }

  map.header.frame_id = "map";

  ndt_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/ndt_map", 1000);
  current_pose_pub =
      nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 1000);

  ros::Subscriber param_sub =
      nh.subscribe("config/ndt_mapping", 10, param_callback);

  ros::Subscriber odom_sub =
      nh.subscribe("/vehicle/odom", 100000, odom_callback);
  ros::Subscriber imu_sub = nh.subscribe(_imu_topic, 100000, imu_callback);

  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 10, gnss_callback);
  ros::Subscriber points_sub =
      nh.subscribe("points_raw", 100000, points_callback);
  ros::Subscriber output_sub =
      nh.subscribe("config/ndt_mapping_output", 10, output_callback);
  ros::spin();

  return 0;
}
