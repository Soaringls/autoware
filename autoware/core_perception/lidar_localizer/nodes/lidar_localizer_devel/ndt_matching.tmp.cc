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
 Localization program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#include <autoware_config_msgs/ConfigNDT.h>
#include <autoware_msgs/NDTStat.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pthread.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>

// headers in Autoware Health Checker
// #include <autoware_health_checker/health_checker/health_checker.h>

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

// static std::shared_ptr<autoware_health_checker::HealthChecker>
//     health_checker_ptr_;

struct pose {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

enum class MethodType {
  PCL_GENERIC = 0,
  PCL_ANH = 1,
  PCL_ANH_GPU = 2,
  PCL_OPENMP = 3,
};
static MethodType _method_type = MethodType::PCL_GENERIC;

double config_time_threshold = 0.02;
std::mutex gps_mutex;
std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;

static pose predict_pose, predict_pose_imu, predict_pose_odom,
    predict_pose_imu_odom, previous_pose, current_pose, localizer_pose;

static double offset_x, offset_y, offset_z,
    offset_yaw;  // current_pos - previous_pose

// Can't load if typed "pcl::PointCloud<pcl::PointXYZRGB> map, add;"
static pcl::PointCloud<pcl::PointXYZ> map, add;

// If the map is loaded, map_loaded will be 1.
static int map_loaded = 0;
static int _use_gnss = 1;
static int init_pos_set = 0;
static Eigen::Vector3d map_init_pos = Eigen::Vector3d(0, 0, 0);
static Eigen::Affine3d map_init_pose = Eigen::Affine3d::Identity();

static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
static cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> anh_ndt;

// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon

static ros::Publisher predict_pose_odom_pub;
static geometry_msgs::PoseStamped predict_pose_odom_msg;

static geometry_msgs::PoseStamped predict_pose_imu_odom_msg;

static geometry_msgs::PoseStamped ndt_pose_msg;

static ros::Publisher gps_path_pub;
static ros::Publisher lio_path_pub;

static geometry_msgs::PoseStamped localizer_pose_msg;

static geometry_msgs::TwistStamped estimate_twist_msg;

static ros::Duration scan_duration;

static double exe_time = 0.0;
static bool has_converged;
static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0;

// reference for comparing fitness_score, default value set to 500.0
static double _gnss_reinit_fitness = 500.0;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;

static double current_velocity = 0.0, previous_velocity = 0.0,
              previous_previous_velocity = 0.0;  // [m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
// static double current_velocity_yaw = 0.0, previous_velocity_yaw = 0.0;
static double current_velocity_smooth = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;

static double current_accel = 0.0, previous_accel = 0.0;  // [m/s^2]
static double current_accel_x = 0.0;
static double current_accel_y = 0.0;
static double current_accel_z = 0.0;
// static double current_accel_yaw = 0.0;

static double angular_velocity = 0.0;

static int use_predict_pose = 0;

static std_msgs::Float32 estimated_vel_mps, estimated_vel_kmph,
    previous_estimated_vel_kmph;

static std::chrono::time_point<std::chrono::system_clock> matching_start,
    matching_end;

static std_msgs::Float32 time_ndt_matching;

static int _queue_size = 1000;

static autoware_msgs::NDTStat ndt_stat_msg;

static double predict_pose_error = 0.0;

static double _tf_x, _tf_y, _tf_z, _tf_roll, _tf_pitch, _tf_yaw;
static Eigen::Matrix4f tf_btol = Eigen::Matrix4f::Identity();
static Eigen::Affine3d init_pose = Eigen::Affine3d::Identity();

static std::string _localizer = "velodyne";
static std::string _offset = "linear";  // linear, zero, quadratic

static std_msgs::Float32 ndt_reliability;

static bool _get_height = false;
static bool _use_local_transform = false;
static bool _use_imu = false;
static bool _use_odom = false;
static bool _imu_upside_down = false;
static bool _output_log_data = false;

static std::string _imu_topic = "/imu_raw";

static std::ofstream ofs;
static std::string filename;

static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

// static tf::TransformListener local_transform_listener;
static tf::StampedTransform local_transform;

static unsigned int points_map_num = 0;

pthread_mutex_t mutex;

void SetMapInitPos(const std::string& map_init_filename) {
  if (!boost::filesystem::exists(map_init_filename)) {
    LOG(FATAL) << "file:" << map_init_filename << " is not exist!";
  }
  std::ifstream ifs(map_init_filename, std::ios::in);
  std::string line = "";
  while (getline(ifs, line)) {
    if (line.empty()) continue;
    std::vector<std::string> parts;
    boost::split(parts, line, boost::is_any_of(" "));
    if (parts.empty()) {
      LOG(FATAL) << "there is empty in the file!";
    }
    // map_init_pos = Eigen::Vector3d(std::stod(parts[0]), std::stod(parts[1]),
    // std::stod(parts[2]));
    Eigen::Quaterniond q(std::stod(parts[3]), std::stod(parts[4]),
                         std::stod(parts[5]), std::stod(parts[6]));

    map_init_pose =
        Eigen::Translation3d(std::stod(parts[0]), std::stod(parts[1]),
                             std::stod(parts[2])) *
        q.normalized();
  }
  ifs.close();
}

void AddPoseToNavPath(nav_msgs::Path& path, const pose& pose,
                      std::string frame_id = "world") {
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time::now();
  pose_stamped.header.frame_id = frame_id;
  pose_stamped.pose.position.x =
      pose.x - init_pose.translation().x();  // init_pos_.x();
  pose_stamped.pose.position.y =
      pose.y - init_pose.translation().y();  // init_pos_.y();
  pose_stamped.pose.position.z =
      pose.z - init_pose.translation().z();  // init_pos_.z();
  pose_stamped.pose.orientation.w = 0;       // pose->q.w();
  pose_stamped.pose.orientation.x = 0;       // pose->q.x();
  pose_stamped.pose.orientation.y = 0;       // pose->q.y();
  pose_stamped.pose.orientation.z = 0;       // pose->q.z();
  path.header = pose_stamped.header;
  path.poses.push_back(pose_stamped);
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

static pose convertPoseIntoRelativeCoordinate(const pose& target_pose,
                                              const pose& reference_pose) {
  tf::Quaternion target_q;
  target_q.setRPY(target_pose.roll, target_pose.pitch, target_pose.yaw);
  tf::Vector3 target_v(target_pose.x, target_pose.y, target_pose.z);
  tf::Transform target_tf(target_q, target_v);

  tf::Quaternion reference_q;
  reference_q.setRPY(reference_pose.roll, reference_pose.pitch,
                     reference_pose.yaw);
  tf::Vector3 reference_v(reference_pose.x, reference_pose.y, reference_pose.z);
  tf::Transform reference_tf(reference_q, reference_v);

  tf::Transform trans_target_tf = reference_tf.inverse() * target_tf;

  pose trans_target_pose;
  trans_target_pose.x = trans_target_tf.getOrigin().getX();
  trans_target_pose.y = trans_target_tf.getOrigin().getY();
  trans_target_pose.z = trans_target_tf.getOrigin().getZ();
  tf::Matrix3x3 tmp_m(trans_target_tf.getRotation());
  tmp_m.getRPY(trans_target_pose.roll, trans_target_pose.pitch,
               trans_target_pose.yaw);

  return trans_target_pose;
}

static void map_callback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  // if (map_loaded == 0)
  if (points_map_num != input->width) {
    std::cout << "Update points_map." << std::endl;

    points_map_num = input->width;

    // Convert the data type(from sensor_msgs to pcl).
    pcl::fromROSMsg(*input, map);
    // for(const auto& pt : map.points){
    //   std::cout<<"pt: x:"<<pt.x<<" y:"<<pt.y<<" pt.z:"<<pt.z<<std::endl;
    // }
    if (_use_local_transform == true) {
      tf::TransformListener local_transform_listener;
      try {
        ros::Time now = ros::Time(0);
        local_transform_listener.waitForTransform("/map", "/world", now,
                                                  ros::Duration(10.0));
        local_transform_listener.lookupTransform("/map", "world", now,
                                                 local_transform);
      } catch (tf::TransformException& ex) {
        ROS_ERROR("%s", ex.what());
      }

      pcl_ros::transformPointCloud(map, map, local_transform.inverse());
    }

    // temp
    for (int i = 0; i < 20; i++) {
      LOG(INFO) << std::fixed << std::setprecision(6)
                << "evaleate before:" << map.points[i].x << ", "
                << map.points[i].y;
    }
    auto t = map_init_pose.translation();
    LOG(INFO) << std::fixed << std::setprecision(6) << "map_init:" << t.x()
              << ", " << t.y();
    pcl::transformPointCloud(map, map, map_init_pose.matrix());
    // for(auto& pt : map.points){
    //   pt.x += map_init_pos[0];
    //   pt.y += map_init_pos[1];
    //   pt.z += map_init_pos[2];
    // }
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_ptr(
        new pcl::PointCloud<pcl::PointXYZ>(map));
    for (int i = 0; i < 20; i++) {
      LOG(INFO) << std::fixed << std::setprecision(6)
                << "evaleate after:" << map_ptr->points[i].x << ", "
                << map_ptr->points[i].y;
    }

    // Setting point cloud to be aligned to.
    if (_method_type == MethodType::PCL_GENERIC) {
      pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> new_ndt;
      pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      new_ndt.setResolution(ndt_res);
      new_ndt.setInputTarget(map_ptr);
      new_ndt.setMaximumIterations(max_iter);
      new_ndt.setStepSize(step_size);
      new_ndt.setTransformationEpsilon(trans_eps);

      new_ndt.align(*output_cloud, Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      ndt = new_ndt;
      pthread_mutex_unlock(&mutex);
    } else if (_method_type == MethodType::PCL_ANH) {
      cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>
          new_anh_ndt;
      new_anh_ndt.setResolution(ndt_res);
      new_anh_ndt.setInputTarget(map_ptr);
      new_anh_ndt.setMaximumIterations(max_iter);
      new_anh_ndt.setStepSize(step_size);
      new_anh_ndt.setTransformationEpsilon(trans_eps);

      pcl::PointCloud<pcl::PointXYZ>::Ptr dummy_scan_ptr(
          new pcl::PointCloud<pcl::PointXYZ>());
      pcl::PointXYZ dummy_point;
      dummy_scan_ptr->push_back(dummy_point);
      new_anh_ndt.setInputSource(dummy_scan_ptr);

      new_anh_ndt.align(Eigen::Matrix4f::Identity());

      pthread_mutex_lock(&mutex);
      anh_ndt = new_anh_ndt;
      pthread_mutex_unlock(&mutex);
    }
    map_loaded = 1;
  }
}

// INIT_init_pos_set = 1; callback of topic:"gnss_pose"
static void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& input) {
  static bool init_flag = false;
  gps_mutex.lock();
  if (gps_msgs.size() > 500) gps_msgs.pop();
  gps_msgs.push(input);
  gps_mutex.unlock();

  if (!init_flag) {
    init_pose =
        Eigen::Translation3d(input->pose.position.x, input->pose.position.y,
                             input->pose.position.z) *
        Eigen::Quaterniond(input->pose.orientation.w, input->pose.orientation.x,
                           input->pose.orientation.y,
                           input->pose.orientation.z);
    init_flag = true;
    // return;
  }

  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y,
                        input->pose.orientation.z, input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);

  pose current_gnss_pose;
  current_gnss_pose.x = input->pose.position.x;
  current_gnss_pose.y = input->pose.position.y;
  current_gnss_pose.z = input->pose.position.z;
  gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch,
                current_gnss_pose.yaw);

  static nav_msgs::Path ins_path;
  AddPoseToNavPath(ins_path, current_gnss_pose);
  gps_path_pub.publish(ins_path);  // pub ins path

  static pose previous_gnss_pose = current_gnss_pose;
  ros::Time current_gnss_time = input->header.stamp;
  static ros::Time previous_gnss_time = current_gnss_time;

  if ((_use_gnss == 1 && init_pos_set == 0) ||
      fitness_score >= _gnss_reinit_fitness) {
    previous_pose.x = previous_gnss_pose.x;
    previous_pose.y = previous_gnss_pose.y;
    previous_pose.z = previous_gnss_pose.z;
    previous_pose.roll = previous_gnss_pose.roll;
    previous_pose.pitch = previous_gnss_pose.pitch;
    previous_pose.yaw = previous_gnss_pose.yaw;

    current_pose.x = current_gnss_pose.x;
    current_pose.y = current_gnss_pose.y;
    current_pose.z = current_gnss_pose.z;
    current_pose.roll = current_gnss_pose.roll;
    current_pose.pitch = current_gnss_pose.pitch;
    current_pose.yaw = current_gnss_pose.yaw;

    diff_x = current_pose.x - previous_pose.x;
    diff_y = current_pose.y - previous_pose.y;
    diff_z = current_pose.z - previous_pose.z;
    diff_yaw = current_pose.yaw - previous_pose.yaw;
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    // current_pose相对previous_pose的pose
    const pose trans_current_pose =
        convertPoseIntoRelativeCoordinate(current_pose, previous_pose);

    const double diff_time = (current_gnss_time - previous_gnss_time).toSec();
    current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
    // trans_current_pose.x 前进方向   ros系: red-x  green-y blue-z
    current_velocity =
        (trans_current_pose.x >= 0) ? current_velocity : -current_velocity;
    current_velocity_x = (diff_time > 0) ? (diff_x / diff_time) : 0;
    current_velocity_y = (diff_time > 0) ? (diff_y / diff_time) : 0;
    current_velocity_z = (diff_time > 0) ? (diff_z / diff_time) : 0;
    angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

    current_accel = 0.0;
    current_accel_x = 0.0;
    current_accel_y = 0.0;
    current_accel_z = 0.0;

    init_pos_set = 1;
  }

  previous_gnss_pose.x = current_gnss_pose.x;
  previous_gnss_pose.y = current_gnss_pose.y;
  previous_gnss_pose.z = current_gnss_pose.z;
  previous_gnss_pose.roll = current_gnss_pose.roll;
  previous_gnss_pose.pitch = current_gnss_pose.pitch;
  previous_gnss_pose.yaw = current_gnss_pose.yaw;
  previous_gnss_time = current_gnss_time;
}

static double calcDiffForRadian(const double lhs_rad, const double rhs_rad) {
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr FilterCloudFrame(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& input, const double leaf_size = 1) {
  // nan filter
  std::vector<int> indices;
  // LOG(INFO)<<"filter_nan=====1:"<<input->size();
  pcl::removeNaNFromPointCloud<pcl::PointXYZ>(*input, *input, indices);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < input->size(); ++i) {
    auto pt = input->points[i];
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
        std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z))
      continue;
    filtered->push_back(pt);
  }
  // voxel filter
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered_pts(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.setInputCloud(filtered);
  voxel_filter.filter(*voxel_filtered_pts);
  // distance filter
  filtered->clear();
  filtered->reserve(voxel_filtered_pts->size());
  std::copy_if(voxel_filtered_pts->begin(), voxel_filtered_pts->end(),
               std::back_inserter(filtered->points),
               [&](const pcl::PointXYZ& pt) {
                 double dist = pt.getVector3fMap().norm();
                 return dist > 1 && dist < 50;
               });

  filtered->header = input->header;
  // SetCloudAttributes(filtered);

  // Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  // transform.rotate(
  //     Eigen::AngleAxisd(90 * M_PI / 180, Eigen::Vector3d::UnitZ()));
  // pcl::transformPointCloud(*filtered, *filtered, transform);
  return filtered;
}

static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  if (map_loaded == 1 && init_pos_set == 1) {
    matching_start = std::chrono::system_clock::now();

    static tf::TransformBroadcaster br;
    tf::Quaternion predict_q, ndt_q, current_q, localizer_q;

    pcl::PointXYZ p;
    pcl::PointCloud<pcl::PointXYZ> filtered_scan;

    ros::Time current_scan_time = input->header.stamp;
    double timestamp = input->header.stamp.toSec();
    static ros::Time previous_scan_time = current_scan_time;

    pcl::fromROSMsg(*input, filtered_scan);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(
        new pcl::PointCloud<pcl::PointXYZ>(filtered_scan));
    filtered_scan_ptr = FilterCloudFrame(filtered_scan_ptr);

    int scan_points_num = filtered_scan_ptr->size();

    Eigen::Matrix4f t(Eigen::Matrix4f::Identity());   // base_link
    Eigen::Matrix4f t2(Eigen::Matrix4f::Identity());  // localizer

    std::chrono::time_point<std::chrono::system_clock> align_start, align_end,
        getFitnessScore_start, getFitnessScore_end;
    static double align_time, getFitnessScore_time = 0.0;

    pthread_mutex_lock(&mutex);

    if (_method_type == MethodType::PCL_GENERIC)
      ndt.setInputSource(filtered_scan_ptr);
    else if (_method_type == MethodType::PCL_ANH)
      anh_ndt.setInputSource(filtered_scan_ptr);

    // Guess the initial gross estimation of the transformation
    double diff_time = (current_scan_time - previous_scan_time).toSec();

    if (_offset == "linear") {
      offset_x = current_velocity_x * diff_time;
      offset_y = current_velocity_y * diff_time;
      offset_z = current_velocity_z * diff_time;
      offset_yaw = angular_velocity * diff_time;
    }

    predict_pose.x = previous_pose.x + offset_x;
    predict_pose.y = previous_pose.y + offset_y;
    predict_pose.z = previous_pose.z + offset_z;
    predict_pose.roll = previous_pose.roll;
    predict_pose.pitch = previous_pose.pitch;
    predict_pose.yaw = previous_pose.yaw + offset_yaw;

    pose predict_pose_for_ndt = predict_pose;

    Eigen::Translation3f init_translation(
        predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
    Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll,
                                      Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch,
                                      Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw,
                                      Eigen::Vector3f::UnitZ());
    // Eigen::Matrix4f init_guess = (init_translation * init_rotation_z *
    //                               init_rotation_y * init_rotation_x) *
    //                              tf_btol;

    // test
    Eigen::Affine3d real_ins = Eigen::Affine3d::Identity();
    static Eigen::Affine3d last_ins;
    if (!FindCorrespondGpsMsg(timestamp, real_ins)) {
      real_ins = last_ins;
    }
    last_ins = real_ins;
    Eigen::Matrix4f init_guess = real_ins.matrix().cast<float>();
    // test over

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    if (_method_type == MethodType::PCL_GENERIC) {
      align_start = std::chrono::system_clock::now();
      // init_guess = Eigen::Matrix4f::Identity(); //test  score is too large
      ndt.align(*output_cloud, init_guess);
      align_end = std::chrono::system_clock::now();

      has_converged = ndt.hasConverged();

      t = ndt.getFinalTransformation();
      iteration = ndt.getFinalNumIteration();

      getFitnessScore_start = std::chrono::system_clock::now();
      fitness_score = ndt.getFitnessScore();
      getFitnessScore_end = std::chrono::system_clock::now();

      trans_probability = ndt.getTransformationProbability();

      LOG(INFO) << std::fixed << std::setprecision(6)
                << "init guess:" << init_guess(0, 3) << ", " << init_guess(1, 3)
                << ", score:" << fitness_score;
      LOG(INFO) << std::fixed << std::setprecision(6)
                << "ndt result:" << t(0, 3) << ", " << t(1, 3);
    }
    align_time = std::chrono::duration_cast<std::chrono::microseconds>(
                     align_end - align_start)
                     .count() /
                 1000.0;

    t2 = t * tf_btol.inverse();  // tf_btol.inverse()为pre_pose   t2:cur_pose

    getFitnessScore_time =
        std::chrono::duration_cast<std::chrono::microseconds>(
            getFitnessScore_end - getFitnessScore_start)
            .count() /
        1000.0;

    pthread_mutex_unlock(&mutex);

    tf::Matrix3x3 mat_l;  // localizer_pose == t 即 tf = cur/pre
    mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)),
                   static_cast<double>(t(0, 2)), static_cast<double>(t(1, 0)),
                   static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                   static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)),
                   static_cast<double>(t(2, 2)));

    // Update localizer_pose
    localizer_pose.x = t(0, 3);
    localizer_pose.y = t(1, 3);
    localizer_pose.z = t(2, 3);
    mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw,
                 1);

    static nav_msgs::Path lio_path;
    AddPoseToNavPath(lio_path, localizer_pose);
    lio_path_pub.publish(lio_path);  // pub lio path
    //////////////////////////////////////////////////

    // Calculate the difference between localizer_pose and predict_pose
    predict_pose_error = sqrt((localizer_pose.x - predict_pose_for_ndt.x) *
                                  (localizer_pose.x - predict_pose_for_ndt.x) +
                              (localizer_pose.y - predict_pose_for_ndt.y) *
                                  (localizer_pose.y - predict_pose_for_ndt.y) +
                              (localizer_pose.z - predict_pose_for_ndt.z) *
                                  (localizer_pose.z - predict_pose_for_ndt.z));

    if (predict_pose_error <= PREDICT_POSE_THRESHOLD)  // threshold 0.5
    {
      use_predict_pose = 0;
    } else {
      use_predict_pose = 1;
    }
    use_predict_pose = 0;

    if (use_predict_pose == 0) {
      current_pose.x = localizer_pose.x;
      current_pose.y = localizer_pose.y;
      current_pose.z = localizer_pose.z;
      current_pose.roll = localizer_pose.roll;
      current_pose.pitch = localizer_pose.pitch;
      current_pose.yaw = localizer_pose.yaw;
    } else {
      current_pose.x = predict_pose_for_ndt.x;
      current_pose.y = predict_pose_for_ndt.y;
      current_pose.z = predict_pose_for_ndt.z;
      current_pose.roll = predict_pose_for_ndt.roll;
      current_pose.pitch = predict_pose_for_ndt.pitch;
      current_pose.yaw = predict_pose_for_ndt.yaw;
    }

    // Compute the velocity and acceleration
    diff_x = current_pose.x - previous_pose.x;
    diff_y = current_pose.y - previous_pose.y;
    diff_z = current_pose.z - previous_pose.z;
    diff_yaw = calcDiffForRadian(current_pose.yaw, previous_pose.yaw);
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    const pose trans_current_pose =
        convertPoseIntoRelativeCoordinate(current_pose, previous_pose);

    current_velocity = (diff_time > 0) ? (diff / diff_time) : 0;
    current_velocity =
        (trans_current_pose.x >= 0) ? current_velocity : -current_velocity;
    current_velocity_x = (diff_time > 0) ? (diff_x / diff_time) : 0;
    current_velocity_y = (diff_time > 0) ? (diff_y / diff_time) : 0;
    current_velocity_z = (diff_time > 0) ? (diff_z / diff_time) : 0;
    angular_velocity = (diff_time > 0) ? (diff_yaw / diff_time) : 0;

    current_velocity_imu_x = current_velocity_x;
    current_velocity_imu_y = current_velocity_y;
    current_velocity_imu_z = current_velocity_z;

    current_velocity_smooth =
        (current_velocity + previous_velocity + previous_previous_velocity) /
        3.0;
    if (std::fabs(current_velocity_smooth) < 0.2) {
      current_velocity_smooth = 0.0;
    }

    current_accel = (diff_time > 0)
                        ? ((current_velocity - previous_velocity) / diff_time)
                        : 0;
    current_accel_x =
        (diff_time > 0)
            ? ((current_velocity_x - previous_velocity_x) / diff_time)
            : 0;
    current_accel_y =
        (diff_time > 0)
            ? ((current_velocity_y - previous_velocity_y) / diff_time)
            : 0;
    current_accel_z =
        (diff_time > 0)
            ? ((current_velocity_z - previous_velocity_z) / diff_time)
            : 0;

    // Set values for publishing pose
    predict_q.setRPY(predict_pose.roll, predict_pose.pitch, predict_pose.yaw);

    ndt_q.setRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw);

    current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

    localizer_q.setRPY(localizer_pose.roll, localizer_pose.pitch,
                       localizer_pose.yaw);

    matching_end = std::chrono::system_clock::now();
    exe_time = std::chrono::duration_cast<std::chrono::microseconds>(
                   matching_end - matching_start)
                   .count() /
               1000.0;
    time_ndt_matching.data = exe_time;

    // Update previous_***
    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;

    previous_scan_time = current_scan_time;

    previous_previous_velocity = previous_velocity;
    previous_velocity = current_velocity;
    previous_velocity_x = current_velocity_x;
    previous_velocity_y = current_velocity_y;
    previous_velocity_z = current_velocity_z;
    previous_accel = current_accel;

    previous_estimated_vel_kmph.data = estimated_vel_kmph.data;
  }
}

void* thread_func(void* args) {
  ros::NodeHandle nh_map;
  ros::CallbackQueue map_callback_queue;
  nh_map.setCallbackQueue(&map_callback_queue);

  ros::Subscriber map_sub = nh_map.subscribe("points_map", 10, map_callback);
  ros::Rate ros_rate(10);
  while (nh_map.ok()) {
    map_callback_queue.callAvailable(ros::WallDuration());
    ros_rate.sleep();
  }

  return nullptr;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_matching_ndt");
  pthread_mutex_init(&mutex, NULL);

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Geting parameters
  int method_type_tmp = 0;
  private_nh.getParam("method_type", method_type_tmp);
  _method_type = static_cast<MethodType>(method_type_tmp);
  private_nh.getParam("use_gnss", _use_gnss);
  private_nh.getParam("queue_size", _queue_size);
  private_nh.getParam("offset", _offset);
  private_nh.getParam("get_height", _get_height);
  private_nh.getParam("use_local_transform", _use_local_transform);
  private_nh.getParam("use_imu", _use_imu);
  private_nh.getParam("use_odom", _use_odom);
  private_nh.getParam("imu_upside_down", _imu_upside_down);
  private_nh.getParam("imu_topic", _imu_topic);
  private_nh.param<double>("gnss_reinit_fitness", _gnss_reinit_fitness, 500.0);

  std::string map_init_config_file;
  private_nh.getParam("map_init_config", map_init_config_file);
  SetMapInitPos(map_init_config_file);

  std::cout
      << "-----------------------------------------------------------------"
      << std::endl;
  std::cout << "Log file: " << filename << std::endl;
  std::cout << "method_type: " << static_cast<int>(_method_type) << std::endl;
  std::cout << "use_gnss: " << _use_gnss << std::endl;
  std::cout << "queue_size: " << _queue_size << std::endl;
  std::cout << "offset: " << _offset << std::endl;
  std::cout << "get_height: " << _get_height << std::endl;
  std::cout << "use_local_transform: " << _use_local_transform << std::endl;
  std::cout << "use_odom: " << _use_odom << std::endl;
  std::cout << "use_imu: " << _use_imu << std::endl;
  std::cout << "imu_upside_down: " << _imu_upside_down << std::endl;
  std::cout << "imu_topic: " << _imu_topic << std::endl;
  std::cout << "localizer: " << _localizer << std::endl;
  std::cout << "gnss_reinit_fitness: " << _gnss_reinit_fitness << std::endl;

  // Subscribers
  ros::Subscriber gnss_sub = nh.subscribe("gnss_pose", 10, gnss_callback);
  ros::Subscriber points_sub =
      nh.subscribe("/points_raw", _queue_size, points_callback);

  gps_path_pub = nh.advertise<nav_msgs::Path>("/lidar_locator/path/ins", 100);
  lio_path_pub = nh.advertise<nav_msgs::Path>("/lidar_locator/path/lio", 100);

  pthread_t thread;
  pthread_create(&thread, NULL, thread_func, NULL);

  ros::spin();

  return 0;
}
