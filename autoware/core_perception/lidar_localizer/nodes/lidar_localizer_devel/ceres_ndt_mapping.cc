

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
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

#include "ceres_impl/ceres_optimizer.h"

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

ros::Publisher pub_scan, pub_map;
ros::Publisher ins_pos_pub, odom_pos_pub, optimized_pos_pub;

std::mutex gps_mutex, keyframe_mutex;
std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;

CeresOptimizer ceres_optimizer;
static Eigen::Affine3d odom_init = Eigen::Affine3d::Identity();

static pcl::PointCloud<pcl::PointXYZI> map_cloud;

static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

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
} config;

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointCloudPtr cloud;
  Eigen::Isometry3d pose;
  // Eigen::Affine3d pose;
  double stamp = 0.0;
  KeyFrame() {}
  KeyFrame(const double& stamp, const Eigen::Isometry3d& pose,
           const PointCloudPtr& cloud)
      : stamp(stamp), pose(pose), cloud(cloud) {}
};
typedef std::shared_ptr<KeyFrame> KeyFramePtr;
typedef std::shared_ptr<const KeyFrame> KeyFrameConstPtr;
std::vector<KeyFramePtr> key_frame_ptrs;

void PublishPose(ros::Publisher& pub, const Pose pose, std::string frame_id) {
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
void SetCloudAttributes(PointCloudPtr& cloud, std::string frame_id = "world") {
  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->header.frame_id = frame_id;
}

PointCloudPtr VoxelFilter(const PointCloudPtr& cloud, const double size) {
  // voxel filter
  auto pts = cloud->size();
  PointCloudPtr filtered_scan_ptr(new PointCloud());
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(size, size, size);
  voxel_grid_filter.setInputCloud(cloud);
  voxel_grid_filter.filter(*filtered_scan_ptr);
  filtered_scan_ptr->header = cloud->header;
  LOG(INFO) << "origin:" << pts << ", filtered:" << filtered_scan_ptr->size();
  return filtered_scan_ptr;
}

Pose Matrix2Pose(const Eigen::Matrix4f matrix, const double stamp) {
  tf::Matrix3x3 mat_tf;
  mat_tf.setValue(
      static_cast<double>(matrix(0, 0)), static_cast<double>(matrix(0, 1)),
      static_cast<double>(matrix(0, 2)), static_cast<double>(matrix(1, 0)),
      static_cast<double>(matrix(1, 1)), static_cast<double>(matrix(1, 2)),
      static_cast<double>(matrix(2, 0)), static_cast<double>(matrix(2, 1)),
      static_cast<double>(matrix(2, 2)));
  Pose result;
  result.stamp = stamp;
  result.x = matrix(0, 3);
  result.y = matrix(1, 3);
  result.z = matrix(2, 3);
  mat_tf.getRPY(result.roll, result.pitch, result.yaw, 1);

  return result;
}

static double calcDiffForRadian(const double lhs_rad, const double rhs_rad) {
  double diff_rad = lhs_rad - rhs_rad;
  if (diff_rad >= M_PI)
    diff_rad = diff_rad - 2 * M_PI;
  else if (diff_rad < -M_PI)
    diff_rad = diff_rad + 2 * M_PI;
  return diff_rad;
}

Eigen::Matrix4f TransformPoseToEigenMatrix4f(const Pose& ref) {
  Eigen::AngleAxisf x_rotation_vec(ref.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf y_rotation_vec(ref.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf z_rotation_vec(ref.yaw, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f t(ref.x, ref.y, ref.z);
  return (t * z_rotation_vec * y_rotation_vec * x_rotation_vec).matrix();
}

void ClearQueue(std::queue<geometry_msgs::PoseStamped::ConstPtr>& queue) {
  std::queue<geometry_msgs::PoseStamped::ConstPtr> empty;
  std::swap(queue, empty);
}

bool FindCorrespondGpsMsg(const double pts_stamp, Eigen::Affine3d& ins_pose,
                          const double& threshold = config.time_threshold) {
  bool is_find(false);
  gps_mutex.lock();
  while (!gps_msgs.empty()) {
    auto stamp_diff = gps_msgs.front()->header.stamp.toSec() - pts_stamp;
    if (std::abs(stamp_diff) <= threshold) {
      auto gps_msg = gps_msgs.front();
      ins_pose = Eigen::Translation3d(gps_msg->pose.position.x,
                                      gps_msg->pose.position.y,
                                      gps_msg->pose.position.z) *
                 Eigen::Quaterniond(
                     gps_msg->pose.orientation.w, gps_msg->pose.orientation.x,
                     gps_msg->pose.orientation.y, gps_msg->pose.orientation.z);
      // gps_msgs.pop();
      is_find = true;
      ClearQueue(gps_msgs);
      break;
    } else if (stamp_diff < -threshold) {
      gps_msgs.pop();
    } else if (stamp_diff > threshold) {
      LOG(INFO) << "(gps_time - pts_time = " << stamp_diff << " > " << threshold
                << ") lidar msgs is delayed! ";
      // ClearQueue(gps_msgs);
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

static void InsCallback(const geometry_msgs::PoseStamped::ConstPtr& input) {
  gps_mutex.lock();
  // if (gps_msgs.size() > 30000) gps_msgs.pop();
  gps_msgs.push(input);
  gps_mutex.unlock();

  POSEPtr ins_factor(new POSE);
  ins_factor->time = input->header.stamp.toSec();
  ins_factor->pos = Eigen::Vector3d(
      input->pose.position.x, input->pose.position.y, input->pose.position.z);
  ins_factor->q =
      Eigen::Quaterniond(input->pose.orientation.w, input->pose.orientation.x,
                         input->pose.orientation.y, input->pose.orientation.z);

  static nav_msgs::Path gps_path;
  ceres_optimizer.AddPoseToNavPath(gps_path, ins_factor);
  ins_pos_pub.publish(gps_path);
}
void FilterByDist(const PointCloud& raw_scan, PointCloudPtr& output) {
  LOG(INFO) << "original raw_scan-msg:" << raw_scan.size();
  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = raw_scan.begin();
       item != raw_scan.end(); item++) {
    if (std::isnan(item->x) || std::isnan(item->y) || std::isnan(item->z) ||
        std::isinf(item->x) || std::isinf(item->y) || std::isinf(item->z))
      continue;
    PointT pt;
    pt.x = (double)item->x;
    pt.y = (double)item->y;
    pt.z = (double)item->z;
    pt.intensity = (double)item->intensity;

    auto r = sqrt(pow(pt.x, 2.0) + pow(pt.y, 2.0));
    if (config.min_scan_range < r && r < config.max_scan_range) {  // 5-60m
      output->push_back(pt);
    }
  }
}

void AddKeyFrame(const double& time, const Eigen::Matrix4d& tf,
                 const Eigen::Matrix4d& pose, const PointCloudPtr cloud) {
  // publish current frame
  PointCloudPtr current_frame(new PointCloud);
  Eigen::Matrix4d visual_pose = pose;
  visual_pose = odom_init.matrix().inverse() * visual_pose;
  pcl::transformPointCloud(*cloud, *current_frame, visual_pose);
  sensor_msgs::PointCloud2::Ptr output(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(*current_frame, *output);
  pub_scan.publish(output);

  // check if the condition is meet
  static Eigen::Matrix4d pre_keyframe_pos = pose;
  static Eigen::Matrix4d frame_pose = pose;
  frame_pose *= tf;
  auto matrix_dist = pre_keyframe_pos.inverse() * frame_pose;
  auto distance = matrix_dist.block<3, 1>(0, 3).norm();
  if (distance < config.keyframe_delta_trans) {
    LOG(INFO) << "Add KeyFrame failed. the distance to last keyframe:"
              << distance << ", threshold:" << config.keyframe_delta_trans;
    return;
  }
  pre_keyframe_pos = pose;
  frame_pose = pose;

  CHECK_NOTNULL(cloud);
  Eigen::Isometry3d key_pose = Eigen::Isometry3d::Identity();

  key_pose.linear() = pose.block<3, 3>(0, 0);
  key_pose.translation() = pose.block<3, 1>(0, 3);
  KeyFramePtr key_frame(new KeyFrame(time, key_pose, cloud));
  keyframe_mutex.lock();
  key_frame_ptrs.push_back(key_frame);
  keyframe_mutex.unlock();
  LOG(INFO) << "add keyframe's dist:" << std::fixed << std::setprecision(6)
            << distance << ", threshold:" << config.keyframe_delta_trans
            << ", at pos: " << pose(0, 3) << ", " << pose(1, 3) << ", "
            << pose(2, 3);
}

bool Align(const PointCloudPtr scan_ptr, const double stamp,
           Eigen::Matrix4f& matrix_output, double& score) {
  // init map
  static bool init_scan_loaded = false;
  if (!init_scan_loaded) {
    if (!FindCorrespondGpsMsg(stamp, odom_init, config.time_threshold / 2)) {
      LOG(INFO) << "PointCloud Callback init failed!!!";
      return false;
    }
    map_cloud += *scan_ptr;
    init_scan_loaded = true;

    POSEPtr odom_init_factor = Affine2POSEPtr(odom_init, stamp);
    ceres_optimizer.InsertOdom(odom_init_factor);
    ceres_optimizer.InsertGPS(odom_init_factor);
    return true;
  }
  // find the init guess pose
  static Eigen::Affine3d last_ins = Eigen::Affine3d::Identity();
  Eigen::Affine3d real_ins = Eigen::Affine3d::Identity();
  if (!FindCorrespondGpsMsg(stamp, real_ins)) {
    LOG(INFO) << "failed to search ins_pose at:" << std::fixed
              << std::setprecision(6) << stamp;
    return false;
    real_ins = last_ins;
  }
  last_ins = real_ins;

  static int cnt(1);
  // insert ins-factor
  if (cnt++ % config.ceres_config.num_every_scans == 0) {
    // if (cnt % 2 == 0) {
    LOG(INFO) << "Anchor is to correct the pose....";
    ceres_optimizer.InsertGPS(Affine2POSEPtr(real_ins, stamp));
  }

  // filter scan-msg
  // PointCloudPtr filtered_scan_ptr(new PointCloud());
  // pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  // voxel_grid_filter.setLeafSize(config.voxel_leaf_size,
  // config.voxel_leaf_size,
  //                               config.voxel_leaf_size);  // 1.0m
  // voxel_grid_filter.setInputCloud(scan_ptr);
  // voxel_grid_filter.filter(*filtered_scan_ptr);

  auto filtered_scan_ptr = VoxelFilter(scan_ptr, config.voxel_leaf_size);

  Eigen::Matrix4f init_guess =
      (odom_init.inverse() * real_ins).matrix().cast<float>();

  // align by ndt
  PointCloudPtr map_ptr(new PointCloud(map_cloud));
  ndt.setInputTarget(map_ptr);
  ndt.setInputSource(filtered_scan_ptr);

  bool has_converged;
  LOG(INFO) << std::fixed << std::setprecision(6)
            << "init_guess:" << init_guess(0, 3) << " " << init_guess(1, 3)
            << " " << init_guess(2, 3);
  PointCloudPtr output_cloud(new PointCloud);
  ndt.align(*output_cloud, init_guess);
  score = ndt.getFitnessScore();
  matrix_output = ndt.getFinalTransformation();
  has_converged = ndt.hasConverged();

  LOG(INFO) << std::fixed << std::setprecision(6) << "ndt_regis map->frame:("
            << map_ptr->size() << " -> " << filtered_scan_ptr->size()
            << "),score:" << score << " result:[" << matrix_output(0, 3) << ", "
            << matrix_output(1, 3) << "] ,cnt:" << cnt;
  // publish odom path
  static nav_msgs::Path odom_path;
  POSEPtr odom_pose(new POSE);
  odom_pose->time = stamp;
  auto odom_tf = odom_init.matrix() * matrix_output.cast<double>();
  odom_pose->pos = Eigen::Vector3d(odom_tf(0, 3), odom_tf(1, 3), odom_tf(2, 3));
  odom_pose->q = Eigen::Quaterniond(odom_tf.block<3, 3>(0, 0));
  ceres_optimizer.AddPoseToNavPath(odom_path, odom_pose);
  odom_pos_pub.publish(odom_path);

  return has_converged;
}
static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  lidar_alignment::Timer elapsed_t;
  double timestamp = input->header.stamp.toSec();
  // preprocess scan-msg
  PointCloud tmp;
  pcl::fromROSMsg(*input, tmp);
  PointCloudPtr scan_ptr(new PointCloud());
  FilterByDist(tmp, scan_ptr);

  Eigen::Matrix4f tf_pose(Eigen::Matrix4f::Identity());
  static Eigen::Affine3d last_tf(tf_pose.cast<double>());
  double fitness_score = 0;
  if (!Align(scan_ptr, timestamp, tf_pose, fitness_score)) {
    LOG(INFO) << std::fixed << std::setprecision(6)
              << "ndt align failed, score:" << fitness_score;
    return;
  }
  // insert odom
  Eigen::Affine3d tf =
      last_tf.inverse() * Eigen::Affine3d(tf_pose.cast<double>());
  last_tf = tf_pose.cast<double>();
  POSEPtr odom_factor = Affine2POSEPtr(tf, timestamp);
  ceres_optimizer.InsertOdom(odom_factor);

  AddKeyFrame(timestamp, tf.matrix(), tf_pose.cast<double>(), scan_ptr);

  // extend map's cloud
  static Pose current_pose, added_pose;
  current_pose = Matrix2Pose(tf_pose, timestamp);
  double shift = std::sqrt(std::pow(current_pose.x - added_pose.x, 2.0) +
                           std::pow(current_pose.y - added_pose.y, 2.0));
  if (shift >= config.min_add_scan_shift) {
    PointCloudPtr transformed_scan_ptr(new PointCloud());
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tf_pose);
    map_cloud += *transformed_scan_ptr;
    added_pose = current_pose;
  }

  // publish optimized path
  static nav_msgs::Path* optimized_path = &ceres_optimizer.optimized_path;
  if (!optimized_path->poses.empty()) {
    optimized_pos_pub.publish(*optimized_path);
  }
  LOG(INFO) << "elapsed time:" << elapsed_t.end() << " [ms]";
}

PointCloudPtr GenerateMap(double resolution = 0.05) {
  if (key_frame_ptrs.empty()) return nullptr;
  // update optimized pose for keyframe
  int i(0);
  auto optimized_poses = ceres_optimizer.GetOptimizedResult();
  LOG(INFO) << "Generating optimized_pose:" << optimized_poses.size();

  for (auto& keyframe : key_frame_ptrs) {
    for (; i < optimized_poses.size(); ++i) {
      auto elem = optimized_poses[i];
      if (keyframe->stamp != elem->time) continue;
      keyframe->pose.linear() = elem->q.toRotationMatrix();
      keyframe->pose.translation() = elem->pos;
      break;
    }
  }

  PointCloudPtr map_data_ptr(new PointCloud);
  map_data_ptr->reserve(key_frame_ptrs.size() *
                        key_frame_ptrs.front()->cloud->size());

  for (const auto& frame : key_frame_ptrs) {
    // coordinate system
    Eigen::Isometry3d pose = frame->pose;  // global coordinate system
    PointCloudPtr temp_cloud(new PointCloud);
    // transfer to local coordinate system
    // pose.matrix()(0, 3) -= init_pose.matrix()(0, 3);
    // pose.matrix()(1, 3) -= init_pose.matrix()(1, 3);
    // pose.matrix()(2, 3) -= init_pose.matrix()(2, 3);
    pose.matrix() = odom_init.inverse() * pose.matrix();
    pcl::transformPointCloud(*(frame->cloud), *temp_cloud, pose.matrix());
    *map_data_ptr += *temp_cloud;
  }
  map_data_ptr->header = key_frame_ptrs.back()->cloud->header;

  SetCloudAttributes(map_data_ptr);

  pcl::octree::OctreePointCloud<PointT> octree(resolution);
  octree.setInputCloud(map_data_ptr);
  octree.addPointsFromInputCloud();

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
  octree.getOccupiedVoxelCenters(filtered->points);
  filtered->header = map_data_ptr->header;
  SetCloudAttributes(filtered);

  return filtered;
}

void MappingTimerCallback(const ros::TimerEvent&) {
  Timer t;
  LOG(INFO) << "================mapping timer callback====================";
  if (!pub_map.getNumSubscribers()) return;
  keyframe_mutex.lock();
  auto map_data_ptr = GenerateMap();
  keyframe_mutex.unlock();
  if (!map_data_ptr) return;

  map_data_ptr = VoxelFilter(map_data_ptr, config.map_voxel_filter_size);

  sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(*map_data_ptr, *cloud_msg);

  pub_map.publish(cloud_msg);
  LOG(WARNING)
      << "Generating map cloud is done, and publish success.elspaed's time:"
      << t.end() << " [ms]";
}
static void output_callback(
    const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input) {
  if (map_cloud.empty()) {
    LOG(WARNING) << "map is empty!";
    return;
  }

  double filter_res = input->filter_res;
  std::string filename = input->filename;
  std::cout << "output callback" << std::endl;
  std::cout << "filter_res: " << filter_res << std::endl;
  std::cout << "filename: " << filename << std::endl;

  // PointCloudPtr output_map_ptr(new PointCloud(map_cloud));
  keyframe_mutex.lock();
  auto output_map_ptr = GenerateMap();
  keyframe_mutex.unlock();

  output_map_ptr->header.frame_id = "map";
  for (int i = 0; i < 20; ++i) {
    LOG(INFO) << std::fixed << std::setprecision(6)
              << "map-point:" << output_map_ptr->points[i].x << ", "
              << output_map_ptr->points[i].y << ", "
              << output_map_ptr->points[i].z;
  }

  // Writing Point Cloud data to PCD file
  if (filter_res == 0.0) {
    LOG(INFO) << "Original: " << output_map_ptr->points.size() << " points.";
    pcl::io::savePCDFileASCII(filename, *output_map_ptr);
    LOG(INFO) << "Saved " << output_map_ptr->points.size()
              << " data origin points to " << filename;
  } else {
    PointCloudPtr map_filtered(new PointCloud());
    map_filtered->header.frame_id = "map";
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(filter_res, filter_res, filter_res);
    voxel_grid_filter.setInputCloud(output_map_ptr);
    voxel_grid_filter.filter(*map_filtered);
    pcl::io::savePCDFileBinary(filename, *map_filtered);
    // pcl::io::savePCDFileASCII1(filename, *map_filtered);
    LOG(INFO) << "Original map points'size:" << output_map_ptr->size();
    LOG(INFO) << "Saved " << map_filtered->points.size()
              << " data filtered's size[" << filter_res << "] points to "
              << filename;
  }
  LOG(INFO) << "map_points save finished.";
}

void InputParams(ros::NodeHandle& private_nh) {
  // params timestamp thershold
  private_nh.getParam("config_time_threshold", config.time_threshold);
  private_nh.getParam("scan_calibration_angle", config.scan_calibration_angle);

  // params filter
  private_nh.getParam("min_scan_range", config.min_scan_range);
  private_nh.getParam("max_scan_range", config.max_scan_range);
  private_nh.getParam("voxel_leaf_size", config.voxel_leaf_size);

  // params registration
  // ndt
  private_nh.getParam("ndt_trans_epsilon", config.trans_eps);
  private_nh.getParam("ndt_step_size", config.step_size);
  private_nh.getParam("ndt_resolution", config.ndt_res);
  private_nh.getParam("ndt_maxiterations", config.max_iter);

  private_nh.getParam("min_add_scan_shift", config.min_add_scan_shift);
  // params ceres
  private_nh.getParam("optimize_num_every_scans",
                      config.ceres_config.num_every_scans);
  private_nh.getParam("optimize_iter_num", config.ceres_config.iters_num);
  private_nh.getParam("optimize_var_anchor", config.ceres_config.var_anchor);
  private_nh.getParam("optimize_var_odom_t", config.ceres_config.var_odom_t);
  private_nh.getParam("optimize_var_odom_q", config.ceres_config.var_odom_q);
  // mapping params
  private_nh.getParam("map_init_filename", config.map_init_filename);
  private_nh.getParam("map_cloud_update_interval",
                      config.map_cloud_update_interval);
  private_nh.getParam("keyframe_delta_trans", config.keyframe_delta_trans);
  private_nh.getParam("map_voxel_filter_size", config.map_voxel_filter_size);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "ceres_ndt_mapping");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  InputParams(private_nh);
  // OutputParams();

  ceres_optimizer.SetConfig(config.ceres_config);
  ndt.setTransformationEpsilon(config.trans_eps);  // 0.01
  ndt.setStepSize(config.step_size);               // 0.1
  ndt.setResolution(config.ndt_res);               // 1
  ndt.setMaximumIterations(config.max_iter);       // 30
  map_cloud.header.frame_id = "map";

  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 100000, InsCallback);
  ros::Subscriber points_sub =
      nh.subscribe("points_raw", 100000, points_callback);

  // publisher
  pub_scan =
      nh.advertise<sensor_msgs::PointCloud2>("/mapping/filtered_frame", 10);

  ins_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/gps", 100);
  odom_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/odom", 100);
  optimized_pos_pub =
      nh.advertise<nav_msgs::Path>("/mapping/path/optimized", 100);

  // maping
  ros::Timer map_publish_timer = nh.createTimer(
      ros::Duration(config.map_cloud_update_interval), MappingTimerCallback);
  pub_map = nh.advertise<sensor_msgs::PointCloud2>("/mapping/localizer_map", 1);

  ros::Subscriber output_sub =
      nh.subscribe("config/ndt_mapping_output", 10, output_callback);
  ros::spin();

  return 0;
}
