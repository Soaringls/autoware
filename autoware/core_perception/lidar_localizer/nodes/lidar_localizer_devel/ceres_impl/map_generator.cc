#include "map_generator.h"

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

void ClearQueue(std::queue<geometry_msgs::PoseStamped::ConstPtr>& queue) {
  std::queue<geometry_msgs::PoseStamped::ConstPtr> empty;
  std::swap(queue, empty);
}
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
void InputParams(ros::NodeHandle& private_nh, MappingConfig& config) {
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
void OutputParams(const MappingConfig& config) {
  LOG(INFO) << "***********config params***********";
  LOG(INFO) << "config_time_threshold:" << config.config_time_threshold;
  LOG(INFO) << "scan_calibration_angle:" << config.scan_calibration_angle;
  LOG(INFO) << "\nfiltere:----------";
  LOG(INFO) << "min_scan_range    :" << config.min_scan_range;
  LOG(INFO) << "max_scan_range     :" << config.max_scan_range;
  LOG(INFO) << "voxel_leaf_size  :" << config.voxel_leaf_size;
  LOG(INFO) << "\nndt:--------------";
  LOG(INFO) << "ndt_trans_epsilon  :" << config.trans_eps;
  LOG(INFO) << "ndt_step_size      :" << config.step_size;
  LOG(INFO) << "ndt_resolution     :" << config.ndt_res;
  LOG(INFO) << "ndt_maxiterations  :" << config.max_iter;
  LOG(INFO) << "\nceres:------------";
  LOG(INFO) << "optimize_iter_num  :" << config.ceres_config.iters_num;
  LOG(INFO) << "optimize_var_anchor:" << config.ceres_config.var_anchor;
  LOG(INFO) << "optimize_var_odom_t:" << config.ceres_config.var_odom_t;
  LOG(INFO) << "optimize_var_odom_q:" << config.ceres_config.var_odom_q;
  LOG(INFO) << "\nmapping:----------";
  LOG(INFO) << "map_init_position_file:" << config.map_init_position_file;
  LOG(INFO) << "keyframe_delta_trans  :" << config.keyframe_delta_trans;
  LOG(INFO) << "map_cloud_update_interval:" << config.map_cloud_update_interval;
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

double calcDiffForRadian(const double lhs_rad, const double rhs_rad) {
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

void AddKeyFrame(const double& time, const Eigen::Matrix4d& tf,
                 const Eigen::Matrix4d& pose, const PointCloudPtr cloud,
                 const Eigen::Affine3d& map_init, const MappingConfig& config) {
  // publish current frame
  PointCloudPtr current_frame(new PointCloud);
  Eigen::Matrix4d visual_pose = pose;
  visual_pose = map_init.matrix().inverse() * visual_pose;
  pcl::transformPointCloud(*cloud, *current_frame, visual_pose);
  sensor_msgs::PointCloud2::Ptr output(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(*current_frame, *output);
  pub_filtered.publish(output);

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