#define OUTPUT

#include <autoware_config_msgs/ConfigNDTMapping.h>
#include <autoware_config_msgs/ConfigNDTMappingOutput.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/MarkerArray.h>

#include <fstream>
#include <iostream>
#include <string>

#include "mapping_impl/ceres_optimizer.h"
// #include "mapping_impl/align_pointmatcher.h"
// #include "mapping_impl/lidar_segmentation.h"

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

using namespace lidar_alignment;
enum class MethodType {
  PCL_GENERIC = 0,
  ICP_ETH = 1,
  PCL_OPENMP = 2,
};

struct Config {
  double time_synchro_threshold = 0.1;  // 100ms

  bool use_every_scan = false;
  double frame_dist = 0.1;
  // filter params
  double scan_calibration_angle = 0.0;
  double min_scan_range = 1.0;
  double max_scan_range = 50;
  double voxel_filter_size = 0.2;

  // registeration params
  MethodType align_method = MethodType::PCL_GENERIC;
  // ndt
  double ndt_trans_epsilon = 0.01;
  double ndt_step_size = 0.1;
  double ndt_resolution = 1.0;
  double ndt_maxiterations = 64;
  // icp
  std::string seg_config_file;
  std::string icp_config_file;

  // back-end optimization params
  CeresConfig ceres_config;
  // mapping
  double map_voxel_filter_size = 0.2;
  std::string map_init_position_file;
  double keyframe_delta_trans = 8;
  double map_cloud_update_interval = 10.0;

} config;

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

// global variables
static PointCloud global_map;
std::mutex gps_mutex, keyframe_mutex;
Eigen::Affine3d init_pose = Eigen::Affine3d::Identity();

std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;
pcl::VoxelGrid<PointT> voxel_filter;
std::vector<KeyFramePtr> key_frame_ptrs;

pcl::NormalDistributionsTransform<PointT, PointT> ndt_matching;
// AlignPointMatcher::Ptr align_pointmatcher_ptr;
CeresOptimizer ceres_optimizer;
// LidarSegmentationPtr lidar_segmentation_ptr;

visualization_msgs::MarkerArray marker_array;

// publisher
ros::Publisher ins_pos_pub, lio_pos_pub, optimized_pos_pub;
ros::Publisher pub_filtered, pub_map;

void SetCloudAttributes(PointCloudPtr& cloud, std::string frame_id = "world") {
  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = false;
  cloud->header.frame_id = frame_id;
}

Eigen::Isometry3d Matrix2Isometry(const Eigen::Matrix4d matrix) {
  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = matrix.block<3, 3>(0, 0);
  isometry.translation() = matrix.block<3, 1>(0, 3);
  return isometry;
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

static Eigen::Affine3d ins_init = Eigen::Affine3d::Identity();
void InsCallback(const geometry_msgs::PoseStamped::ConstPtr& msgs) {
  gps_mutex.lock();
  if (gps_msgs.size() > 30000) gps_msgs.pop();
  gps_msgs.push(msgs);
  gps_mutex.unlock();

  POSEPtr ins_factor(new POSE);
  ins_factor->time = msgs->header.stamp.toSec();
  ins_factor->pos = Eigen::Vector3d(
      msgs->pose.position.x, msgs->pose.position.y, msgs->pose.position.z);
  ins_factor->q =
      Eigen::Quaterniond(msgs->pose.orientation.w, msgs->pose.orientation.x,
                         msgs->pose.orientation.y, msgs->pose.orientation.z);

  static nav_msgs::Path gps_path;
  ceres_optimizer.AddPoseToNavPath(gps_path, ins_factor);
  ins_pos_pub.publish(gps_path);
}

void FilterByDistance(PointCloudPtr& cloud) {
  PointCloudPtr filtered(new PointCloud);
  std::copy_if(
      cloud->begin(), cloud->end(), std::back_inserter(filtered->points),
      [&](const PointT& pt) {
        //  double dist = pt.getVector3fMap().norm();
        double dist = std::sqrt(std::pow(pt.x, 2.0) + std::pow(pt.y, 2.0));
        return dist > config.min_scan_range && dist < config.max_scan_range;
      });
  filtered->header = cloud->header;
  cloud = filtered;
}
PointCloudPtr FilterCloudFrame(PointCloudPtr& input) {
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud<PointT>(*input, *input, indices);
  PointCloudPtr filtered(new PointCloud);
  for (int i = 0; i < input->size(); ++i) {
    auto pt = input->points[i];
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
        std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z))
      continue;
    filtered->push_back(pt);
  }
  FilterByDistance(filtered);
  SetCloudAttributes(filtered);
  return filtered;
}
void VoxelFilter(PointCloudPtr& cloud, const double size) {
  // voxel filter
  auto pts = cloud->size();
  PointCloudPtr filtered(new PointCloud);
  voxel_filter.setLeafSize(size, size, size);
  voxel_filter.setInputCloud(cloud);
  voxel_filter.filter(*filtered);

  filtered->header = cloud->header;
  cloud = filtered;
  LOG(INFO) << "origin:" << pts << ", filtered:" << filtered->size()
            << ", cloud:" << cloud->size();
}

void AddKeyFrame(const double& time, const Eigen::Matrix4d& tf,
                 const Eigen::Matrix4d& pose, const PointCloudPtr cloud) {
  // publish current frame
  PointCloudPtr current_frame(new PointCloud);
  Eigen::Matrix4d visual_pose = pose;
  visual_pose(0, 3) -= init_pose.matrix()(0, 3);
  visual_pose(1, 3) -= init_pose.matrix()(1, 3);
  visual_pose(2, 3) -= init_pose.matrix()(2, 3);
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
    pose.matrix()(0, 3) -= init_pose.matrix()(0, 3);
    pose.matrix()(1, 3) -= init_pose.matrix()(1, 3);
    pose.matrix()(2, 3) -= init_pose.matrix()(2, 3);
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

PointCloudPtr SegmentedCloud(PointCloudPtr cloud) {
  auto header = cloud->header;
  // lidar_segmentation_ptr->CloudMsgHandler(cloud);

  // auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloudPure();
  // seg_cloud->header = header;
  // return seg_cloud;
}

bool AlignPointCloud(const PointCloudPtr& cloud_in, Eigen::Matrix4d& matrix,
                     double& score) {
  static bool init(false);
  PointCloudPtr source_ptr(new PointCloud);
  pcl::copyPointCloud(*cloud_in, *source_ptr);
  if (config.align_method == MethodType::ICP_ETH) {
    source_ptr = SegmentedCloud(source_ptr);
  }

  static PointCloudPtr target_ptr(new PointCloud);
  if (!init) {
    pcl::copyPointCloud(*source_ptr, *target_ptr);
    init = true;
    return false;
  }
  Timer t;
  bool is_converged(true);
  if (config.align_method == MethodType::PCL_GENERIC) {
    VoxelFilter(source_ptr, config.voxel_filter_size);
    Eigen::Matrix4f init_guss = matrix.cast<float>();

    PointCloudPtr map_ptr(new PointCloud(global_map));

    ndt_matching.setInputTarget(map_ptr);
    ndt_matching.setInputSource(source_ptr);
    PointCloudPtr output(new PointCloud);
    ndt_matching.align(*output, init_guss);

    init_guss = ndt_matching.getFinalTransformation();

    matrix = init_guss.cast<double>();
    score = ndt_matching.getFitnessScore();
    is_converged = ndt_matching.hasConverged();
    target_ptr = map_ptr;
  } else if (config.align_method == MethodType::ICP_ETH) {
    // if (!align_pointmatcher_ptr->Align<PointT>(target_ptr, source_ptr, matrix,
    //                                            score)) {
    //   score = std::numeric_limits<double>::max();
    // }
    target_ptr->clear();
    pcl::copyPointCloud(*source_ptr, *target_ptr);
  }
  static uint32_t cnt(1);
  LOG(INFO) << "frame:[" << std::fixed << std::setprecision(6)
            << "id:" << cnt - 1 << "-pts:" << target_ptr->size() << " --> "
            << "id:" << cnt++ << "-pts:" << source_ptr->size()
            << "], score:" << score << ", elspaed's time:" << t.end()
            << " [ms]";

  return is_converged;
}

void OuputPointCloud(const PointCloudPtr& cloud) {
  for (int i = 0; i < 20; i++) {
    LOG(INFO) << "pts:" << std::fixed << std::setprecision(6)
              << cloud->points[i].x << ", " << cloud->points[i].y << ", "
              << cloud->points[i].z;
  }
}

bool FindCorrespondGpsMsg(const double pts_stamp, POSEPtr& ins_pose) {
  bool is_find(false);
  gps_mutex.lock();
  while (!gps_msgs.empty()) {
    auto stamp_diff = gps_msgs.front()->header.stamp.toSec() - pts_stamp;
    if (std::abs(stamp_diff) <= config.time_synchro_threshold) {
      POSEPtr pose(new POSE);
      pose->time = pts_stamp;
      auto gps_msg = gps_msgs.front();
      pose->pos =
          Eigen::Vector3d(gps_msg->pose.position.x, gps_msg->pose.position.y,
                          gps_msg->pose.position.z);
      pose->q = Eigen::Quaterniond(
          gps_msg->pose.orientation.w, gps_msg->pose.orientation.x,
          gps_msg->pose.orientation.y, gps_msg->pose.orientation.z);
      ins_pose = pose;
      is_find = true;
      break;
    } else if (stamp_diff < -config.time_synchro_threshold) {
      gps_msgs.pop();
    } else if (stamp_diff > config.time_synchro_threshold) {
      LOG(INFO) << "(gps_time - pts_time = " << stamp_diff
                << ") lidar msgs is delayed! ";
      break;
    }
  }
  gps_mutex.unlock();
  return is_find;
}

void PointsCallback(const sensor_msgs::PointCloud2::ConstPtr& input) {
  static bool init_pose_flag(false);
  if (!init_pose_flag && gps_msgs.empty()) {
    LOG(INFO) << "waiting to initizlizing by gps...";
    return;
  }

  static std::size_t frame_cnt(1);
  double timestamp = input->header.stamp.toSec();

  PointCloudPtr point_msg(new PointCloud);
  pcl::fromROSMsg(*input, *point_msg);
  LOG(INFO) << "original scan-msg:" << point_msg->size();
  point_msg = FilterCloudFrame(point_msg);
  if (point_msg->size() == 0) return;

  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.rotate(Eigen::AngleAxisd(config.scan_calibration_angle * M_PI / 180,
                                     Eigen::Vector3d::UnitZ()));
  pcl::transformPointCloud(*point_msg, *point_msg, transform);
  // test
  static int scan_cnt(1);
  LOG(INFO) << std::fixed << std::setprecision(6) << "map:" << global_map.size()
            << ", distance filtered:" << point_msg->size()
            << ", time:" << timestamp << " --cnt:" << scan_cnt++;

  Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
  double fitness_score = std::numeric_limits<double>::max();

  static Eigen::Matrix4d lio_pose = Eigen::Matrix4d::Identity();

  if (!init_pose_flag) {
    if (!AlignPointCloud(point_msg, tf, fitness_score)) {
      LOG(INFO) << "Align PointCloud init success.";
    }
    POSEPtr init_odom;
    if (!FindCorrespondGpsMsg(timestamp, init_odom)) {
      LOG(INFO) << "PointCloud Callback init failed!!!";
      return;
    }

    CHECK_NOTNULL(init_odom);
    ceres_optimizer.InsertOdom(init_odom);
    init_pose = POSE2Affine(init_odom);
    lio_pose = init_pose.matrix();

    init_pose_flag = true;
    // dump map's init pose
    DumpPose(init_pose, config.map_init_position_file);
    global_map = *point_msg;

    return;
  }

  // find real-time gps-pose
  POSEPtr ins_pose;
  if (!FindCorrespondGpsMsg(timestamp, ins_pose)) {
    LOG(INFO) << "failed to search ins_pose at:" << std::fixed
              << std::setprecision(6) << timestamp;
    return;
  }
  CHECK_NOTNULL(ins_pose);
  auto ins_pose_affine3d = POSE2Affine(ins_pose);

  if (!config.use_every_scan) {
    static Eigen::Affine3d last_ins = init_pose;
    Eigen::Matrix4d trans = (last_ins.inverse() * ins_pose_affine3d).matrix();
    auto dist = trans.block<3, 1>(0, 3).norm();
    if (dist < config.frame_dist) {
      LOG(INFO) << "the distance to last frame is :" << dist << " too small!!!";
      return;
    }
    last_ins = ins_pose_affine3d;
  }

  LOG(INFO) << "--------------------------new lidar-msg......";
  Timer t;
  tf = (init_pose.inverse() * ins_pose_affine3d).matrix();
  if (!AlignPointCloud(point_msg, tf, fitness_score)) {
    LOG(WARNING) << "regis between:(" << frame_cnt - 1 << ", " << frame_cnt
                 << ") is failed!!!";
  }

  static Eigen::Matrix4d add_pose = Eigen::Matrix4d::Identity();
  double shift = std::sqrt(std::pow(tf(0, 3) - add_pose(0, 3), 2.0) +
                           std::pow(tf(1, 3) - add_pose(1, 3), 2.0));
  if (shift < config.frame_dist) {
    LOG(INFO) << "shift:" << shift << " < " << config.frame_dist;
    return;
  }

  PointCloudPtr transformed_scan_ptr(new PointCloud);
  pcl::transformPointCloud(*point_msg, *transformed_scan_ptr, tf);
  global_map += *transformed_scan_ptr;

  auto trans = add_pose.inverse() * tf;
  add_pose = tf;
  tf = trans;

  // insert odom-trans to optimizer
  POSEPtr lio_factor(new POSE);
  lio_factor->time = timestamp;
  lio_factor->pos = Eigen::Vector3d(tf(0, 3), tf(1, 3), tf(2, 3));
  lio_factor->q = Eigen::Quaterniond(tf.block<3, 3>(0, 0));
  lio_factor->score = fitness_score;
  ceres_optimizer.InsertOdom(lio_factor);

  lio_pose *= tf;
  AddKeyFrame(timestamp, tf, ins_pose_affine3d.matrix(), point_msg);

  // insert gps data to optimizer
  if (frame_cnt++ % config.ceres_config.num_every_scans == 0) {
    LOG(INFO) << "Anchor is to correct the pose....";
    ceres_optimizer.InsertGPS(ins_pose);
  }

  // publish lidar_odometry's path
  static nav_msgs::Path odom_path;
  POSEPtr odom_pose(new POSE);
  odom_pose->time = timestamp;
  odom_pose->pos =
      Eigen::Vector3d(lio_pose(0, 3), lio_pose(1, 3), lio_pose(2, 3));
  odom_pose->q = Eigen::Quaterniond(lio_pose.block<3, 3>(0, 0));
  ceres_optimizer.AddPoseToNavPath(odom_path, odom_pose);
  lio_pos_pub.publish(odom_path);

  // publish optimized path
  static nav_msgs::Path* optimized_path = &ceres_optimizer.optimized_path;
  if (!optimized_path->poses.empty()) {
    optimized_pos_pub.publish(*optimized_path);
  }
  LOG(INFO) << "elspaed's time:" << t.end() << " [ms]";
}

void MappingTimerCallback(const ros::TimerEvent&) {
  Timer t;
  LOG(INFO) << "================mapping timer callback====================";
  if (!pub_map.getNumSubscribers()) return;
  keyframe_mutex.lock();
  auto map_data_ptr = GenerateMap();
  keyframe_mutex.unlock();
  if (!map_data_ptr) return;

  VoxelFilter(map_data_ptr, config.map_voxel_filter_size);

  sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
  pcl::toROSMsg(*map_data_ptr, *cloud_msg);

  pub_map.publish(cloud_msg);
  LOG(WARNING)
      << "Generating map cloud is done, and publish success.elspaed's time:"
      << t.end() << " [ms]";
}

static void output_callback(
    const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input) {
  Timer t;
  LOG(INFO) << "================begin to save map====================";
  PointCloudPtr map_data_ptr(new PointCloud);
  if (config.align_method == MethodType::PCL_GENERIC) {
    PointCloudPtr output_map_ptr(new PointCloud(global_map));
    map_data_ptr = output_map_ptr;
  } else {
    keyframe_mutex.lock();
    auto output_map_ptr = GenerateMap();
    keyframe_mutex.unlock();
    if (!output_map_ptr) return;
    map_data_ptr = output_map_ptr;
  }
  if (map_data_ptr->empty()) {
    LOG(INFO) << "map is empty!";
    return;
  }
  LOG(WARNING) << "Generate map is done .elspaed's time:" << t.end() << " [ms]";

  // save map
  double voxel_size = input->filter_res;
  std::string filename = input->filename;
  LOG(INFO) << "voxel_size:" << voxel_size
            << " ,output_map filename: " << filename;

  map_data_ptr->header.frame_id = "map";
  if (voxel_size > 0.0) {
    auto pts_original = map_data_ptr->size();
    VoxelFilter(map_data_ptr, voxel_size);
    LOG(INFO) << "filter map done, from(" << pts_original << "->"
              << map_data_ptr->size() << ")";
  }
  // pcl::io::savePCDFileASCII(filename, *map_data_ptr);
  pcl::io::savePCDFileBinary(filename, *map_data_ptr);
  LOG(INFO) << "Saved " << map_data_ptr->points.size() << " pts success!!!";
}

void InputParams(ros::NodeHandle& private_nh) {
  // params timestamp thershold
  private_nh.getParam("time_synchro_threshold", config.time_synchro_threshold);
  config.use_every_scan = private_nh.param<bool>("use_every_scan", true);

  private_nh.getParam("frame_dist", config.frame_dist);

  // params filter
  private_nh.getParam("scan_calibration_angle", config.scan_calibration_angle);
  private_nh.getParam("min_scan_range", config.min_scan_range);
  private_nh.getParam("max_scan_range", config.max_scan_range);
  private_nh.getParam("voxel_filter_size", config.voxel_filter_size);

  // params registration
  int ndt_method = 0;
  private_nh.getParam("method_type", ndt_method);
  config.align_method = static_cast<MethodType>(ndt_method);
  // icp
  private_nh.getParam("segment_config_file", config.seg_config_file);
  private_nh.getParam("icp_config_file", config.icp_config_file);
  // ndt
  private_nh.getParam("ndt_trans_epsilon", config.ndt_trans_epsilon);
  private_nh.getParam("ndt_step_size", config.ndt_step_size);
  private_nh.getParam("ndt_resolution", config.ndt_resolution);
  private_nh.getParam("ndt_maxiterations", config.ndt_maxiterations);
  // params ceres
  private_nh.getParam("optimize_num_every_scans",
                      config.ceres_config.num_every_scans);
  private_nh.getParam("optimize_iter_num", config.ceres_config.iters_num);
  private_nh.getParam("optimize_var_anchor", config.ceres_config.var_anchor);
  private_nh.getParam("optimize_var_odom_t", config.ceres_config.var_odom_t);
  private_nh.getParam("optimize_var_odom_q", config.ceres_config.var_odom_q);
  // mapping params
  private_nh.getParam("map_init_position_file", config.map_init_position_file);
  private_nh.getParam("map_cloud_update_interval",
                      config.map_cloud_update_interval);
  private_nh.getParam("keyframe_delta_trans", config.keyframe_delta_trans);
}
void OutputParams() {
  LOG(INFO) << "***********config params***********";
  LOG(INFO) << "time_synchro_threshold:" << config.time_synchro_threshold;
  LOG(INFO) << "use_every_scan:" << config.use_every_scan;
  LOG(INFO) << "frame_dist    :" << config.frame_dist;

  LOG(INFO) << "scan_calibration_angle:" << config.scan_calibration_angle;
  LOG(INFO) << "min_scan_range     :" << config.min_scan_range;
  LOG(INFO) << "max_scan_range     :" << config.max_scan_range;
  LOG(INFO) << "voxel_filter_size  :" << config.voxel_filter_size;

  LOG(INFO) << "seg_config_file    :" << config.seg_config_file;
  LOG(INFO) << "icp_config_file    :" << config.icp_config_file;

  LOG(INFO) << "ndt_trans_epsilon  :" << config.ndt_trans_epsilon;
  LOG(INFO) << "ndt_step_size      :" << config.ndt_step_size;
  LOG(INFO) << "ndt_resolution     :" << config.ndt_resolution;
  LOG(INFO) << "ndt_maxiterations  :" << config.ndt_maxiterations;

  LOG(INFO) << "optimize_iter_num  :" << config.ceres_config.iters_num;
  LOG(INFO) << "optimize_var_anchor:" << config.ceres_config.var_anchor;
  LOG(INFO) << "optimize_var_odom_t:" << config.ceres_config.var_odom_t;
  LOG(INFO) << "optimize_var_odom_q:" << config.ceres_config.var_odom_q;

  LOG(INFO) << "map_init_position_file:" << config.map_init_position_file;
  LOG(INFO) << "keyframe_delta_trans  :" << config.keyframe_delta_trans;
  LOG(INFO) << "map_cloud_update_interval:" << config.map_cloud_update_interval;
}
int main(int argc, char** argv) {
  ros::init(argc, argv, "lidar_mapping_ceres");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  InputParams(private_nh);
  OutputParams();

  if (config.align_method == MethodType::PCL_GENERIC) {
    ndt_matching.setTransformationEpsilon(config.ndt_trans_epsilon);
    ndt_matching.setStepSize(config.ndt_step_size);
    ndt_matching.setResolution(config.ndt_resolution);
    ndt_matching.setMaximumIterations(config.ndt_maxiterations);
  }

  // lidar_segmentation_ptr.reset(new LidarSegmentation(config.seg_config_file));
  // align_pointmatcher_ptr.reset(new AlignPointMatcher(config.icp_config_file));
  ceres_optimizer.SetConfig(config.ceres_config);

  // sub and pub
  ins_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/gps", 100);
  lio_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/odom", 100);
  optimized_pos_pub =
      nh.advertise<nav_msgs::Path>("/mapping/path/optimized", 100);
  pub_filtered =
      nh.advertise<sensor_msgs::PointCloud2>("/mapping/filtered_frame", 10);

  ros::Subscriber points_sub =
      nh.subscribe("/points_raw", 10000, PointsCallback);
  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 30000, InsCallback);

  ros::Timer map_publish_timer =
      nh.createTimer(ros::Duration(5.0), MappingTimerCallback);
  pub_map = nh.advertise<sensor_msgs::PointCloud2>("/mapping/localizer_map", 1);
  ros::Subscriber output_sub =
      nh.subscribe("config/ndt_mapping_output", 10, output_callback);

  ros::spin();
  return 0;
}