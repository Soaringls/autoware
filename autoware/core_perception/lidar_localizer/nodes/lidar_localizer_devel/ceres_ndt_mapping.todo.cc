#include "ceres_impl/ceres_optimizer.h"
#include "ceres_impl/map_generator.h"

ros::Publisher ins_pos_pub, odom_pos_pub, optimized_pos_pub;
MappingConfig config;
std::mutex gps_mutex;
std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;

CeresOptimizer ceres_optimizer;
static Eigen::Affine3d odom_init = Eigen::Affine3d::Identity();

static pcl::PointCloud<pcl::PointXYZI> map_cloud;

static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
/////////////////////////////////////////////////////////////////////////////

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

  PointCloudPtr output_map_ptr(new PointCloud(map_cloud));
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
  PointCloudPtr filtered_scan_ptr(new PointCloud());
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(config.voxel_leaf_size, config.voxel_leaf_size,
                                config.voxel_leaf_size);  // 1.0m
  voxel_grid_filter.setInputCloud(scan_ptr);
  voxel_grid_filter.filter(*filtered_scan_ptr);

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
void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input) {
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "ceres_ndt_mapping");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  InputParams(private_nh, config);
  OutputParams(config);

  ceres_optimizer.SetConfig(config.ceres_config);
  ndt.setTransformationEpsilon(config.trans_eps);  // 0.01
  ndt.setStepSize(config.step_size);               // 0.1
  ndt.setResolution(config.ndt_res);               // 1
  ndt.setMaximumIterations(config.max_iter);       // 30
  map_cloud.header.frame_id = "map";

  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 100000, InsCallback);
  ros::Subscriber points_sub =
      nh.subscribe("points_raw", 100000, points_callback);
  ros::Subscriber output_sub =
      nh.subscribe("config/ndt_mapping_output", 10, output_callback);

  // publisher
  ins_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/gps", 100);
  odom_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/odom", 100);
  optimized_pos_pub =
      nh.advertise<nav_msgs::Path>("/mapping/path/optimized", 100);
  ros::spin();

  return 0;
}
