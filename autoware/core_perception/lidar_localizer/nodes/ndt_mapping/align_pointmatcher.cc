#include "align_pointmatcher.h"

AlignPointMatcher::AlignPointMatcher(const std::string& yaml_file) {
  std::ifstream ifs(yaml_file.c_str());
  if (boost::filesystem::extension(yaml_file) != ".yaml" || !ifs.good()) {
    ROS_WARN_STREAM( "Cannot open the config file:" << yaml_file
                 << ", switch to uses ICP.Default's config...");
    icp.setDefault();
  } else {
    icp.loadFromYaml(ifs);
  }

  // CHECK(Init(yaml_file));
}

bool AlignPointMatcher::Init(const std::string& config_file) {
  prev_trans_.setIdentity();
  odom_pose_.setIdentity();

  // load config
  YAML::Node config = YAML::LoadFile(config_file);
  config_.max_trans = config["configThreshold"]["max_trans"].as<double>();
  config_.max_angle = config["configThreshold"]["max_angle"].as<double>();
  config_.min_trans = config["configThreshold"]["min_trans"].as<double>();

  rigid_trans_ = PM::get().REG(Transformation).create("RigidTransformation");

  // init  matcher_hausdorff_

  PointMatcherSupport::Parametrizable::Parameters params;
  params["knn"] =
      PointMatcherSupport::toParam(1);  // find the fist closest point
  params["epsilon"] = PointMatcherSupport::toParam(0);

  matcher_hausdorff_ =
      PM::get().MatcherRegistrar.create("KDTreeMatcher", params);
  return true;
}
void AlignPointMatcher::CheckInitTransformation(
    const Eigen::Matrix4d& init_matrix,
    PM::TransformationParameters& init_trans) {
  init_trans = init_matrix;
  if (!rigid_trans_->checkParameters(init_matrix)) {
    ROS_WARN_STREAM( "Init transformation is not rigid, identity will be used!");
    init_trans = PM::TransformationParameters::Identity(4, 4);
  }
}

template <typename PointT>
void AlignPointMatcher::ConverToDataPoints(
    const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP& cur_pts) {
  std::size_t num = cloud->size();
  cur_pts.features.resize(4, num);
  // std::cout<<"pts:"<<num<<std::endl;
  for (std::size_t i = 0; i < num; i++) {
    cur_pts.features(0, i) = cloud->points[i].x;
    cur_pts.features(1, i) = cloud->points[i].y;
    cur_pts.features(2, i) = cloud->points[i].z;
    cur_pts.features(3, i) = 1.0;
  }
}
template void AlignPointMatcher::ConverToDataPoints<pcl::PointXYZ>(
    const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, DP& cur_pts);
template void AlignPointMatcher::ConverToDataPoints<pcl::PointXYZI>(
    const typename pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud,
    DP& cur_pts);
// template function of class
template <typename PointT>
bool AlignPointMatcher::Align(
    const typename pcl::PointCloud<PointT>::ConstPtr reference_cloud,
    const typename pcl::PointCloud<PointT>::ConstPtr reading_cloud,
    Eigen::Matrix4d& trans_matrix, double& score) {
  DP reference_pts;
  DP reading_pts;
  // std::cout<<"----------------------1\n";
  ConverToDataPoints<PointT>(reference_cloud, reference_pts);
  ConverToDataPoints<PointT>(reading_cloud, reading_pts);
  // std::cout<<"----------------------2\n";
  PM::TransformationParameters T = icp(reading_pts, reference_pts);
  // std::cout<<"----------------------3\n";
  trans_matrix = T;
  CalRobustScore(score);

  // Eigen::Matrix4d delta = prev_trans_.inverse() * T;
  // odom_pose_ = odom_pose_ * delta;
  // double delta_trans = std::abs(odom_pose_.block<2, 1>(0, 3).norm());
  // double angle = std::acos(Eigen::Quaterniond(delta.block<3, 3>(0, 0)).w()) *
  //                common::RAD_TO_DEG;

  // if (delta_trans > config_.max_trans || angle > config_.max_angle) {
  //   ROS_INFO_STREAM("too large transform:" << delta_trans << "[m],threshold["
  //             << config_.max_trans << "] " << angle << "[degree],threshold["
  //             << config_.max_angle << "]...ignore frame!!!");
  //   T = prev_trans_;
  // } else if (delta_trans < config_.min_trans) {
  //   ROS_INFO_STREAM("too small transform:" << delta_trans << "[m],threshold["
  //             << config_.min_trans << "] "
  //             << "...ignore frame!!!");
  //   trans_matrix.setIdentity();
  //   odom_pose_.setIdentity();
  // }

  // prev_trans_ = T;
  return true;
}
template bool AlignPointMatcher::Align<pcl::PointXYZ>(
    const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr reference_cloud,
    const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr reading_cloud,
    Eigen::Matrix4d& trans_matrix, double& score);
template bool AlignPointMatcher::Align<pcl::PointXYZI>(
    const typename pcl::PointCloud<pcl::PointXYZI>::ConstPtr reference_cloud,
    const typename pcl::PointCloud<pcl::PointXYZI>::ConstPtr reading_cloud,
    Eigen::Matrix4d& trans_matrix, double& score);

void AlignPointMatcher::CalRobustScore(double& score) {
  auto matched_pts = icp.errorMinimizer->getErrorElements();
  // extract relevant infos for convenience
  const int dim = matched_pts.reading.getEuclideanDim();
  const int num_matched_pts = matched_pts.reading.getNbPoints();
  // get matched point of ref and reading
  const PM::Matrix matched_reading = matched_pts.reading.features.topRows(dim);
  const PM::Matrix matched_ref = matched_pts.reference.features.topRows(dim);
  // compute mean distance
  const PM::Matrix dist = (matched_reading - matched_ref).colwise().norm();
  const double mean_dist = dist.sum() / num_matched_pts;

  score = mean_dist;
  ROS_INFO_STREAM("libpointmatcher mean robust distance:" << mean_dist);
}
void AlignPointMatcher::CalHaussdorffScore(
    const DP& ref, const DP& reading_pts, const PM::TransformationParameters& T,
    double& score) {
  // distance from ref to reading(optional)
  matcher_hausdorff_->init(reading_pts);
  auto matches = matcher_hausdorff_->findClosests(ref);
  double max_dist2 = matches.getDistsQuantile(1.0);
  double max_robust_dist2 = matches.getDistsQuantile(0.85);

  score = std::sqrt(max_robust_dist2);
  ROS_INFO_STREAM("libpointmatcher hausdorff_dist(ref->reading):" << score);
}