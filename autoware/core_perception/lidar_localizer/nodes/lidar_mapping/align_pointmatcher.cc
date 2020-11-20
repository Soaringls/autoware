#include "align_pointmatcher.h"

namespace autobot {
namespace cyberverse {
namespace mapping {

AlignPointMatcher::AlignPointMatcher(const std::string& yaml_file) {
  SetIcpSetupByYamlFile(yaml_file);
}

void AlignPointMatcher::SetIcpSetupByYamlFile(const std::string& yaml_file) {
  std::ifstream ifs(yaml_file.c_str());
  if (boost::filesystem::extension(yaml_file) != ".yaml" || !ifs.good()) {
    LOG(WARNING) << "Cannot open the config file:" << yaml_file
                 << ", switch to uses ICP.Default's config...";
  } else {
    icp.loadFromYaml(ifs);
  }
  CHECK(Init(yaml_file));
}

bool AlignPointMatcher::Init(const std::string& config_file) {
  rigid_trans_ = PM::get().REG(Transformation).create("RigidTransformation");
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
    LOG(WARNING) << "Init transformation is not rigid, identity will be used!";
    init_trans = PM::TransformationParameters::Identity(4, 4);
  }
}

template <typename PointT>
void AlignPointMatcher::ConverToDataPoints(
    const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP& cur_pts) {
  std::size_t num = cloud->size();
  cur_pts.features.resize(4, num);
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

bool AlignPointMatcher::Align(const DP& reference_cloud,
                              const DP& reading_cloud,
                              Eigen::Matrix4d& trans_matrix, double& score,
                              std::string info,
                              const Eigen::Matrix4d& init_tf) {
  trans_matrix = icp(reading_cloud, reference_cloud, init_tf);
  CalOdomRobustScore(score, info);
  // align_pts_ = reference_cloud;
  // tf_reading_pts_ = reading_cloud;

  // trans_matrix = icp(tf_reading_pts_, align_pts_, init_tf);

  // icp.transformations.apply(tf_reading_pts_, trans_matrix);

  // align_pts_.concatenate(tf_reading_pts_);

  //  DP reading_pts_aligned(tf_reading_pts_);
  // CalHaussdorffScore(reference_pts, reading_pts_aligned, T, score);
  //  CalRobustScore(reference_pts, reading_pts_aligned, T, score);
  return true;
}
void AlignPointMatcher::CalOdomRobustScore(double& score, std::string info) {
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
  LOG(INFO) << "libpointmatcher mean robust distance(" << info
            << "):" << mean_dist;
}
template <typename PointT>
bool AlignPointMatcher::Align(
    const typename pcl::PointCloud<PointT>::ConstPtr reference_cloud,
    const typename pcl::PointCloud<PointT>::ConstPtr reading_cloud,
    Eigen::Matrix4d& trans_matrix, double& score) {
  DP reference_pts;
  DP reading_pts;

  ConverToDataPoints<PointT>(reference_cloud, reference_pts);
  ConverToDataPoints<PointT>(reading_cloud, reading_pts);

  this->Align(reference_pts, reading_pts, trans_matrix, score);
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

void AlignPointMatcher::CalRobustScore(const DP& ref, const DP& reading_pts,
                                       const PM::TransformationParameters& T,
                                       double& score) {
  icp.matcher->init(ref);

  PM::Matches matches;
  matches = icp.matcher->findClosests(reading_pts);

  // weight paired points
  const PM::OutlierWeights outlier_weights =
      icp.outlierFilters.compute(reading_pts, ref, matches);
  // generate tuples of matched points and remove pairs with zero weight
  const PM::ErrorMinimizer::ErrorElements matched_pts(reading_pts, ref,
                                                      outlier_weights, matches);
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
  LOG(INFO) << "libpointmatcher mean robust distance:" << mean_dist;
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
  LOG(INFO) << "libpointmatcher hausdorff_dist(ref->reading):" << score;
}
}  // namespace mapping
}  // namespace cyberverse
}  // namespace autobot