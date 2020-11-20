
#include "align_util.h"

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>


#include <glog/logging.h>

using namespace pcl;

struct PointXYZIT {
  float x;
  float y;
  float z;
  unsigned char intensity;
  double timestamp;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (uint8_t, intensity, intensity)
                                  (double, timestamp, timestamp))

namespace autobot {
namespace cyberverse {
namespace mapping {

using namespace std;

DP LoadDPFromParseFile(const std::string &pcd_file) {
  pcl::PointCloud<PointXYZIT> points;

  if (pcl::io::loadPCDFile(pcd_file, points) < 0) {
    LOG(FATAL) << "Can not load pcd file " << pcd_file;
  }

  std::size_t point_number = points.size();

  DP pm_points;

  pm_points.features.resize(4, point_number);
  pm_points.featureLabels.push_back(DP::Label("x", 1));
  pm_points.featureLabels.push_back(DP::Label("y", 1));
  pm_points.featureLabels.push_back(DP::Label("z", 1));
  pm_points.featureLabels.push_back(DP::Label("pad", 1));

  pm_points.descriptors.resize(1, point_number);
  pm_points.descriptorLabels.push_back(DP::Label("intensity", 1));

  pm_points.times.resize(1, point_number);
  pm_points.timeLabels.push_back(DP::Label("timestamp", 1));

  for (std::size_t i = 0; i < point_number; i++) {
    pm_points.features(0, i) = points.points[i].x;
    pm_points.features(1, i) = points.points[i].y;
    pm_points.features(2, i) = points.points[i].z;
    pm_points.features(3, i) = 1.0;

    pm_points.descriptors(0, i) = points.points[i].intensity;

    pm_points.times(0, i) = points.points[i].timestamp * 1e6;
  }
  return pm_points;
}

DP LoadDPFromFile(const std::string &pcd_file, const std::string &seg_file) {
  pcl::PointCloud<PointXYZIT> points;

  if (pcl::io::loadPCDFile(pcd_file, points) < 0) {
    LOG(FATAL) << "Can not load pcd file " << pcd_file;
  }

  pcl::PointCloud<pcl::PointXYZI> seg_cloud;

  if (pcl::io::loadPCDFile(seg_file, seg_cloud) < 0) {
    LOG(FATAL) << "Can not load pcd file " << seg_file;
  }

  pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZI>);

  kdtree->setEpsilon(0.02);
  kdtree->setInputCloud(seg_cloud.makeShared());

  pcl::PointIndicesPtr indices(new pcl::PointIndices);

  std::vector<int> k_indices;
  std::vector<float> k_sqr_distances;

  points.points.erase(
      remove_if(points.points.begin(), points.points.end(),
                [&](const PointXYZIT& p) -> bool {

                  if(p.timestamp < 1e-2) return true;
                  if(isnan(p.x)) return true;
                  if(isnan(p.y)) return true;
                  if(isnan(p.z)) return true;

                  pcl::PointXYZI query_p;
                  query_p.x = p.x;
                  query_p.y = p.y;
                  query_p.z = p.z;
                  kdtree->nearestKSearch(query_p, 1, k_indices, k_sqr_distances);
                  if(k_sqr_distances[0] > 0.05) return true;

                  return false;
                }), points.points.end());


  sort(points.points.begin(), points.points.end(), [](
      const PointXYZIT& p1, const PointXYZIT& p2){
    return p1.timestamp < p2.timestamp;
  });

  CHECK(points.points.front().timestamp <= points.points.back().timestamp);

  std::size_t point_number = points.size();

  DP pm_points;

  pm_points.features.resize(4, point_number);
  pm_points.featureLabels.push_back(DP::Label("x", 1));
  pm_points.featureLabels.push_back(DP::Label("y", 1));
  pm_points.featureLabels.push_back(DP::Label("z", 1));
  pm_points.featureLabels.push_back(DP::Label("pad", 1));

  pm_points.descriptors.resize(1, point_number);
  pm_points.descriptorLabels.push_back(DP::Label("intensity", 1));

  pm_points.times.resize(1, point_number);
  pm_points.timeLabels.push_back(DP::Label("timestamp", 1));

  for (std::size_t i = 0; i < point_number; i++) {
    pm_points.features(0, i) = points.points[i].x;
    pm_points.features(1, i) = points.points[i].y;
    pm_points.features(2, i) = points.points[i].z;
    pm_points.features(3, i) = 1.0;

    pm_points.descriptors(0, i) = points.points[i].intensity;

    pm_points.times(0, i) = points.points[i].timestamp * 1e6;
//    LOG(INFO) << "timestamp: " << std::to_string(pm_points.times(0, i) * 1e-6);
  }

  return pm_points;
}


/** From PointMatcher Libs */
PM::DataPoints Transformation(const PM::DataPoints& input,
                              const PM::TransformationParameters& parameters) {

  assert(input.features.rows() == parameters.rows());
  assert(parameters.rows() == parameters.cols());

  const unsigned int nbRows = parameters.rows()-1;
  const unsigned int nbCols = parameters.cols()-1;

  const PM::TransformationParameters R(parameters.topLeftCorner(nbRows, nbCols));

  //DataPoints transformedCloud(input.featureLabels, input.descriptorLabels, input.timeLabels, input.features.cols());
  PM::DataPoints transformedCloud = input;

  // Apply the transformation to features
  transformedCloud.features = parameters * input.features;

  // Apply the transformation to descriptors
  int row(0);
  const int descCols(input.descriptors.cols());
  for (size_t i = 0; i < input.descriptorLabels.size(); ++i)
  {
    const int span(input.descriptorLabels[i].span);
    const std::string& name(input.descriptorLabels[i].text);
    const BOOST_AUTO(inputDesc, input.descriptors.block(row, 0, span, descCols));
    BOOST_AUTO(outputDesc, transformedCloud.descriptors.block(row, 0, span, descCols));
    if (name == "normals" || name == "observationDirections")
      outputDesc = R * inputDesc;

    row += span;
  }

  return transformedCloud;
}

}  // namespace mapping
}  // namespace cyberverse
}  // namespace autobot
