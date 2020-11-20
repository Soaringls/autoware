#include "slam.h"

template <typename PointT>
void AlignSLAM::ConverToDataPoints(
    const typename pcl::PointCloud<PointT>::ConstPtr cloud, DP &cur_pts) {
  std::size_t num = cloud->size();
  cur_pts.features.resize(4, num);
  for (std::size_t i = 0; i < num; i++) {
    cur_pts.features(0, i) = cloud->points[i].x;
    cur_pts.features(1, i) = cloud->points[i].y;
    cur_pts.features(2, i) = cloud->points[i].z;
    cur_pts.features(3, i) = 1.0;
  }
}
template void AlignSLAM::ConverToDataPoints<pcl::PointXYZ>(
    const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, DP &cur_pts);
template void AlignSLAM::ConverToDataPoints<pcl::PointXYZI>(
    const typename pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud,
    DP &cur_pts);

void AlignSLAM::SaveMap(const std::string map_filename,
                        const std::string &method) {
  Timer t;
  bool has_point = false;
  if (method == "pointmatcher") {
    DP map;
    std::shared_ptr<PM::DataPointsFilter> randomSample =
        PM::get().DataPointsFilterRegistrar.create(
            "RandomSamplingDataPointsFilter", {{"prob", toParam(0.4)}});

    for (const auto &scan : raw_scans_) {
      if (poses_.find(scan.first) == poses_.end()) {
        LOG(INFO) << "Failed to find id: " << scan.first;
        continue;
      }
      auto tf = poses_[scan.first];

      auto filter_scan = randomSample->filter(scan.second);
      if (!has_point) {
        map = Transformation(filter_scan, tf.matrix());
        has_point = true;
      } else {
        map.concatenate(Transformation(filter_scan, tf.matrix()));
      }
    }
    LOG(INFO) << "generate map elapsed:" << t.end() << " [ms].";
    map = randomSample->filter(map);
    LOG(INFO) << "filtered map elapsed:" << t.end() << " [ms].";
    map.save(map_filename);
  } else {
    PointCloud map;
    LOG(INFO) << "starto to save map. total:" << raw_frames_.size()
              << " frames.";
    if (raw_frames_.size() == 0) return;
    for (auto scan : raw_frames_) {
      if (poses_.find(scan.first) == poses_.end()) {
        LOG(INFO) << "Failed to find id: " << scan.first;
        continue;
      }
      auto tf = poses_[scan.first];
      auto transformed_cloud = TransformCloud(scan.second, tf);
      if (!has_point) {
        map = *transformed_cloud;
        map.header = transformed_cloud->header;
        has_point = true;
      } else {
        map += *transformed_cloud;
      }
      //   LOG(INFO) << "transformcloud's size:" << transformed_cloud->size()
      //             << ",  map's size:" << map.size();
    }
    map.width = map.size();
    map.height = 1;
    map.is_dense = false;
    PointCloudPtr map_ptr(new PointCloud(map));
    LOG(INFO) << "raw      map's size:" << map_ptr->size();
    map_ptr = FilterByOctree(map_ptr, config_.octree_filter_resol);
    LOG(INFO) << "filtered map's size:" << map_ptr->size();

    map_ptr->header.frame_id = "map";
    pcl::io::savePCDFileASCII(map_filename, *map_ptr);
    LOG(INFO) << "generate map done, map's size:" << map_ptr->size();
  }

  LOG(INFO) << "slam_map_file: " << map_filename << " elapsed:" << t.end()
            << " [ms]. save by:" << method;
}

bool AlignSLAM::InScan(const PointCloudPtr cloud_in, int id) {
  PointCloudPtr segmented_cloud = SegmentedCloud(cloud_in);
  LOG(INFO) << "add scan:" << cloud_in->size()
            << ", segmented:" << segmented_cloud->size() << ", id:" << id;
  LOG(INFO) << "raw_frames_:" << raw_frames_.size()
            << ", poses_:" << poses_.size() << ", scans_:" << scans_.size();
  raw_frames_[id] = cloud_in;

  DP current_scan;
  ConverToDataPoints<PointT>(segmented_cloud, current_scan);
  return InScan(current_scan, id);
}
bool AlignSLAM::InScan(const DP &scan, int id) {
  DP current_scan = scan;

  if (!Odometry(current_scan, id)) {
    LOG(FATAL) << "Failed to Odometry!";
    return false;
  }

  {
    poses_[id] = current_tf_;
    scans_[id] = current_scan;
  }

  if (!Scan2Map(current_scan, id)) {
    LOG(FATAL) << "Failed to Scan2Map!";
    return false;
  }

  {
    poses_[id] = current_tf_;  // modify poses_[id]
    last_scan_ = current_scan;
    last_tf_ = current_tf_;
  }
  return true;
}

// private interface
bool AlignSLAM::Odometry(DP &scan, int id) {
  if (!init_) {
    init_ = true;
    init_id_ = id;
    LOG(INFO) << "Odometry init success.";
    return true;
  }

  double score;
  Eigen::Matrix4d trans_matrix;
  LOG(INFO) << "Odometry:--1--last:" << last_scan_.features.cols()
            << ", cur-scan:" << scan.features.cols();
  if (!odo_align_->Align(last_scan_, scan, trans_matrix, score,
                         std::string("odom"))) {
    LOG(INFO) << "failed to align!!";
    return false;
  }

  current_tf_.matrix() = last_tf_.matrix() * trans_matrix;
  return true;
}
bool AlignSLAM::Scan2Map(const DP &scan, int id) {
  LOG(INFO) << "begin to scan2map, id:" << id;
  if (poses_.empty()) return true;

  auto nearbors = SearchNearbyScan(id);

  if (nearbors.empty()) return true;

  DP nearbor_map = ConcatenateNearbyMap(nearbors);

  double score;
  Eigen::Matrix4d trans_matrix;

  if (!map_align_->Align(nearbor_map, scan, trans_matrix, score,
                         std::string("scan2submap"))) {
    LOG(INFO) << "Failed to AlignCore::align !";
  }
  // detete debug code

  current_tf_.matrix() = current_tf_.matrix() * trans_matrix;
  return true;
}

std::vector<int> AlignSLAM::SearchNearbyScan(int id) {
  std::vector<int> indexs;

  int last_index = id;
  int front_index = id - 1;
  double distance = 0.0;
  double last_distance = 0.0;

  while (front_index > init_id_ && distance < config_.submap_length) {
    if (poses_.find(front_index) == poses_.end()) {
      LOG(INFO) << "Failed to find id: " << front_index;
      front_index--;
      continue;
    }
    auto front_tf = poses_[front_index];

    if (poses_.find(last_index) == poses_.end()) {
      LOG(INFO) << "Failed to find id: " << last_index;
      front_index--;
      continue;
    }
    auto last_tf = poses_[last_index];

    Eigen::Vector3d d_xyz =
        (front_tf.inverse() * last_tf).matrix().block<3, 1>(0, 3);
    distance += d_xyz.norm();

    if (distance - last_distance > config_.submap_frame_resol) {
      indexs.push_back(front_index);
      last_distance = distance;
    }

    last_index = front_index;
    front_index--;
  }
  LOG(INFO) << "distance: " << distance << ", indexs.size = " << indexs.size();
  return indexs;
}
DP AlignSLAM::ConcatenateNearbyMap(const std::vector<int> &nearbors) {
  DP map;
  bool has_point = false;
  for (const auto &id : nearbors) {
    if (scans_.find(id) == scans_.end()) {
      LOG(INFO) << "Failed to find id: " << id;
      continue;
    }
    auto scan = scans_[id];

    if (poses_.find(id) == poses_.end()) {
      LOG(INFO) << "Failed to find id: " << id;
      continue;
    }
    auto tf = poses_[id];

    if (!has_point) {
      map = Transformation(scan, tf.matrix());
      has_point = true;
    } else {
      map.concatenate(Transformation(scan, tf.matrix()));
    }
  }

  std::shared_ptr<PM::DataPointsFilter> randomSample =
      PM::get().DataPointsFilterRegistrar.create(
          "RandomSamplingDataPointsFilter", {{"prob", toParam(0.5)}});

  map = randomSample->filter(map);
  map = Transformation(map, current_tf_.inverse().matrix());
  return map;
}

PointCloudPtr AlignSLAM::TransformCloud(const PointCloudPtr cloud,
                                        const Eigen::Affine3d &pose) {
  PointCloudPtr transformed_cloud(new PointCloud);
  pcl::transformPointCloud(*cloud, *transformed_cloud, pose);
  transformed_cloud->header = cloud->header;
  return transformed_cloud;
}

// PointCloudPtr AlignSLAM::VoxelFilter(const PointCloudPtr cloud,
//                                      const double size) {
//   PointCloudPtr filtered(new PointCloud);
//   pcl::VoxelGrid<PointT> voxel_filter_;
//   voxel_filter_.setLeafSize(size, size, size);
//   voxel_filter_.setInputCloud(cloud);
//   voxel_filter_.filter(*filtered);
//   filtered->header = cloud->header;
//   return filtered;
// }

PointCloudPtr AlignSLAM::FilterByOctree(const PointCloudPtr cloud_ptr,
                                        double resolution) {
  pcl::octree::OctreePointCloud<PointT> octree(resolution);
  octree.setInputCloud(cloud_ptr);
  octree.addPointsFromInputCloud();

  PointCloudPtr filtered_ptr(new PointCloud());
  octree.getOccupiedVoxelCenters(filtered_ptr->points);
  filtered_ptr->header = cloud_ptr->header;
  filtered_ptr->width = filtered_ptr->size();
  filtered_ptr->height = 1;
  filtered_ptr->is_dense = false;
  return filtered_ptr;
}