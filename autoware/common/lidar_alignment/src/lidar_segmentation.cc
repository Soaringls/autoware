#include "lidar_segmentation.h"

namespace ceres_mapping {
namespace lidar_odom{

LidarSegmentation::LidarSegmentation(const std::string& lidar_config_file) {
    LoadLidarConfig(lidar_config_file);
    auto config = LidarFrameConfig::GetInstance().config;
    angle_resolution_x_ = (M_PI * 2)/config.num_horizontal_scans;
    angle_resolution_y_ = 
        common::DEG_TO_RAD_D * 
        (config.vertical_angle_top - config.vertical_angle_bottom) / 
        static_cast<float>(config.num_vertical_scans - 1);
    const size_t  cloud_size = config.num_horizontal_scans * config.num_vertical_scans;
    
    full_cloud_.reset(new pcl::PointCloud<PointT>());
    full_cloud_with_range_.reset(new pcl::PointCloud<PointT>());
  
    ground_cloud_.reset(new pcl::PointCloud<PointT>());
    segmented_cloud_with_ground_.reset(new pcl::PointCloud<PointT>());
    segmented_cloud_pure_.reset(new pcl::PointCloud<PointT>());
    outlier_cloud_.reset(new pcl::PointCloud<PointT>());
  
    full_cloud_->points.resize(cloud_size);
    full_cloud_with_range_->points.resize(cloud_size);
    LOG(INFO)<<"construct success.";
}


void LidarSegmentation::CloudMsgHandler(PointCloudPtr cloud_in){
  // Reset parameters
  resetParameters();
  lidar_cloud_in_ = cloud_in;
  segmented_cloud_pure_->header.stamp = cloud_in->header.stamp;
  segmented_cloud_with_ground_->header.stamp = cloud_in->header.stamp;
  // Copy and remove NAN points
//   std::vector<int> indices;
//   pcl::removeNaNFromPointCloud(*lidar_cloud_in_, *lidar_cloud_in_, indices);
  // Compute lidar scan start and end angle.
  findStartEndAngle();
  // Range image projection
  projectPointCloud();
  // Mark ground points
  groundRemoval();
  // Point cloud segmentation
  cloudClusters();
}


void LidarSegmentation::resetParameters(){
  auto config = LidarFrameConfig::GetInstance().config;
  const size_t cloud_size = config.num_vertical_scans * config.num_horizontal_scans;
  PointT nanPoint;
  nanPoint.x = std::numeric_limits<float>::quiet_NaN();
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();
  nanPoint.intensity = -1;
 
  ground_cloud_->clear();
  segmented_cloud_with_ground_->clear();
  segmented_cloud_pure_->clear();
  outlier_cloud_->clear();
 
  range_mat_.resize(config.num_vertical_scans, config.num_horizontal_scans);
  ground_mat_.resize(config.num_vertical_scans, config.num_horizontal_scans);
  label_mat_.resize(config.num_vertical_scans, config.num_horizontal_scans);
 
  range_mat_.fill(FLT_MAX);
  ground_mat_.setZero();
  label_mat_.setZero();
 
  label_count_ = 1;
 
  std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nanPoint);
  std::fill(full_cloud_with_range_->points.begin(), full_cloud_with_range_->points.end(), nanPoint);
  
  seg_msg_.startRingIndex.assign(config.num_vertical_scans, 0);
  seg_msg_.endRingIndex.assign(config.num_vertical_scans, 0);
  seg_msg_.segmentedCloudGroundFlag.assign(cloud_size, false);
  seg_msg_.segmentedCloudColInd.assign(cloud_size, 0);
  seg_msg_.segmentedCloudRange.assign(cloud_size, 0);
  seg_msg_.indexInFullPointCloud.assign(cloud_size, 0);
  point_cloud_range_.assign(cloud_size, 0);
}

void LidarSegmentation::findStartEndAngle() {
  PointT point = lidar_cloud_in_->points.front();
  seg_msg_.startOrientation = -std::atan2(point.y, point.x);

  point = lidar_cloud_in_->points.back();
  seg_msg_.endOrientation = -std::atan2(point.y, point.x) + 2 * M_PI;

  if(seg_msg_.endOrientation - seg_msg_.startOrientation > 3 * M_PI){
      seg_msg_.endOrientation -= 2 * M_PI;
  } else if(seg_msg_.endOrientation - seg_msg_.startOrientation < M_PI){
      seg_msg_.endOrientation += 2 * M_PI;
  }
  seg_msg_.orientationDiff = seg_msg_.endOrientation - seg_msg_.startOrientation;
}

void LidarSegmentation::projectPointCloud(){
  auto config = LidarFrameConfig::GetInstance().config;
  
  const size_t cloud_size = lidar_cloud_in_->points.size();
  for(size_t i = 0; i < cloud_size; ++i){
      PointT point = lidar_cloud_in_->points[i];

      float range = std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
      float vertical_angle = std::asin(point.z / range);

      int row = static_cast<int>(
          (vertical_angle - config.vertical_angle_bottom * common::DEG_TO_RAD_D) /
          angle_resolution_y_);
      if(row < 0 || row >= config.num_vertical_scans) {
          continue;
      }
      
      float horizontal_angle = std::atan2(point.x, point.y);
      int column = static_cast<int>(
          -std::round((horizontal_angle - M_PI_2)/angle_resolution_x_) + config.num_horizontal_scans * 0.5);
      if(column >= config.num_horizontal_scans){
          column -= config.num_horizontal_scans;
      }

      if(column < 0 || column >= config.num_horizontal_scans){
          continue;
      }
      if(range < 0.1) {
          continue;
      }

      range_mat_(row, column) = range;
      //integer is row number, decimal * 10000 is column number.
      point.intensity = static_cast<float>(row) + static_cast<float>(column) / 10000.0;
      size_t index = column + row * config.num_horizontal_scans;

      full_cloud_->points[index] = point;
      full_cloud_with_range_->points[index] = point;
      full_cloud_with_range_->points[index].intensity = range;
  }
}

void LidarSegmentation::groundRemoval(){
  auto config = LidarFrameConfig::GetInstance().config;
  for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
    for (size_t i = 0; i < config.ground_scan_index; ++i) {
      size_t lower_index = j + i * config.num_horizontal_scans;
      size_t upper_index = j + (i + 1) * config.num_horizontal_scans;

      if (full_cloud_->points[lower_index].intensity == -1 ||
          full_cloud_->points[upper_index].intensity == -1) {
        // no info to check, invalid points
        ground_mat_(i, j) = -1;
        continue;
      }

      float dX = full_cloud_->points[upper_index].x -
                 full_cloud_->points[lower_index].x;
      float dY = full_cloud_->points[upper_index].y -
                 full_cloud_->points[lower_index].y;
      float dZ = full_cloud_->points[upper_index].z -
                 full_cloud_->points[lower_index].z;

      float vertical_angle = std::atan2(dZ, sqrt(dX * dX + dY * dY + dZ * dZ));

      if ((vertical_angle - config.sensor_mount_angle * common::DEG_TO_RAD) <=
          config.surf_segmentation_angle_threshold * common::DEG_TO_RAD) {
        ground_mat_(i, j) = 1;
        ground_mat_(i + 1, j) = 1;
      }
    }
  }

  // extract ground cloud (ground_mat == 1)
  // mark entry that doesn't need to label (ground and invalid point) for
  // segmentation note that ground remove is from 0~_N_scan-1, need range_mat
  // for mark label matrix for the 16th scan
  for (size_t i = 0; i < config.num_vertical_scans; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (ground_mat_(i, j) == 1 || range_mat_(i, j) == FLT_MAX) {
        label_mat_(i, j) = -1;
      }
    }
  }

  for (size_t i = 0; i < config.ground_scan_index; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      if (ground_mat_(i, j) == 1)
        ground_cloud_->push_back(
            full_cloud_->points[j + i * config.num_horizontal_scans]);
    }
  }
}


void LidarSegmentation::labelComponents(int row, int col) {
  auto config = LidarFrameConfig::GetInstance().config;
  const float segment_theta_threshold =
      tan(config.segment_theta * common::DEG_TO_RAD);
  std::vector<bool> line_count_flag(config.num_vertical_scans, false);
  const size_t cloud_size =
      config.num_vertical_scans * config.num_horizontal_scans;
  using Coord2D = Eigen::Vector2i;
  boost::circular_buffer<Coord2D> queue(cloud_size);
  boost::circular_buffer<Coord2D> all_pushed(cloud_size);

  queue.push_back({row, col});
  all_pushed.push_back({row, col});

  const Coord2D neighbor_iterator[4] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};

  while (queue.size() > 0) {
    // Pop point
    Coord2D from_ind = queue.front();
    queue.pop_front();

    // Mark popped point
    label_mat_(from_ind.x(), from_ind.y()) = label_count_;
    // Loop through all the neighboring grids of popped grid

    for (const auto &iter : neighbor_iterator) {
      // new index
      int indX = from_ind.x() + iter.x();
      int indY = from_ind.y() + iter.y();
      // index should be within the boundary
      if (indX < 0 || indX >= config.num_vertical_scans) {
        continue;
      }
      // at range image margin (left or right side)
      if (indY < 0) {
        indY = config.num_horizontal_scans - 1;
      }
      if (indY >= config.num_horizontal_scans) {
        indY = 0;
      }
      // prevent infinite loop (caused by put already examined point back)
      if (label_mat_(indX, indY) != 0) {
        continue;
      }

      float d1 = std::max(range_mat_(from_ind.x(), from_ind.y()),
                          range_mat_(indX, indY));
      float d2 = std::min(range_mat_(from_ind.x(), from_ind.y()),
                          range_mat_(indX, indY));

      float alpha = (iter.x() == 0) ? angle_resolution_x_ : angle_resolution_y_;
      // Compute the vertical or horizontal angle to see whether this point is
      // located around the current point.
      float tang = (d2 * sin(alpha) / (d1 - d2 * cos(alpha)));

      if (tang > segment_theta_threshold) {
        queue.push_back({indX, indY});

        label_mat_(indX, indY) = label_count_;
        line_count_flag[indX] = true;

        all_pushed.push_back({indX, indY});
      }
    }
  }

  // check if this segment is valid
  bool feasible_segment = false;
  if (all_pushed.size() >= config.segment_cluster_num) {
    feasible_segment = true;
  } else if (all_pushed.size() >= config.segment_valid_point_num) {
    // amount of points do not reach 30, three lines with 5 points must be
    // marked.
    int line_count = 0;
    for (size_t i = 0; i < config.num_vertical_scans; ++i) {
      if (line_count_flag[i] == true) ++line_count;
    }
    if (line_count >= config.segment_valid_line_num) feasible_segment = true;
  }
  // segment is valid, mark these points
  if (feasible_segment == true) {
    ++label_count_;
  } else {  // segment is invalid, mark these points
    for (size_t i = 0; i < all_pushed.size(); ++i) {
      label_mat_(all_pushed[i].x(), all_pushed[i].y()) = 999999;
    }
  }
}


void LidarSegmentation::cloudClusters() {
  auto config = LidarFrameConfig::GetInstance().config;
  // segmentation process
  for (size_t i = 0; i < config.num_vertical_scans; ++i)
    for (size_t j = 0; j < config.num_horizontal_scans; ++j)
      if (label_mat_(i, j) == 0) labelComponents(i, j);

  int seg_cloud_index = 0;
  int point_cloud_index = 0;
  // extract segmented cloud for lidar odometry
  for (size_t i = 0; i < config.num_vertical_scans; ++i) {
    // seg_cloud_index only record the index of segmented cloud and
    // it is continuous recording num.
    seg_msg_.startRingIndex[i] = seg_cloud_index - 1 + 5;
    // record outliers which have not been marked and down sample it by j.
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      point_cloud_range_[point_cloud_index] = range_mat_(i, j);
      ++point_cloud_index;
      if (label_mat_(i, j) > 0 || ground_mat_(i, j) == 1) {
        // outliers that will not be used for optimization (always continue)
        if (label_mat_(i, j) == 999999) {
          // only record upper half of the scans as outliers.
          if (i > config.ground_scan_index && j % 5 == 0) {
            outlier_cloud_->push_back(
                full_cloud_->points[j + i * config.num_horizontal_scans]);
            continue;
          } else {
            continue;
          }
        }
        // majority of ground points are skipped, down sample and remove
        // surrounding 5 points.
        if (ground_mat_(i, j) == 1) {
          if (j % 2 != 0 && j > 5 &&
              j < static_cast<std::uint16_t>(config.num_horizontal_scans - 5))
            continue;
        }
        // mark ground points so they will not be considered as edge features
        // later
        seg_msg_.segmentedCloudGroundFlag[seg_cloud_index] =
            (ground_mat_(i, j) == 1);
        // mark the points' column index for marking occlusion later
        seg_msg_.segmentedCloudColInd[seg_cloud_index] = j;
        seg_msg_.segmentedCloudRange[seg_cloud_index] = range_mat_(i, j);
        seg_msg_.indexInFullPointCloud[seg_cloud_index] = point_cloud_index;
        // save seg cloud
        segmented_cloud_with_ground_->push_back(
            full_cloud_->points[j + i * config.num_horizontal_scans]);
        // size of seg cloud
        ++seg_cloud_index;
      }
    }
    // remove inital and last five points of each line.
    seg_msg_.endRingIndex[i] = seg_cloud_index - 1 - 5;
  }

  for (size_t i = 0; i < config.num_vertical_scans; ++i) {
    for (size_t j = 0; j < config.num_horizontal_scans; ++j) {
      // segmentation result without ground
      if (label_mat_(i, j) > 0 && label_mat_(i, j) != 999999) {
        segmented_cloud_pure_->push_back(
            full_cloud_->points[j + i * config.num_horizontal_scans]);
        // intensity is the cluster label
        segmented_cloud_pure_->points.back().intensity = label_mat_(i, j);
      }
    }
  }
}

}//end space lidar-odom
}