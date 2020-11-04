#pragma once
#include <glog/logging.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "singleton_base.h"

namespace ceres_mapping {
namespace lidar_odom{

namespace common{
    const float  DEG_TO_RAD = M_PI / 180;
    const double DEG_TO_RAD_D = M_PI / 180.0;

    const float  RAD_TO_DEG = 180 / M_PI;
    const double RAD_TO_DEG_D = 180.0 / M_PI;
}


class LidarFrameConfig : public SingleInstance<LidarFrameConfig>{
 friend class SingleInstance<LidarFrameConfig>;

 public:
  struct Config{
    //lidar config
    std::uint16_t num_vertical_scans   = 16;
    std::uint16_t num_horizontal_scans = 1800;
    std::uint16_t ground_scan_index = 7;
  
    //degrees
    float vertical_angle_bottom = -15;
    float vertical_angle_top = 15;
    float sensor_mount_angle = 0;
    float scan_period = 0.1;
  
    //projection
    std::uint16_t segment_cluster_num = 30;
    std::uint16_t segment_valid_point_num = 5;
    std::uint16_t segment_valid_line_num = 3;
  
    //improve accuracy by decrease the value
    float segment_theta = 60.0;
    float surf_segmentation_angle_threshold = 10.0;
  
    //feature extraction
    float edge_threshold = 0.1;
    float surf_threshold = 0.1;
    float nearest_feature_search_dist = 5;
  }; 


//   ~LidarFrameConfig() = default;

//   static LidarFrameConfig* GetInstance(const Config& config){
//       static LidarFrameConfig instance(config);
//       return &instance;
//   }
  LidarFrameConfig(const LidarFrameConfig& rhs) = delete;
  LidarFrameConfig& operator=(const LidarFrameConfig& rhs) = delete;

  
  
  const bool init = false;
  Config config;

 private:
  LidarFrameConfig():instance_cnt_(0){
    instance_cnt_++;
    if(!init) {
      LOG(ERROR)<<"Attempting to use lidar frame config without initilization!";
    }
  }
  LidarFrameConfig(const Config& config):config(config),init(true){}
  int instance_cnt_;
};

}
}