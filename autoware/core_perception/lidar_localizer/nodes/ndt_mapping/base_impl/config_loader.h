#pragma once
#include <glog/logging.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "lidar_config.h"

namespace ceres_mapping {
namespace lidar_odom{

void LoadLidarConfig(const std::string& config_yaml_file);
void LoadExtrinsic(const std::string& config_file);

}}