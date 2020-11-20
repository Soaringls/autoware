#pragma once

#include "pointmatcher_macro.h"

namespace autobot {
namespace cyberverse {
namespace mapping {

/** From PointMatcher Libs */
PM::DataPoints Transformation(const PM::DataPoints& input,
                              const PM::TransformationParameters& parameters);

DP LoadDPFromParseFile(const std::string &pcd_file);

DP LoadDPFromFile(const std::string &pcd_file, const std::string &seg_file);

}  // namespace mapping
}  // namespace cyberverse
}  // namespace autobot
