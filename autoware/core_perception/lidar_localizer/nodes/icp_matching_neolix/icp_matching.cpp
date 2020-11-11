#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <lidar_alignment/align_pointmatcher.h>
#include <lidar_alignment/lidar_segmentation.h>
#include <lidar_alignment/timer.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <autoware_config_msgs/ConfigNDT.h>

#include <autoware_msgs/NDTStat.h>

//headers in Autoware Health Checker
#include <autoware_health_checker/health_checker/health_checker.h>

typedef PointMatcher<double> PM;
typedef PM::DataPoints DP;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;

//global variable
Eigen::Vector3d map_init_pos = Eigen::Vector3d(0,0,0);

// PM::ICPSequence icp_sequence;
AlignPointMatcher::Ptr align_pointmatcher_ptr;

void SetMapInitPos(const std::string& map_init_filename){
  if(!boost::filesystem::exists(map_init_filename)){
    LOG(FATAL)<<"file:"<<map_init_filename<<" is not exist!";
  }
  std::ifstream ifs(map_init_filename, std::ios::in);
  std::string line="";
  while(getline(ifs, line)){
    if(line.empty()) continue;
    std::vector<std::string> parts;
    boost::split(parts, line, boost::is_any_of(","));
    if(parts.empty()){
      LOG(FATAL)<<"there is empty in the file!";
    }
    map_init_pos = Eigen::Vector3d(std::stod(parts[0]), std::stod(parts[1]), std::stod(parts[2]));
  }
  ifs.close();
}
bool LoadPCDFile(const std::string& path, 
                 PointCloudPtr& output) {
  PointCloudPtr cloud(new pcl::PointCloud<PointT>);
  if (!boost::filesystem::exists(path)) {
    LOG(INFO) << path << " is not exist!!!";
    return false;
  }
  // LOG(INFO) << path << " is loading";
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) == -1) {
    LOG(INFO) << "failed to load cloud:" << path;
    return false;
  }
  PointCloudPtr filtered(new pcl::PointCloud<PointT>);
  for(int i =0; i < cloud->size(); ++i) {
      auto pt = cloud->points[i];
      if(std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) || 
         std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z)) continue;
      filtered->push_back(pt);
  }
  output = filtered;
  return true;
}

int main(int argc, char** argv){
    ros::init(argc, argv, "icp_locator");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string map_init_config_file;
    private_nh.getParam("map_init_config", map_init_config_file);
    SetMapInitPos(map_init_config_file);

    align_pointmatcher_ptr.reset(new AlignPointMatcher("_icp_config"));
    //test start
    std::string map_file = "/autoware/workspace/data/map/map_hengtong/hengtong_1106_local.pcd";
    // PointCloudPtr map_ptr(new pcl::PointCloud<PointT>);
    // LOG(INFO)<<"begin to load map:"<<map_file;
    // if(LoadPCDFile(map_file, map_ptr)){
    //     LOG(WARNING)<<"failed to load:"<<map_file;
    // }
    // LOG(INFO)<<"load map success..:"<<map_ptr->size();
    // if(!align_pointmatcher_ptr->SetMap<PointT>(map_ptr)){
    //     LOG(WARNING)<<"failed to load map!";
    // }
    PM::ICPSequence icp_sequence;
    icp_sequence.setDefault();
    LOG(INFO)<<"begin to load map file:"<<map_file;
    DP map_pts(DP::load(map_file));
    LOG(INFO)<<"set pcd to datapoints success.";
    icp_sequence.setMap(map_pts);
    LOG(INFO)<<"set map success.";
    
    return 0;
}