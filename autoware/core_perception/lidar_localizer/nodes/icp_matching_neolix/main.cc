#include <lidar_alignment/align_pointmatcher.h>
#include <lidar_alignment/lidar_segmentation.h>
#include <lidar_alignment/timer.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloudI;
typedef PointCloudI::Ptr PointCloudPtr;

bool LoadPCDFile(const std::string& path, PointCloudPtr& output) {
  PointCloudPtr cloud(new PointCloudI);
  if (!boost::filesystem::exists(path)) {
    LOG(INFO) << path << " is not exist!!!";
    return false;
  }
  // LOG(INFO) << path << " is loading";
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) == -1) {
    LOG(INFO) << "failed to load cloud:" << path;
    return false;
  }
  std::vector<int> indices;
//   pcl::removeNaNFromPointCloud<PointT>(*cloud, *cloud, indices);
  indices.clear();
  output = cloud;
  return true;
}

int main(){
    // ceres_mapping::lidar_odom::LidarSegmentation test("asdfasdfa");
    AlignPointMatcher::Ptr align_pointmatcher_ptr(new AlignPointMatcher(""));
    std::string map_file = "/autoware/workspace/data/map/map_hengtong/hengtong_1106_local.pcd";
    PointCloudPtr map_ptr(new PointCloudI);
    common::Timer t;
    if(LoadPCDFile(map_file, map_ptr)){
        LOG(WARNING)<<"failed to load:"<<map_file;
    }
    LOG(INFO)<<"load pcd file:"<<map_ptr->size()<<", elapsed time:"<<t.end(true)<<" [ms]";
    LOG(INFO)<<"test elapsed time:"<<t.end(true)<<" [ms]";
    if(!align_pointmatcher_ptr->SetMap<PointT>(map_ptr)){
        LOG(WARNING)<<"failed to load map!";
    }
    LOG(INFO)<<"SetMap, elapsed time:"<<t.end(true)<<" [ms]";
    if(!align_pointmatcher_ptr->setmap(map_file)){
        std::cout<<"failed to setmap";
    }
    LOG(INFO)<<"setmap, elapsed time:"<<t.end(true)<<" [ms]";
    return 0;
}