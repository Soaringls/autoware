#include <glog/logging.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
void LoadPointCloud(const std::string filename, PointCloud::Ptr& cloud) {
  LOG(INFO) << "load file:" << filename;
  if (!boost::filesystem::exists(filename)) {
    LOG(FATAL) << "the file:" << filename << " not exist!";
    // return -1;
  }
  if (pcl::io::loadPCDFile(filename.c_str(), *cloud) == -1) {
    LOG(ERROR) << "could not load the:" << filename;
    // return -1;
  }
}

int main(int argc, char** argv) {
  // convert acii pcd to binary pcd
  LOG(INFO) << "main's argc:" << argc;
  if (argc < 2) {
    LOG(INFO) << "usage: <excutable file> acii'sPCDfile..";
    return -1;
  }
  std::string filename = argv[1];
  std::string output_filename = filename + "binary";
  PointCloud::Ptr cloud1(new PointCloud);
  LoadPointCloud(filename, cloud1);

  if (argc == 3) {
    PointCloud::Ptr cloud2(new PointCloud);
    std::string filename2 = argv[2];
    LoadPointCloud(filename2, cloud2);
    *cloud1 += *cloud2;
  }

  LOG(INFO) << "load success, begin to save cloud to binary format..";
  pcl::io::savePCDFileBinary(output_filename, *cloud1);  // unity3D support
  // pcl::io::savePCDFileBinaryCompressed(output_filename, *cloud);
  // pcl::io::savePCDFileASCII1(filename, *map_filtered);
  LOG(INFO) << "save success.";
  return 0;
}