#include <glog/logging.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>

int main(int argc, char** argv) {
  // convert acii pcd to binary pcd
  if (argc < 2) {
    LOG(INFO) << "usage: <excutable file> acii'sPCDfile..";
    return -1;
  }
  std::string filename = argv[1];
  LOG(INFO) << "load file:" << filename;
  if (!boost::filesystem::exists(filename)) {
    LOG(FATAL) << "the file:" << filename << " not exist!";
    return -1;
  }
  std::string output_filename = filename + "binary";
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile(filename.c_str(), *cloud) == -1) {
    LOG(ERROR) << "could not load the:" << filename;
    return -1;
  }
  LOG(INFO) << "load success, begin to save cloud to binary format..";
  pcl::io::savePCDFileBinary(output_filename, *cloud);
  // pcl::io::savePCDFileBinaryCompressed(output_filename, *cloud);
  LOG(INFO) << "save success.";
  return 0;
}