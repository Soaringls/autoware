#define OUTPUT

#include <fstream>
#include <iostream>
#include <string>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/octree/octree_search.h>
// #define CUDA_FOUND
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif


#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <autoware_config_msgs/ConfigNDTMapping.h>
#include <autoware_config_msgs/ConfigNDTMappingOutput.h>

#include "base_impl/ceres_optimizer.h"
#include "base_impl/align_pointmatcher.h"
#include "base_impl/lidar_segmentation.h"


typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT>      PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

enum class MethodType{
    PCL_GENERIC = 0,
    PCL_ANH = 1,
    PCL_ANH_GPU = 2,
    PCL_OPENMP = 3, 
    ICP_ETH = 4,
};

struct Config{
    double time_synchronize_threshold = 0.1; //100ms
    //filter params
    double lidar_yaw_calib_degree = 90.0;
    double distance_near_thresh = 1.0;
    double distance_far_thresh = 50;
    double voxel_filter_size = 0.2;
    //registeration params
    double frame_dist = 0.1;
    MethodType register_method_type = MethodType::PCL_GENERIC;
    std::string seg_config_file;
    std::string icp_config_file;
    double ndt_trans_epsilon = 0.01;
    double ndt_step_size = 0.1;
    double ndt_resolution = 1.0;
    double ndt_maxiterations = 64;
    //back-end optimization params
    CeresConfig ceres_config;
    //mapping
    double keyframe_delta_trans = 8;
    double map_cloud_update_interval = 10.0;

}config;

struct KeyFrame{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloudPtr cloud;
    Eigen::Isometry3d pose;
    KeyFrame(){}
    KeyFrame(const Eigen::Isometry3d& pose, const PointCloudPtr& cloud):pose(pose),cloud(cloud){}
};
typedef std::shared_ptr<KeyFrame> KeyFramePtr;
typedef std::shared_ptr<const KeyFrame> KeyFrameConstPtr;

//global variables
static PointCloud map_autoware;
std::mutex gps_mutex, keyframe_mutex;
Eigen::Affine3d init_pose =  Eigen::Affine3d::Identity();

std::queue<geometry_msgs::PoseStamped::ConstPtr> gps_msgs;
pcl::VoxelGrid<PointT> voxel_filter;
std::vector<KeyFramePtr> key_frame_ptrs;

pcl::NormalDistributionsTransform<PointT, PointT> ndt_matching;
AlignPointMatcher::Ptr align_pointmatcher_ptr;
CeresOptimizer ceres_optimizer;
ceres_mapping::lidar_odom::LidarSegmentationPtr lidar_segmentation_ptr;

visualization_msgs::MarkerArray marker_array;

//publisher
ros::Publisher ins_pos_pub, lio_pos_pub, optimized_pos_pub;
ros::Publisher pub_filtered, pub_map;


#ifdef CUDA_FOUND
static gpu::GNormalDistributionsTransform ndt_gpu;
#endif

//todo
void PublisherVisualiztion(ros::Publisher& pub, POSEPtr pose, int color[],
                           std::string model = ""){
    visualization_msgs::Marker marker_msg;

    marker_msg.header.stamp = ros::Time(pose->time);
    marker_msg.header.frame_id = "world";
    marker_msg.action = visualization_msgs::Marker::ADD;
    marker_msg.id = 0;
    if(!model.empty()) {
        marker_msg.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker_msg.mesh_resource = "/autoware/workspace/data/models/car.dae";
    }

}
void SetCloudAttributes(PointCloudPtr& cloud, std::string frame_id = "world"){
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;
    // cloud->header = input->header;
    cloud->header.frame_id = frame_id;
}

Eigen::Isometry3d Matrix2Isometry(const Eigen::Matrix4d matrix) {
  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = matrix.block<3,3>(0,0);
  isometry.translation() = matrix.block<3,1>(0,3);
  return isometry;
}

void InsCallback(const geometry_msgs::PoseStamped::ConstPtr &msgs){
   gps_mutex.lock();
   if(gps_msgs.size() > 16000) gps_msgs.pop();
   gps_msgs.push(msgs);
   gps_mutex.unlock();

   POSEPtr ins_factor(new POSE);
   ins_factor->time = msgs->header.stamp.toSec();
   ins_factor->pos = Eigen::Vector3d(msgs->pose.position.x,msgs->pose.position.y,msgs->pose.position.z);
   ins_factor->q = Eigen::Quaterniond(msgs->pose.orientation.w, msgs->pose.orientation.x, msgs->pose.orientation.y, msgs->pose.orientation.z);

   static nav_msgs::Path gps_path;
   ceres_optimizer.AddPoseToNavPath(gps_path, ins_factor);
   ins_pos_pub.publish(gps_path);
}

PointCloudPtr FilterCloudFrame(PointCloudPtr &input, const double leaf_size){
    //nan filter
    std::vector<int> indices;
    // ROS_INFO_STREAM("filter_nan=====1:"<<input->size());
    pcl::removeNaNFromPointCloud<PointT>(*input, *input, indices);
    PointCloudPtr filtered(new PointCloud);
    for(int i =0; i < input->size(); ++i) {
        auto pt = input->points[i];
        if(std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) || 
           std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z)) continue;
        filtered->push_back(pt);
    }
    // ROS_INFO_STREAM("filter_nan=====2:"<<filtered->size());

    //voxel filter
    PointCloudPtr voxel_filtered_pts(new PointCloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.setInputCloud(filtered);
    voxel_filter.filter(*voxel_filtered_pts);
    //distance filter
    filtered->clear();
    filtered->reserve(voxel_filtered_pts->size());
    std::copy_if(voxel_filtered_pts->begin(), voxel_filtered_pts->end(), std::back_inserter(filtered->points),
      [&](const PointT& pt){
          double dist = pt.getVector3fMap().norm();
          return dist > config.distance_near_thresh && dist < config.distance_far_thresh;
      }
    );

    filtered->header = input->header;
    SetCloudAttributes(filtered);

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.rotate (Eigen::AngleAxisd (config.lidar_yaw_calib_degree*M_PI/180, Eigen::Vector3d::UnitZ()));
    pcl::transformPointCloud(*filtered, *filtered, transform);
    return filtered;
}

void AddKeyFrame(const Eigen::Matrix4d& tf, const Eigen::Matrix4d& pose, const PointCloudPtr cloud){
    PointCloudPtr current_frame(new PointCloud);
    pcl::transformPointCloud(*cloud, *current_frame, init_pose.inverse() * pose);
    sensor_msgs::PointCloud2::Ptr output(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*current_frame, *output);
    pub_filtered.publish(output);
 
    static Eigen::Matrix4d pre_keyframe_pos = pose;
    auto tf_keyframe = pre_keyframe_pos * tf;
    auto distance = tf_keyframe.block<3,1>(0,3).norm();
    if(distance < config.keyframe_delta_trans) return;
    pre_keyframe_pos = pose;
    CHECK_NOTNULL(cloud);
    Eigen::Isometry3d key_pose = Eigen::Isometry3d::Identity();
    key_pose.linear() = pose.block<3,3>(0,0);
    key_pose.translation() = pose.block<3,1>(0,3);
    KeyFramePtr key_frame(new KeyFrame(key_pose, cloud));
    keyframe_mutex.lock();
    key_frame_ptrs.push_back(key_frame);
    keyframe_mutex.unlock();
    ROS_INFO_STREAM("add keyframe at pos: "<<std::fixed<<std::setprecision(6)<<
                    pose(0,3)<<", "<<pose(1,3)<<", "<<pose(2,3));
}

PointCloudPtr GenerateMap(double resolution = 0.05){
    if(key_frame_ptrs.empty()) return nullptr;

    PointCloudPtr map_cloud(new PointCloud);
    map_cloud->reserve(key_frame_ptrs.size() * key_frame_ptrs.front()->cloud->size());
    auto init_pos = Matrix2Isometry(init_pose.matrix());
    for(const auto& frame : key_frame_ptrs){
        Eigen::Isometry3d pose = (init_pos.inverse()) * frame->pose;//local coordinate system
        PointCloudPtr temp_cloud(new PointCloud);
        pcl::transformPointCloud(*(frame->cloud), *temp_cloud, pose.matrix());
        *map_cloud += *temp_cloud;
    }
    map_cloud->header = key_frame_ptrs.back()->cloud->header;
    SetCloudAttributes(map_cloud);

    pcl::octree::OctreePointCloud<PointT> octree(resolution);
    octree.setInputCloud(map_cloud);
    octree.addPointsFromInputCloud();
  
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    octree.getOccupiedVoxelCenters(filtered->points);
    filtered->header = map_cloud->header;
    SetCloudAttributes(filtered);

    return filtered;
}

bool RegisterPointCloud(const PointCloudPtr& target, const PointCloudPtr& source, Eigen::Matrix4d& matrix, double& score){
    PointCloudPtr output_cloud(new PointCloud);
    bool is_converged(true);
    std::cout<<"matrix1:"<<std::fixed<<std::setprecision(6)<<matrix(0,3)<<", "<<matrix(1,3)<<", "<<matrix(2,3)<<std::endl;

    if(config.register_method_type == MethodType::PCL_GENERIC) {
        ndt_matching.setInputTarget(target);
        ndt_matching.setInputSource(source);
        Eigen::Matrix4f init_guss = matrix.cast<float>();
        ndt_matching.align(*output_cloud, init_guss);
        init_guss = ndt_matching.getFinalTransformation();
        matrix = init_guss.cast<double>();
        score =  ndt_matching.getFitnessScore();
        std::cout<<"MethodType::PCL_GENERIC score:"<<score<<std::endl;
        is_converged = ndt_matching.hasConverged();
    } 
    #ifdef CUDA_FOUND
    else if(config.register_method_type == MethodType::PCL_ANH_GPU){
       ndt_gpu.setInputTarget(target);
       ndt_gpu.setInputSource(source); 
       ndt_gpu.align(matrix); 
       matrix = ndt_matching.getFinalTransformation();
       score =  ndt_matching.getFitnessScore();
       return   ndt_matching.hasConverged();
    }
    #endif

    if(config.register_method_type == MethodType::ICP_ETH){
        if (!align_pointmatcher_ptr->Align<PointT>(target, source,
                                                  matrix, score)){
            score = std::numeric_limits<double>::max();
        }
        std::cout<<"MethodType::ICP_ETH score:"<<score<<std::endl;
    }
    
    std::cout<<"matrix2:"<<std::fixed<<std::setprecision(6)<<matrix(0,3)<<", "<<matrix(1,3)<<", "<<matrix(2,3)<<std::endl;
    return is_converged;
}

PointCloudPtr SegmentedCloud(PointCloudPtr cloud){
    auto header = cloud->header;
    lidar_segmentation_ptr->CloudMsgHandler(cloud);
    auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloud();
    // auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloudPure();
    seg_cloud->header = header;
    return seg_cloud;
}

bool AlignPointCloud(const PointCloudPtr& cloud_in, Eigen::Matrix4d& matrix, double& score){
    static bool init(false);
    static uint32_t frame_cnt(1);
    PointCloudPtr source_ptr(new PointCloud);
    pcl::copyPointCloud(*cloud_in, *source_ptr);
    // source_ptr = SegmentedCloud(source_ptr);

    static PointCloudPtr target_ptr(new PointCloud);
    if(!init){
      pcl::copyPointCloud(*source_ptr, *target_ptr);
      init = true;
      return false;
    }

    PointCloudPtr output(new PointCloud);
    bool is_converged(true);
    if(config.register_method_type == MethodType::PCL_GENERIC) {
        Eigen::Matrix4f init_guss = matrix.cast<float>();
        ndt_matching.setInputTarget(target_ptr);
        ndt_matching.setInputSource(source_ptr);       
        ndt_matching.align(*output, init_guss);
        init_guss = ndt_matching.getFinalTransformation();
        matrix = init_guss.cast<double>();
        score =  ndt_matching.getFitnessScore();
        is_converged = ndt_matching.hasConverged();
        ROS_INFO_STREAM("MethodType::PCL_GENERIC score:"<<score);
    } else if(config.register_method_type == MethodType::ICP_ETH){
        if (!align_pointmatcher_ptr->Align<PointT>(target_ptr, source_ptr, matrix, score)){
            score = std::numeric_limits<double>::max();
        }
        ROS_INFO_STREAM("MethodType::ICP_ETH score:"<<score);
    }
    ROS_INFO_STREAM("frame:["<<std::fixed<<std::setprecision(6)<<"id:"<<frame_cnt<<"-pts:"<<target_ptr->size()
                    <<" --> "<<"id:"<<frame_cnt++<<"-pts:"<<source_ptr->size()<<"], score:"<<score);
    target_ptr->clear();
    pcl::copyPointCloud(*source_ptr, *target_ptr);
    return is_converged;
}

void OuputPointCloud(const PointCloudPtr& cloud){
    for(int i = 0; i < 20; i++){
        ROS_INFO_STREAM("pts:"<<std::fixed<<std::setprecision(6)<<
           cloud->points[i].x<<", "<<cloud->points[i].y<<", "<<cloud->points[i].z);
    }
}

bool FindCorrespondGpsMsg(const double pts_stamp, POSEPtr& ins_pose) {
    bool is_find(false);
    gps_mutex.lock();
    while(!gps_msgs.empty()){
        auto stamp_diff = gps_msgs.front()->header.stamp.toSec() - pts_stamp;
        if(std::abs(stamp_diff) <= config.time_synchronize_threshold) {
            POSEPtr pose(new POSE);
            pose->time = pts_stamp;
            auto gps_msg = gps_msgs.front();
            pose->pos = Eigen::Vector3d(gps_msg->pose.position.x, gps_msg->pose.position.y, gps_msg->pose.position.z);
            pose->q = Eigen::Quaterniond(gps_msg->pose.orientation.w, gps_msg->pose.orientation.x, gps_msg->pose.orientation.y, gps_msg->pose.orientation.z);
            ins_pose = pose;
            is_find = true;
            break;
        } else if(stamp_diff < -config.time_synchronize_threshold) {
            gps_msgs.pop();
        } else if(stamp_diff > config.time_synchronize_threshold) {
            ROS_INFO_STREAM("(gps_time - pts_time = "<<stamp_diff<<") lidar msgs is delayed! ");
            break;
        } 
    }
    gps_mutex.unlock();
    return is_find;
}


void PointsCallback(const sensor_msgs::PointCloud2::ConstPtr &input){
    static bool init_pose_flag(false);
    if(!init_pose_flag && gps_msgs.empty()) {
        ROS_INFO_STREAM("waiting to initizlizing by gps...");
        return;
    }

    static std::size_t frame_cnt(1);
    double timestamp = input->header.stamp.toSec();

    PointCloudPtr point_msg(new PointCloud);
    pcl::fromROSMsg(*input, *point_msg);
    point_msg = FilterCloudFrame(point_msg, config.voxel_filter_size);
    if(point_msg->size() == 0) return;

    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
    double fitness_score = std::numeric_limits<double>::max();

    static Eigen::Matrix4d lio_pose = Eigen::Matrix4d::Identity();
    
    if(!init_pose_flag)
    { 
        if(!AlignPointCloud(point_msg, tf, fitness_score)){
            ROS_INFO_STREAM("AlignPointCloud init success.");
        }
        POSEPtr init_odom;
        if(!FindCorrespondGpsMsg(timestamp, init_odom)){
            ROS_INFO_STREAM("PointCloud Callback init failed!!!");
            return;
        }
        
        CHECK_NOTNULL(init_odom);
        ceres_optimizer.InsertOdom(init_odom);
        init_pose = POSE2Affine(init_odom);
        lio_pose = init_pose.matrix();

        init_pose_flag = true;
        return;    
    }
    
    //find real-time gps-pose
    POSEPtr ins_pose;
    if(!FindCorrespondGpsMsg(timestamp, ins_pose)){
        ROS_INFO_STREAM("failed to search ins_pose at:"<<std::fixed<<std::setprecision(6)<<timestamp);
        return;
    }
    CHECK_NOTNULL(ins_pose);

    Eigen::Matrix4d trans = lio_pose.inverse() * POSE2Affine(ins_pose).matrix();
    auto dist = trans.block<3,1>(0,3).norm();
    if(dist < config.frame_dist) {
        ROS_INFO_STREAM("the distance to last frame is :"<<dist<<" too small!!!");
        return;
    }

    if(!AlignPointCloud(point_msg, tf, fitness_score)){
        ROS_WARN_STREAM("regis between:("<<frame_cnt-1<<", "<<frame_cnt<<") is failed!!!");
    }

    //insert odom-trans to optimizer
    POSEPtr lio_factor(new POSE);
    lio_factor->time = timestamp;
    lio_factor->pos = Eigen::Vector3d(tf(0,3), tf(1,3), tf(2,3));
    lio_factor->q   = Eigen::Quaterniond(tf.block<3,3>(0,0));
    lio_factor->score = fitness_score;
    ceres_optimizer.InsertOdom(lio_factor);
    
    lio_pose *= tf;
    //add keyframe TODO
    point_msg = SegmentedCloud(point_msg); //test the lidar-seg
    AddKeyFrame(tf, lio_pose, point_msg);    
    
    //insert gps data to optimizer
    if(frame_cnt++ % config.ceres_config.num_every_scans == 0) {
        ROS_INFO_STREAM("Anchor is to correct the pose....");
        ceres_optimizer.InsertGPS(ins_pose);
    }    

    //publish lidar_odometry's path
    static nav_msgs::Path odom_path;
    POSEPtr odom_pose(new POSE);
    odom_pose->time = timestamp;
    odom_pose->pos = Eigen::Vector3d(lio_pose(0,3), lio_pose(1,3), lio_pose(2,3));
    odom_pose->q = Eigen::Quaterniond(lio_pose.block<3, 3>(0, 0));
    ceres_optimizer.AddPoseToNavPath(odom_path, odom_pose);
    lio_pos_pub.publish(odom_path);
    ROS_INFO_STREAM("Num of Lidar-odom  pose:"<<odom_path.poses.size());

    //publish optimized path
    static nav_msgs::Path *optimized_path = &ceres_optimizer.optimized_path;
    if(!optimized_path->poses.empty()){
        optimized_pos_pub.publish(*optimized_path);
        ROS_INFO_STREAM("Num of Optimized's pose:"<<optimized_path->poses.size());
    }
}

void MappingTimerCallback(const ros::WallTimerEvent& event){
    if(!pub_map.getNumSubscribers()) return;
    keyframe_mutex.lock();
    auto map_cloud = GenerateMap();
    keyframe_mutex.unlock();
    if(!map_cloud) return;

    sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*map_cloud, *cloud_msg);

    pub_map.publish(cloud_msg);
}
static void MappingCallback(const geometry_msgs::PoseStamped::ConstPtr &gps, 
                            const sensor_msgs::PointCloud2::ConstPtr &pts){
    static int msg_cnt(0);
    double stamp_gps = gps->header.stamp.toSec();
    double stamp_pts = pts->header.stamp.toSec();
    ROS_INFO_STREAM("timestamp between gps and pts:["<<std::fixed<<std::setprecision(6)<<msg_cnt++<<"]"<<
                    stamp_gps - stamp_pts);

    Eigen::Affine3d pose = Eigen::Translation3d(gps->pose.position.x, gps->pose.position.y, gps->pose.position.z)*
                        Eigen::Quaterniond(gps->pose.orientation.w, gps->pose.orientation.x, gps->pose.orientation.y, gps->pose.orientation.z);
    if(msg_cnt == 1) init_pose = pose;

    PointCloudPtr point_msg(new PointCloud);
    pcl::fromROSMsg(*pts, *point_msg);
    if(point_msg->size() == 0) return;
    // point_msg = FilterCloudFrame(point_msg, config.voxel_filter_size);

    PointCloudPtr transformed_scan_ptr(new PointCloud);    
    pcl::transformPointCloud(*point_msg, *transformed_scan_ptr, pose.matrix());  
    map_autoware += *transformed_scan_ptr;
    
    map_autoware.width = map_autoware.size();
    ROS_INFO_STREAM("current pts and map's size:"<<point_msg->size()<<", "<<map_autoware.size());
}

static void output_callback(const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input){
    if(map_autoware.size() == 0) {
        ROS_INFO_STREAM("map is empty()!!!");
        return;
    }

    double filter_res = input->filter_res;
  std::string filename = input->filename;
  std::cout << "output_callback" << std::endl;
  std::cout << "filter_res: " << filter_res << std::endl;
  std::cout << "filename: " << filename << std::endl;

  pcl::PointCloud<PointT>::Ptr map_ptr(new pcl::PointCloud<PointT>(map_autoware));
  pcl::PointCloud<PointT>::Ptr map_filtered(new pcl::PointCloud<PointT>());
  map_ptr->header.frame_id = "map";
  map_filtered->header.frame_id = "map";
  
//   pcl::transformPointCloud(*map_ptr, *map_ptr, init_pose.matrix().inverse());
//   for(const auto&pt : map_ptr->points){
//     std::cout<<"map_pt:"<<pt.x<<" "<<pt.y<<" "<<pt.z<<std::endl;
//   }
//   // Apply voxelgrid filter
  map_filtered = FilterCloudFrame(map_ptr, filter_res);
  ROS_INFO_STREAM("Map--Original: " << map_ptr->points.size() << " points." );
  ROS_INFO_STREAM("Map--Filtered: " << map_filtered->points.size() << " points.");

//   sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
//   pcl::toROSMsg(*map_ptr, *map_msg_ptr);
//   pub_map.publish(map_msg_ptr);
  std::string filename_filtered = filename + "_filtered";
  pcl::io::savePCDFileASCII(filename, *map_ptr);
  pcl::io::savePCDFileASCII(filename_filtered, *map_filtered);
  ROS_INFO_STREAM("Saved " << map_filtered->points.size() << " data points to " << filename << " success!!!");
}

int main(int argc, char** argv) {
   ros::init(argc, argv, "ceres_mapping");

   ros::NodeHandle nh;
   ros::NodeHandle private_nh("~");
   {
       //params timestamp thershold
       private_nh.getParam("gps_lidar_time_threshold",config.time_synchronize_threshold);
       //params filter
       private_nh.getParam("lidar_yaw_calib_degree", config.lidar_yaw_calib_degree);
       private_nh.getParam("voxel_filter_size", config.voxel_filter_size);
       private_nh.getParam("distance_near_thresh", config.distance_near_thresh);
       private_nh.getParam("distance_far_thresh", config.distance_far_thresh);
       //params registration
       private_nh.getParam("segment_config_file", config.seg_config_file);
       private_nh.getParam("icp_config_file", config.icp_config_file);
       private_nh.getParam("frame_dist", config.frame_dist);
       int ndt_method = 0;
       private_nh.getParam("method_type", ndt_method);
       config.register_method_type  = static_cast<MethodType>(ndt_method);
       private_nh.getParam("ndt_trans_epsilon", config.ndt_trans_epsilon);
       private_nh.getParam("ndt_step_size",     config.ndt_step_size);
       private_nh.getParam("ndt_resolution",    config.ndt_resolution);
       private_nh.getParam("ndt_maxiterations", config.ndt_maxiterations);
       //params ceres
       private_nh.getParam("optimize_num_every_scans", config.ceres_config.num_every_scans);
       private_nh.getParam("optimize_iter_num",   config.ceres_config.iters_num);
       private_nh.getParam("optimize_var_anchor", config.ceres_config.var_anchor);
       private_nh.getParam("optimize_var_odom_t", config.ceres_config.var_odom_t);
       private_nh.getParam("optimize_var_odom_q", config.ceres_config.var_odom_q);
       //mapping params  
       private_nh.getParam("map_cloud_update_interval", config.map_cloud_update_interval);
       private_nh.getParam("keyframe_delta_trans", config.keyframe_delta_trans);
       //log of config
       ROS_INFO_STREAM("***********config params***********");
       ROS_INFO_STREAM("gps_lidar_time_threshold:"<<config.time_synchronize_threshold);
       ROS_INFO_STREAM("voxel_filter_size  :"<<     config.voxel_filter_size);
       ROS_INFO_STREAM("ndt_trans_epsilon  :"<< config.ndt_trans_epsilon);
       ROS_INFO_STREAM("ndt_step_size      :"<< config.ndt_step_size);
       ROS_INFO_STREAM("ndt_resolution     :"<< config.ndt_resolution);
       ROS_INFO_STREAM("ndt_maxiterations  :"<< config.ndt_maxiterations);
       ROS_INFO_STREAM("optimize_iter_num  :"<< config.ceres_config.iters_num);
       ROS_INFO_STREAM("optimize_var_anchor:"<< config.ceres_config.var_anchor);
       ROS_INFO_STREAM("optimize_var_odom_t:"<< config.ceres_config.var_odom_t);
       ROS_INFO_STREAM("optimize_var_odom_q:"<< config.ceres_config.var_odom_q);
    }
       

   if(config.register_method_type == MethodType::PCL_GENERIC) {
       ndt_matching.setTransformationEpsilon(config.ndt_trans_epsilon);
    //    ndt_matching.setStepSize(config.ndt_step_size);
       ndt_matching.setResolution(config.ndt_resolution);
       ndt_matching.setMaximumIterations(config.ndt_maxiterations);
    } 
    #ifdef CUDA_FOUND
    else if(config.register_method_type == MethodType::PCL_ANH_GPU){
       ndt_gpu.setTransformationEpsilon(ndt_trans_epsilon);
       ndt_gpu.setStepSize(ndt_step_size);
       ndt_gpu.setResolution(ndt_resolution);
       ndt_gpu.setMaximumIterations(ndt_maxiterations);
    }
    #endif
   
   lidar_segmentation_ptr.reset(new ceres_mapping::lidar_odom::LidarSegmentation(config.seg_config_file));
   align_pointmatcher_ptr.reset(new AlignPointMatcher(config.icp_config_file));
   ceres_optimizer.SetConfig(config.ceres_config);
   
   //sub and pub
   ins_pos_pub = nh.advertise<nav_msgs::Path>("/mapping/path/gps", 100);
   lio_pos_pub     = nh.advertise<nav_msgs::Path>("/mapping/path/odom",100);
   optimized_pos_pub     = nh.advertise<nav_msgs::Path>("/mapping/path/optimized",100);
   pub_filtered = nh.advertise<sensor_msgs::PointCloud2>("/mapping/filtered_frame", 10);

   ros::Subscriber points_sub = nh.subscribe("/points_raw", 10000, PointsCallback);
   ros::Subscriber gnss_sub   = nh.subscribe("/gnss_pose",  16000,  InsCallback);

   //mapping
   ros::WallTimer map_publish_timer = nh.createWallTimer(ros::WallDuration(config.map_cloud_update_interval), MappingTimerCallback);
   pub_map = nh.advertise<sensor_msgs::PointCloud2>("/mapping/localizer_map", 1);
   ros::Subscriber output_sub = nh.subscribe("config/ndt_mapping_output", 10, output_callback);
   
   ros::spin();
   return 0;
}