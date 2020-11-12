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
    double stamp = 0.0;
    KeyFrame(){}
    KeyFrame(const double& stamp, const Eigen::Isometry3d& pose, const PointCloudPtr& cloud)
        :stamp(stamp), pose(pose),cloud(cloud){}
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
   if(gps_msgs.size() > 30000) gps_msgs.pop();
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

void FilterByDistance(PointCloudPtr cloud){
    auto pts_num = cloud->size();
    PointCloudPtr filtered(new PointCloud);
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points),
      [&](const PointT& pt){
          double dist = pt.getVector3fMap().norm();
          return dist > config.distance_near_thresh && dist < config.distance_far_thresh;
      }
    );
    filtered->header = cloud->header;
    SetCloudAttributes(filtered);
    auto pts_num_filtered = filtered->size();
    LOG(INFO)<<std::fixed<<std::setprecision(6)<<"distance filtered:"<<pts_num-pts_num_filtered;
}
PointCloudPtr FilterCloudFrame(PointCloudPtr &input, const double leaf_size){
    //nan filter
    std::vector<int> indices;
    // LOG(INFO)<<"filter_nan=====1:"<<input->size();
    pcl::removeNaNFromPointCloud<PointT>(*input, *input, indices);
    PointCloudPtr filtered(new PointCloud);
    for(int i =0; i < input->size(); ++i) {
        auto pt = input->points[i];
        if(std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) || 
           std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z)) continue;
        filtered->push_back(pt);
    }
    // LOG(INFO)<<"filter_nan=====2:"<<filtered->size();

    //voxel filter
    PointCloudPtr voxel_filtered_pts(new PointCloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.setInputCloud(filtered);
    voxel_filter.filter(*voxel_filtered_pts);

    voxel_filtered_pts->header = input->header;
    SetCloudAttributes(voxel_filtered_pts);
    return voxel_filtered_pts;
}

void AddKeyFrame(const double& time, const Eigen::Matrix4d& tf, const Eigen::Matrix4d& pose, 
                 const PointCloudPtr cloud){
    //publish current frame
    PointCloudPtr current_frame(new PointCloud);
    Eigen::Matrix4d visual_pose = pose;
    visual_pose(0,3) -= init_pose.matrix()(0,3);
    visual_pose(1,3) -= init_pose.matrix()(1,3);
    visual_pose(2,3) -= init_pose.matrix()(2,3);
    pcl::transformPointCloud(*cloud, *current_frame, visual_pose);
    sensor_msgs::PointCloud2::Ptr output(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*current_frame, *output);
    pub_filtered.publish(output);
    
    //check if the condition is meet
    static Eigen::Matrix4d pre_keyframe_pos = pose;
    static Eigen::Matrix4d frame_pose = pose;
    frame_pose *= tf;
    auto matrix_dist = pre_keyframe_pos.inverse() * frame_pose;
    auto distance = matrix_dist.block<3,1>(0,3).norm();
    if(distance < config.keyframe_delta_trans) {
        LOG(INFO)<<"Add KeyFrame failed. the distance to last keyframe:"<<distance
                 <<", threshold:"<<config.keyframe_delta_trans;
        return;
    }
    pre_keyframe_pos = pose;
    frame_pose = pose;

    CHECK_NOTNULL(cloud);
    Eigen::Isometry3d key_pose = Eigen::Isometry3d::Identity();

    key_pose.linear() = pose.block<3,3>(0,0);
    key_pose.translation() = pose.block<3,1>(0,3);
    KeyFramePtr key_frame(new KeyFrame(time, key_pose, cloud));
    keyframe_mutex.lock();
    key_frame_ptrs.push_back(key_frame);
    keyframe_mutex.unlock();
    LOG(INFO)<<"add keyframe's dist:"<<std::fixed<<std::setprecision(6)
             <<distance<<", threshold:"<<config.keyframe_delta_trans
             <<", at pos: "<<pose(0,3)<<", "<<pose(1,3)<<", "<<pose(2,3);
}

PointCloudPtr GenerateMap(double resolution = 0.05){
    if(key_frame_ptrs.empty()) return nullptr;
    
    //update optimized pose for keyframe
    int i(0);
    auto optimized_poses = ceres_optimizer.GetOptimizedResult();
    LOG(INFO)<<"Generating optimized_pose:"<<optimized_poses.size();
    // keyframe_mutex.lock();
    for(auto& keyframe : key_frame_ptrs){
        // double stamp = keyframe->cloud->header.stamp.toSec();
        for(; i < optimized_poses.size(); ++i){
            auto elem = optimized_poses[i];
            // LOG(INFO)<<"keyframe's stamp:"<<std::fixed<<std::setprecision(6)
            //          <<keyframe->stamp<<", optimized_poses["<<i<<"]:"<<elem->time
            //          <<" diff(frame-optimized):"<<keyframe->stamp - elem->time;
            if(keyframe->stamp != elem->time) continue;
            keyframe->pose.linear() = elem->q.toRotationMatrix();
            keyframe->pose.translation() = elem->pos;
            // LOG(INFO)<<"keyframe update optimized pose:"<<elem->pos[0]<<", "<<elem->pos[1]<<", "<<elem->pos[2];
            break;
        }
    }

    PointCloudPtr map_cloud(new PointCloud);
    map_cloud->reserve(key_frame_ptrs.size() * key_frame_ptrs.front()->cloud->size());
    // auto init_pos = Matrix2Isometry(init_pose.matrix()); 

    for(const auto& frame : key_frame_ptrs){
        // Eigen::Isometry3d pose = (init_pos.inverse()) * frame->pose;//local coordinate system
        Eigen::Isometry3d pose = frame->pose;//global coordinate system
        // LOG(INFO)<<"keyframe's mapping pose:"<<std::fixed<<std::setprecision(6)
        //          <<pose.matrix()(0,3)<<", "<<pose.matrix()(1,3)<<", "<<pose.matrix()(2,3)
        //          <<" init pos:"<<init_pose.matrix()(0,3)<<", "<<init_pose.matrix()(1,3)<<", "<<init_pose.matrix()(2,3);
        PointCloudPtr temp_cloud(new PointCloud);
        //transfer to local coordinate system
        pose.matrix()(0,3) -= init_pose.matrix()(0,3);
        pose.matrix()(1,3) -= init_pose.matrix()(1,3);
        pose.matrix()(2,3) -= init_pose.matrix()(2,3);
        // LOG(INFO)<<std::fixed<<std::setprecision(6)
        //          <<"mapping pose:"<<pose.matrix()(0,3)<<", "<<pose.matrix()(2,3);
        pcl::transformPointCloud(*(frame->cloud), *temp_cloud, pose.matrix());
        *map_cloud += *temp_cloud;
    }
    map_cloud->header = key_frame_ptrs.back()->cloud->header;
    // keyframe_mutex.unlock();

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
    // auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloud();
    auto seg_cloud = lidar_segmentation_ptr->GetSegmentedCloudPure();
    seg_cloud->header = header;
    return seg_cloud;
}

bool AlignPointCloud(const PointCloudPtr& cloud_in, Eigen::Matrix4d& matrix, double& score){
    static bool init(false);
    static uint32_t frame_cnt(1);
    PointCloudPtr source_ptr(new PointCloud);
    pcl::copyPointCloud(*cloud_in, *source_ptr);
    source_ptr = SegmentedCloud(source_ptr);

    static PointCloudPtr target_ptr(new PointCloud);
    if(!init){
      pcl::copyPointCloud(*source_ptr, *target_ptr);
      init = true;
      return false;
    }
    common::Timer t;
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
        // LOG(INFO)<<"MethodType::PCL_GENERIC score:"<<score;
    } else if(config.register_method_type == MethodType::ICP_ETH){
        if (!align_pointmatcher_ptr->Align<PointT>(target_ptr, source_ptr, matrix, score)){
            score = std::numeric_limits<double>::max();
        }
        // LOG(INFO)<<"MethodType::ICP_ETH score:"<<score;
    }
    LOG(INFO)<<"frame:["<<std::fixed<<std::setprecision(6)<<"id:"<<frame_cnt<<"-pts:"<<target_ptr->size()
                    <<" --> "<<"id:"<<frame_cnt++<<"-pts:"<<source_ptr->size()<<"], score:"<<score<<
                    ", elspaed's time:"<<t.end()<<" [ms]";
    target_ptr->clear();
    pcl::copyPointCloud(*source_ptr, *target_ptr);
    return is_converged;
}

void OuputPointCloud(const PointCloudPtr& cloud){
    for(int i = 0; i < 20; i++){
        LOG(INFO)<<"pts:"<<std::fixed<<std::setprecision(6)<<
           cloud->points[i].x<<", "<<cloud->points[i].y<<", "<<cloud->points[i].z;
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
            LOG(INFO)<<"(gps_time - pts_time = "<<stamp_diff<<") lidar msgs is delayed! ";
            break;
        } 
    }
    gps_mutex.unlock();
    return is_find;
}


void PointsCallback(const sensor_msgs::PointCloud2::ConstPtr &input){
    static bool init_pose_flag(false);
    if(!init_pose_flag && gps_msgs.empty()) {
        LOG(INFO)<<"waiting to initizlizing by gps...";
        return;
    }

    static std::size_t frame_cnt(1);
    double timestamp = input->header.stamp.toSec();

    PointCloudPtr point_msg(new PointCloud);
    pcl::fromROSMsg(*input, *point_msg);
    point_msg = FilterCloudFrame(point_msg, config.voxel_filter_size);
    FilterByDistance(point_msg);
    if(point_msg->size() == 0) return;
    
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.rotate (Eigen::AngleAxisd (config.lidar_yaw_calib_degree*M_PI/180, Eigen::Vector3d::UnitZ()));
    pcl::transformPointCloud(*point_msg, *point_msg, transform);

    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
    double fitness_score = std::numeric_limits<double>::max();

    static Eigen::Matrix4d lio_pose = Eigen::Matrix4d::Identity();
    
    if(!init_pose_flag)
    { 
        if(!AlignPointCloud(point_msg, tf, fitness_score)){
            LOG(INFO)<<"Align PointCloud init success.";
        }
        POSEPtr init_odom;
        if(!FindCorrespondGpsMsg(timestamp, init_odom)){
            LOG(INFO)<<"PointCloud Callback init failed!!!";
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
        LOG(INFO)<<"failed to search ins_pose at:"<<std::fixed<<std::setprecision(6)<<timestamp;
        return;
    }
    CHECK_NOTNULL(ins_pose);

    Eigen::Matrix4d trans = lio_pose.inverse() * POSE2Affine(ins_pose).matrix();
    auto dist = trans.block<3,1>(0,3).norm();
    if(dist < config.frame_dist) {
        LOG(INFO)<<"the distance to last frame is :"<<dist<<" too small!!!";
        return;
    }
    LOG(INFO)<<"--------------------------new lidar-msg----------------------------------";
    common::Timer t;
    if(!AlignPointCloud(point_msg, tf, fitness_score)){
        LOG(WARNING)<<"regis between:("<<frame_cnt-1<<", "<<frame_cnt<<") is failed!!!";
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
    // point_msg = SegmentedCloud(point_msg); //test the lidar-seg
    AddKeyFrame(timestamp, tf, POSE2Affine(ins_pose).matrix(), point_msg);    
    
    //insert gps data to optimizer
    if(frame_cnt++ % config.ceres_config.num_every_scans == 0) {
        LOG(INFO)<<"Anchor is to correct the pose....";
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
    // LOG(INFO)<<"Num of Lidar-odom  pose:"<<odom_path.poses.size();

    //publish optimized path
    static nav_msgs::Path *optimized_path = &ceres_optimizer.optimized_path;
    if(!optimized_path->poses.empty()){
        optimized_pos_pub.publish(*optimized_path);
        // LOG(INFO)<<"Num of Optimized's pose:"<<optimized_path->poses.size();
    }
    LOG(INFO)<<"***********process the msg elspaed's time:"<<t.end()<<" [ms]";
}
// void MappingTimerCallback(){
void MappingTimerCallback(const ros::TimerEvent&){
// void MappingTimerCallback(const ros::WallTimerEvent& event){
    
    // while(1){
      common::Timer t;
      LOG(INFO)<<"================mapping timer callback====================";
      if(!pub_map.getNumSubscribers()) return;
      keyframe_mutex.lock();
      auto map_cloud = GenerateMap();
      keyframe_mutex.unlock();
      if(!map_cloud) return;
  
      map_cloud = FilterCloudFrame(map_cloud, config.voxel_filter_size);
  
      sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2());
      pcl::toROSMsg(*map_cloud, *cloud_msg);
  
      pub_map.publish(cloud_msg);
      LOG(WARNING)<<"Generating map cloud is done, and publish success.elspaed's time:"<<t.end()<<" [ms]";
    //   std::chrono::milliseconds dura(10000);
    //   std::this_thread::sleep_for(dura);
    // }
}

static void output_callback(const autoware_config_msgs::ConfigNDTMappingOutput::ConstPtr& input){
    common::Timer t;
    LOG(INFO)<<"================begin to save map====================";
    keyframe_mutex.lock();
    auto map_cloud = GenerateMap();
    keyframe_mutex.unlock();
    if(!map_cloud) return;
    
    LOG(WARNING)<<"Generating map cloud is done, and publish success.elspaed's time:"<<t.end()<<" [ms]";
    
    //save map
    double filter_res = input->filter_res;
    std::string filename = input->filename;
    LOG(INFO)<< "output_map params filter_res: " << filter_res;
    LOG(INFO)<< "output_map params filename: " << filename;

    map_cloud->header.frame_id="map";
    pcl::io::savePCDFileASCII(filename, *map_cloud);
    LOG(INFO)<<"Saved " << map_cloud->points.size() << " points to " << filename << " success!!!";


    pcl::PointCloud<PointT>::Ptr map_filtered(new pcl::PointCloud<PointT>());
    map_filtered->header.frame_id = "map";
    // Apply voxelgrid filter
    map_filtered = FilterCloudFrame(map_cloud, filter_res);
    std::string filename_filtered = filename + "_filtered";
    // pcl::io::savePCDFileASCII(filename_filtered, *map_filtered);
    LOG(INFO)<<"Saved " << map_filtered->points.size() << " filtered points to " << filename_filtered << " success!!!";
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
       LOG(INFO)<<"***********config params***********";
       LOG(INFO)<<"gps_lidar_time_threshold:"<<config.time_synchronize_threshold;
       LOG(INFO)<<"voxel_filter_size  :"<<     config.voxel_filter_size;
       LOG(INFO)<<"ndt_trans_epsilon  :"<< config.ndt_trans_epsilon;
       LOG(INFO)<<"ndt_step_size      :"<< config.ndt_step_size;
       LOG(INFO)<<"ndt_resolution     :"<< config.ndt_resolution;
       LOG(INFO)<<"ndt_maxiterations  :"<< config.ndt_maxiterations;
       LOG(INFO)<<"optimize_iter_num  :"<< config.ceres_config.iters_num;
       LOG(INFO)<<"optimize_var_anchor:"<< config.ceres_config.var_anchor;
       LOG(INFO)<<"optimize_var_odom_t:"<< config.ceres_config.var_odom_t;
       LOG(INFO)<<"optimize_var_odom_q:"<< config.ceres_config.var_odom_q;
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
   ros::Subscriber gnss_sub   = nh.subscribe("/gnss_pose",  30000,  InsCallback);

   //mapping
//    ros::WallTimer map_publish_timer = nh.createWallTimer(ros::WallDuration(config.map_cloud_update_interval), MappingTimerCallback);
   ros::Timer map_publish_timer = nh.createTimer(ros::Duration(5.0), MappingTimerCallback); //todo check
//    auto mapping_thread = std::thread(MappingTimerCallback);
   pub_map = nh.advertise<sensor_msgs::PointCloud2>("/mapping/localizer_map", 1);
   ros::Subscriber output_sub = nh.subscribe("config/ndt_mapping_output", 10, output_callback);
   
   ros::spin();
   return 0;
}