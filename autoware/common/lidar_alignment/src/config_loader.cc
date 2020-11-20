#include "config_loader.h"
namespace lidar_alignment{

void LoadLidarConfig(const std::string& config_yaml_file){
    if(!boost::filesystem::exists(config_yaml_file)){
        LOG(FATAL)<<"the file:"<<config_yaml_file<<" is not exist!";
    }
    LidarFrameConfig::Config config;
    YAML::Node yaml_config = YAML::LoadFile(config_yaml_file);

    config.num_vertical_scans =
        yaml_config["lidar"]["num_vertical_scans"].as<std::uint16_t>();
    config.num_horizontal_scans =
        yaml_config["lidar"]["num_horizontal_scans"].as<std::uint16_t>();
    config.ground_scan_index =
        yaml_config["lidar"]["ground_scan_index"].as<std::uint16_t>();
    config.vertical_angle_bottom =
        yaml_config["lidar"]["vertical_angle_bottom"].as<float>();
    config.vertical_angle_top = yaml_config["lidar"]["vertical_angle_top"].as<float>();
    config.sensor_mount_angle = yaml_config["lidar"]["sensor_mount_angle"].as<float>();
    config.scan_period = yaml_config["lidar"]["scan_period"].as<float>();
  
    // Segmentation
    config.segment_valid_point_num =
        yaml_config["segmentation"]["segment_valid_point_num"].as<std::uint16_t>();
    config.segment_valid_line_num =
        yaml_config["segmentation"]["segment_valid_line_num"].as<std::uint16_t>();
    config.segment_cluster_num =
        yaml_config["segmentation"]["segment_cluster_num"].as<std::uint16_t>();
    
    config.segment_theta = yaml_config["segmentation"]["segment_theta"].as<float>();
    config.surf_segmentation_angle_threshold =
        yaml_config["segmentation"]["surf_segmentation_angle_threshold"].as<float>();
  
    // Feature extraction
    config.edge_threshold =
        yaml_config["featureExtraction"]["edge_threshold"].as<float>();
    config.surf_threshold =
        yaml_config["featureExtraction"]["surf_threshold"].as<float>();
    config.nearest_feature_search_dist =
        yaml_config["featureExtraction"]["nearest_feature_search_distance"]
            .as<float>();
    LidarFrameConfig::GetInstance(config);
}

void LoadExtrinsic(const std::string& config_file) {
    if(!boost::filesystem::exists(config_file)){
        LOG(FATAL)<<"the file:"<<config_file<<" is not exist!";
    }

    YAML::Node config = YAML::LoadFile(config_file);

    Eigen::Quaterniond q;
    q.x() = config["transform"]["rotation"]["x"].as<double>();
    q.y() = config["transform"]["rotation"]["y"].as<double>();
    q.z() = config["transform"]["rotation"]["z"].as<double>();
    q.w() = config["transform"]["rotation"]["w"].as<double>();

    Eigen::Vector3d t;
    t.x() = config["transform"]["translation"]["x"].as<double>();
    t.y() = config["transform"]["translation"]["y"].as<double>();
    t.z() = config["transform"]["translation"]["z"].as<double>();
    
    Eigen::Matrix4d T;
    Eigen::Matrix3d R;

    R = q.toRotationMatrix();
    T << R, t, Eigen::RowVector4d(0, 0, 0, 1);

    std::string child_frame_id = config["child_frame_id"].as<std::string>();
    std::string frame_id = config["header"]["frame_id"].as<std::string>();
    std::uint32_t timestamp = config["header"]["stamp"]["secs"].as<std::uint32_t>();
}

}