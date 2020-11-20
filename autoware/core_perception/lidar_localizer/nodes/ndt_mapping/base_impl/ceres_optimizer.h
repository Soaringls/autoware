#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "lidar_alignment/timer.h"

using namespace lidar_alignment;
struct POSE {
  double time = 0;
  Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);         // x y z
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();  // rotation
  double score = 0.0;

  void reset() {
    time = 0.0;
    score = 0.0;
    pos = Eigen::Vector3d(0, 0, 0);
    q = Eigen::Quaterniond(Eigen::Matrix3d::Identity());
  }

  // todo check
  POSE& operator=(const POSE& rhs) {
    if (&rhs == this) return *this;
    this->time = rhs.time;
    this->pos = rhs.pos;
    this->q = rhs.q;
    this->score = rhs.score;
    return *this;
  }
  // friend std::ostream& operator<<(std::ostream& os, POSE pose);
};
using POSEPtr = std::shared_ptr<POSE>;
using POSEConstPtr = std::shared_ptr<const POSE>;

class TicToc {
 public:
  void Tic() { start_ = std::chrono::system_clock::now(); }
  double Toc() {
    end_ = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end_ - start_;
    return duration.count() * 1e3;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
};

template <typename T>
inline void QuaternionInverse(const T q[4], T q_inverse[4]) {
  q_inverse[0] = q[0];   // qw
  q_inverse[1] = -q[1];  // qx
  q_inverse[2] = -q[2];  // qy
  q_inverse[3] = -q[3];  // qz
}
namespace mapping {
struct GPSCostFunction {
  GPSCostFunction(double x, double y, double z, double var)
      : x_(x), y_(y), z_(z), var_(var) {}

  template <typename T>
  bool operator()(const T* tj, T* residuals) const {
    residuals[0] = (tj[0] - T(x_)) / T(var_);  // error-x
    residuals[1] = (tj[1] - T(y_)) / T(var_);  // error-y
    residuals[2] = (tj[2] - T(z_)) / T(var_);  // error-z
    // LOG(INFO)<<"gps-cost func     -tj:["<<tj[0]<<", "<<tj[1]<<",
    // "<<tj[2]<<"]"; LOG(INFO)<<"gps-cost func-default:["<<x_<<", "<<y_<<",
    // "<<z_<<"]"; LOG(INFO)<<"gps-cost    residuals:["<<residuals[0]<<",
    // "<<residuals[1]<<", "<<residuals[2]<<"]";
    return true;
  }

  static ceres::CostFunction* create(const double x, double y, double z,
                                     double var) {
    return (new ceres::AutoDiffCostFunction<GPSCostFunction, 3, 3>(
        new GPSCostFunction(x, y, z, var)));
  }

  double x_, y_, z_, var_;
};
// todo
struct OdomCostFunction {
  OdomCostFunction(const double tf_x, const double tf_y, const double tf_z,
                   const double tf_qw, const double tf_qx, const double tf_qy,
                   const double tf_qz, const double var_t, const double var_q)
      : tf_x_(tf_x),
        tf_y_(tf_y),
        tf_z_(tf_z),
        tf_qw_(tf_qw),
        tf_qx_(tf_qx),
        tf_qy_(tf_qy),
        tf_qz_(tf_qz),
        var_t_(var_t),
        var_q_(var_q) {}

  template <typename T>
  bool operator()(const T* w_q_i, const T* ti,  // t1  i  pose of gps
                  const T* w_q_j, const T* tj,  // t2  j  pose of gps
                  T* residuals) const {
    // step1 error of gps's offset and odom's offset
    T t_w_ij[3];  // translation of j/i, j is newer
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];
    // LOG(INFO)<<"odom-cost func-ti:["<<ti[0]<<", "<<ti[1]<<", "<<ti[2]<<"]";
    // LOG(INFO)<<"odom-cost func-tj:["<<tj[0]<<", "<<tj[1]<<", "<<tj[2]<<"]";
    // LOG(INFO)<<"odom-cost func-default:["<<tf_x_<<", "<<tf_y_<<",
    // "<<tf_z_<<"]";
    T w_q_i_inverse[4];
    QuaternionInverse(w_q_i, w_q_i_inverse);
    T t_i_ij[3];  // offset 姿态由相对world系转到相对i系
    ceres::QuaternionRotatePoint(w_q_i_inverse, t_w_ij, t_i_ij);
    residuals[0] = (t_i_ij[0] - T(tf_x_)) / T(var_t_);
    residuals[1] = (t_i_ij[1] - T(tf_y_)) / T(var_t_);
    residuals[2] = (t_i_ij[2] - T(tf_z_)) / T(var_t_);
    // LOG(INFO)<<"odom-cost   residuals:["<<residuals[0]<<",
    // "<<residuals[1]<<", "<<residuals[2]<<"]";

    // step2 error of gps's q(rotation) and odom's q(rotation)
    T odom_q[4];  // Odom's q of j/i
    odom_q[0] = T(tf_qw_);
    odom_q[1] = T(tf_qx_);
    odom_q[2] = T(tf_qy_);
    odom_q[3] = T(tf_qz_);

    T odom_q_inverse[4];
    QuaternionInverse(odom_q, odom_q_inverse);
    T q_i_j[4];  // GPS's q of j/i
    ceres::QuaternionProduct(w_q_i_inverse, w_q_j, q_i_j);
    T error_q[4];
    ceres::QuaternionProduct(odom_q_inverse, q_i_j, error_q);
    residuals[3] = T(2) * error_q[1] / T(var_q_);
    residuals[4] = T(2) * error_q[2] / T(var_q_);
    residuals[5] = T(2) * error_q[3] / T(var_q_);
    // residuals[6] = T(2) * error_q[4] / T(var_q_);

    return true;
  }
  static ceres::CostFunction* create(const double tf_x, const double tf_y,
                                     const double tf_z, const double tf_qw,
                                     const double tf_qx, const double tf_qy,
                                     const double tf_qz, const double var_t,
                                     const double var_q) {
    return (new ceres::AutoDiffCostFunction<OdomCostFunction, 6, 4, 3, 4, 3>(
        new OdomCostFunction(tf_x, tf_y, tf_z, tf_qw, tf_qx, tf_qy, tf_qz,
                             var_t, var_q)));
  }

  double tf_x_, tf_y_, tf_z_;
  double tf_qw_, tf_qx_, tf_qy_, tf_qz_;
  double var_t_, var_q_;
};
};  // namespace mapping

struct CeresConfig {
  int num_every_scans = 1;
  int iters_num = 10;
  double var_anchor = 0.05;
  double var_odom_t = 0.1;
  double var_odom_q = 0.01;
};

class CeresOptimizer {
 public:
  CeresOptimizer();
  ~CeresOptimizer();
  nav_msgs::Path optimized_path;
  void InsertOdom(const POSEPtr odom);
  void InsertGPS(const POSEPtr gps);

  void SetConfig(const CeresConfig& config) { config_ = config; }
  ceres::Solver::Options GetCeresSolverOptions();
  void Optimize();

  std::vector<POSEPtr> GetOptimizedResult() {
    std::vector<POSEPtr> result;
    odom_mutex_.lock();
    for (const auto elem : optimized_data_) {
      POSEPtr temp(new POSE);
      *temp = *elem;
      result.push_back(temp);
    }
    odom_mutex_.unlock();
    return result;
  }

  void UpdateOptimizedPath();
  void AddPoseToNavPath(nav_msgs::Path& path, POSEPtr pose,
                        std::string frame_id = "world");

 private:
  std::mutex odom_mutex_;
  CeresConfig config_;

  POSEPtr last_optimized_pose_;
  bool flag_anchor_ = false;
  bool odom_init_flag = false;
  std::vector<POSEPtr> odom_data_;
  std::vector<POSEPtr> gps_data_;
  std::vector<POSEPtr> optimized_data_;

  Eigen::Matrix4d W_BODY;
  Eigen::Affine3d init_pos_;
  std::thread thread_optimize_;
};

Eigen::Affine3d POSE2Affine(const POSEPtr& pose);