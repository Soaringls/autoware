#include "ceres_optimizer.h"

CeresOptimizer::CeresOptimizer() {
  thread_optimize_ = std::thread(&CeresOptimizer::Optimize, this);
  W_BODY = Eigen::Matrix4d::Identity();
}
CeresOptimizer::~CeresOptimizer() { thread_optimize_.detach(); }

ceres::Solver::Options CeresOptimizer::GetCeresSolverOptions() {
  ceres::Solver::Options options;
  // options.linear_solver_type = ceres::SPARSE_QR;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = config_.iters_num;  // 15;
  options.update_state_every_iteration = true;
  options.minimizer_progress_to_stdout = true;
  return options;
}

void CeresOptimizer::InsertOdom(const POSEPtr odom) {
  if (!odom_init_flag) {
    init_pos_ = POSE2Affine(odom);
    W_BODY.block<3, 3>(0, 0) = (odom->q).toRotationMatrix();
    W_BODY.block<3, 1>(0, 3) = odom->pos;
    odom_init_flag = true;
    LOG(INFO) << "init W_BODY's pose:" << std::fixed << std::setprecision(6)
              << W_BODY(0, 3) << "," << W_BODY(1, 3) << "," << W_BODY(2, 3);
    return;
  }
  odom_mutex_.lock();
  odom_data_.push_back(odom);

  POSEPtr result_pose(new POSE);
  result_pose->time = odom->time;

  result_pose->q = W_BODY.block<3, 3>(0, 0) * odom->q;
  result_pose->pos =
      W_BODY.block<3, 1>(0, 3) + W_BODY.block<3, 3>(0, 0) * odom->pos;
  W_BODY.block<3, 3>(0, 0) = (result_pose->q).toRotationMatrix();
  W_BODY.block<3, 1>(0, 3) = result_pose->pos;

  optimized_data_.push_back(result_pose);
  // UpdateOptimizedPath();
  last_optimized_pose_ = result_pose;
  odom_mutex_.unlock();
}
void CeresOptimizer::InsertGPS(const POSEPtr gps) {
  gps_data_.push_back(gps);
  flag_anchor_ = true;
}

void CeresOptimizer::Optimize() {
  // todo
  while (1) {
    if (flag_anchor_) {
      flag_anchor_ = false;
      LOG(INFO) << "global optimization .....";
      ceres::Problem problem;
      ceres::Solver::Summary summary;
      ceres::LossFunction* loss_function_anchor;
      loss_function_anchor = new ceres::HuberLoss(1.0);

      ceres::LocalParameterization* local_parameterization =
          new ceres::QuaternionParameterization();

      // add params
      odom_mutex_.lock();
      // Timer t;
      int length = optimized_data_.size();
      double to_optimized_t[length][3];
      double to_optimized_q[length][4];
      LOG(INFO) << "CeresOptimizer::Optimize() data's length:" << length;
      for (int i = 0; i < length; ++i) {
        to_optimized_t[i][0] = optimized_data_[i]->pos[0];
        to_optimized_t[i][1] = optimized_data_[i]->pos[1];
        to_optimized_t[i][2] = optimized_data_[i]->pos[2];

        to_optimized_q[i][0] = optimized_data_[i]->q.w();
        to_optimized_q[i][1] = optimized_data_[i]->q.x();
        to_optimized_q[i][2] = optimized_data_[i]->q.y();
        to_optimized_q[i][3] = optimized_data_[i]->q.z();

        problem.AddParameterBlock(to_optimized_q[i], 4, local_parameterization);
        problem.AddParameterBlock(to_optimized_t[i], 3);
      }

      // ceres::AddResidualBlock-odom and gps's residuals
      for (int i = 0; i < length - 1; ++i) {
        if (i + 1 < length) {
          double info_var_t = config_.var_odom_t;
          ceres::CostFunction* cost_odom_func =
              mapping::OdomCostFunction::create(
                  odom_data_[i]->pos.x(), odom_data_[i]->pos.y(),
                  odom_data_[i]->pos.z(), odom_data_[i]->q.w(),
                  odom_data_[i]->q.x(), odom_data_[i]->q.y(),
                  odom_data_[i]->q.z(), info_var_t,
                  config_.var_odom_q);  // 0.1, 0.01);
          problem.AddResidualBlock(cost_odom_func, NULL, to_optimized_q[i],
                                   to_optimized_t[i], to_optimized_q[i + 1],
                                   to_optimized_t[i + 1]);
        }

        for (int j = 0; j < gps_data_.size(); ++j) {
          if (gps_data_[j]->time == odom_data_[i]->time) {
            ceres::CostFunction* cost_func_gps =
                mapping::GPSCostFunction::create(
                    gps_data_[j]->pos.x(), gps_data_[j]->pos.y(),
                    gps_data_[j]->pos.z(), config_.var_anchor);  // 0.05);
            problem.AddResidualBlock(cost_func_gps, loss_function_anchor,
                                     to_optimized_t[i]);
            break;
          }
        }
      }
      // ceres::Solve problem
      ceres::Solve(GetCeresSolverOptions(), &problem, &summary);
      LOG(INFO) << "ceres::summary:" << summary.BriefReport();

      for (int i = 0; i < length; ++i) {
        optimized_data_[i]->pos = Eigen::Vector3d(
            to_optimized_t[i][0], to_optimized_t[i][1], to_optimized_t[i][2]);
        optimized_data_[i]->q =
            Eigen::Quaterniond(to_optimized_q[i][0], to_optimized_q[i][1],
                               to_optimized_q[i][2], to_optimized_q[i][3]);
        if (i == length - 1) {
          W_BODY.block<3, 3>(0, 0) =
              Eigen::Quaterniond(to_optimized_q[i][0], to_optimized_q[i][1],
                                 to_optimized_q[i][2], to_optimized_q[i][3])
                  .toRotationMatrix();
          W_BODY.block<3, 1>(0, 3) = Eigen::Vector3d(
              to_optimized_t[i][0], to_optimized_t[i][1], to_optimized_t[i][2]);
          LOG(INFO) << "CeresOptimizer::Optimize()-W_BODY:" << W_BODY(0, 3)
                    << ", " << W_BODY(1, 3) << ", " << W_BODY(2, 3);
        }
      }
      UpdateOptimizedPath();
      // LOG(INFO) << "ceres-optimizer, elspaed's time:" << t.end() << " [ms]";
      odom_mutex_.unlock();
    }
    std::chrono::milliseconds dura(1000);
    std::this_thread::sleep_for(dura);
  }
  return;
}

void CeresOptimizer::UpdateOptimizedPath() {
  optimized_path.poses.clear();
  for (int i = 0; i < optimized_data_.size(); ++i) {
    AddPoseToNavPath(optimized_path, optimized_data_[i]);
  }
}

void CeresOptimizer::AddPoseToNavPath(nav_msgs::Path& path, POSEPtr pose,
                                      std::string frame_id) {
  if (!odom_init_flag) return;
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = ros::Time(pose->time);
  pose_stamped.header.frame_id = frame_id;

  auto locla_pose = POSE2Affine(pose).matrix();
  locla_pose(0, 3) -= init_pos_.matrix()(0, 3);
  locla_pose(1, 3) -= init_pos_.matrix()(1, 3);
  locla_pose(2, 3) -= init_pos_.matrix()(2, 3);
  // auto pose_rviz = init_pos_.inverse() * POSE2Affine(pose);
  auto pose_rviz = Eigen::Affine3d(locla_pose);
  auto pose_rviz_p = Eigen::Translation3d(pose_rviz.translation());
  auto pose_rviz_q = Eigen::Quaterniond(pose_rviz.linear());
  pose_stamped.pose.position.x = pose_rviz_p.x();
  pose_stamped.pose.position.y = pose_rviz_p.y();
  pose_stamped.pose.position.z = pose_rviz_p.z();
  pose_stamped.pose.orientation.w = pose_rviz_q.w();
  pose_stamped.pose.orientation.x = pose_rviz_q.x();
  pose_stamped.pose.orientation.y = pose_rviz_q.y();
  pose_stamped.pose.orientation.z = pose_rviz_q.z();
  path.header = pose_stamped.header;
  path.poses.push_back(pose_stamped);
}

Eigen::Affine3d POSE2Affine(const POSEPtr& pose) {
  Eigen::Affine3d result =
      Eigen::Translation3d(pose->pos[0], pose->pos[1], pose->pos[2]) * pose->q;
  // result.translation() = pose->pos;
  // result.linear() = pose->q;
  return result;
}