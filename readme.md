```sh
mkdir catkin_autoware
git clone usr ./src  #src根目录不可有cmake

```

rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO

# compile
## compile with cuda support
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

## compile without cuda support
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug Release

## compile particular module
colcon build --packages-select ekf_localizer

rviz -d ./data/ces_mapping.rviz 