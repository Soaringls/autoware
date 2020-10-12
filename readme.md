```sh
mkdir catkin_autoware
git clone usr ./src

```

rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO

# compile with cuda support
AUTOWARE_COMPILE_WITH_CUDA=1 colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# compile without cuda support
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release