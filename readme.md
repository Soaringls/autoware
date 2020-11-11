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


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/autoware/workspace/install/lidar_alignment/lib/

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-select lidar_localizer
/autoware/workspace/data/map/2230-test.pcd