cmake_minimum_required(VERSION 2.8.3)
project(lidar_alignment)

find_package(autoware_build_flags REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    roslint
)

catkin_package(
  INCLUDE_DIRS src
  LIBRARIES lidar_alignment
)

find_package(PCL REQUIRED)
find_package(Eigen3 QUIET)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


find_package(Ceres REQUIRED)
find_package(libpointmatcher REQUIRED)
find_package(Boost REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Glog)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${YAMLCPP_INCLUDE_DIR}
  ${libpointmatcher_INCLUDE_DIRS}
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIR}
)

add_library(${PROJECT_NAME} SHARED
   src/align_pointmatcher.cc
   src/config_loader.cc
   src/lidar_segmentation.cc
  #  src/ceres_optimizer.cc
)

target_link_libraries(${PROJECT_NAME}
    ${libpointmatcher_LIBRARIES}
    yaml-cpp glog
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
    "/usr/local/lib/libpointmatcher.so.1.3.1"
    "/usr/local/lib/libpointmatcher.so.1"
    "/usr/local/lib/libpointmatcher.so"
)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
add_executable(align_test main.cc)
target_link_libraries(align_test ${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY src/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"  
)