cmake_minimum_required(VERSION 2.8.3)
project(jaco_interaction)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  dynamic_reconfigure
  gazebo_msgs
  geometry_msgs
  kinova_msgs
  message_generation
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  pluginlib
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf_conversions
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  DampState.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  CustomHome.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  # sensor_msgs
  std_msgs
  std_srvs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  nodes/custom_home_arm
  nodes/dampening_zone
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
