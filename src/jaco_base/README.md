# jaco_base

argallab repository for controlling Kinova Robotics Jaco arm. 

## Important
The `master` branch is now supporting **_ROS Melodic_** and **_ROS Kinetic_**.  Refer to the `indigo-devel` branch for Ubuntu 14.04 and Indigo support.  [kinova-ros](https://github.com/Kinovarobotics/kinova-ros) no longer supports ROS Indigo.

This goes without saying, but _any_ development should be done on branches and _not_ on the master branch.  Refer to [git](https://git-scm.com/docs/git-branch) documentation for instructions on how to create and develop on branches. 

## Table of Contents
1. [Installation Instructions](#installation)
2. [Modules and Scripts](#modules)
3. [Usage Guide](#usage)
4. [Useful commands](#useful-commands)

## Installation Instructions <a name="installation"></a>

### General
If using within a container made _with_ the `argallab/jaco_ros` image: 
```
mkdir -p YOUR_WORKSPACE/src
cd YOUR_WORKSPACE
git init
git submodule add https://github.com/argallab/jaco_base.git src/jaco_base
git submodule update --init --recursive
```
The above command should create a `.gitmodules` file inside at the root of your workspace. 

`kinova-ros` is a submodule within `jaco_base`. You will need to clone using the `--recursive` tag and initialize and update the submodule using `submodule update --init --recursive`. 

Additional steps if not using within a container made with the `argallab/jaco_ros` image: 
```
./install_dependencies.sh 
cd ../..
catkin_make
```
Catkin will let you know if you need to install any other dependencies. If so, please update the `install_dependencies.sh` file.

### Using jaco_base with Docker

To use jaco_base within a docker container:
First clone the repository
```
mkdir -p jaco_ws/src
git clone https://github.com/argallab/jaco_base.git --recursive 
```
Then run the following command to create the docker container. 
The argallab/biolodic image used in the command below uses Ubuntu 18.04 and ROS Melodic.
```
sudo docker run -it --privileged \
-v /dev/bus/usb:/dev/bus/usb \
-v /dev/input/by-id:/dev/input/by-id \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
-v /home/ACCOUNT_NAME/jaco_ws/src:/home/jaco_ws/src \
-e DISPLAY \
-e QT_X11_NO_MITSHM=1 \
--name UNIQUE_CONTAINER_NAME \
--net=host \
argallab/biolodic:latest
```
Then build the workspace
```
cd home/jaco_ws
catkin_make
```
Add the following line to the end of your .bashrc file and save it
```
source /home/jaco_ws/devel/setup.bash
```
Then source the workspace
```
source ~/.bashrc
```

### Using `jaco_base` with other packages
The recommended way to include this package in a catkin workspace, along with other packages, is to use [wstool](http://wiki.ros.org/wstool).  This tool depends on both [vcstools](http://wiki.ros.org/vcstools) and [rosinstall](http://wiki.ros.org/rosinstall).  We use this tool for downloading various existing packages (from multiple source repos), version control (eg. interfaces with `git`), and stitching together multiple packages in a single larger workspace.

There are two main workflows we care about: A) setting up our own workspace ourselves or B) setting up a workspace using preexisting `rosinstall` file.

**(A)** To set up our own workspace with `jaco_base` using `wstool`, we can do the following:
1. Create a catkin workspace
```
mkdir -p catkin_ws/src
```
2. Initialize the workspace (creates a `.rosinstall` file)
```
cd catkin_ws/src
wstool init
```
3. Set our git package (updates the `.rosinstall` file with `jaco_base` repository information) for cloning
```
wstool set --git jaco_base https://github.com/argallab/jaco_base.git
```
4. Complete the cloning process
```
wstool update jaco_base
```
5. Build the workspace
```
cd ..
catkin_make
```
Note: this automatically takes care of submodules and `--recursive` behaviors that we generally desire.  

To add other packages to the this workspace, we can simply follow steps 3-5.  What is powerful about this tool is that we can now use a single `rosinstall` file (a mere text file) to recreate the same exact workspace on multiple machines as well as share with collaborators.  In addition, we can maintain _packages_ in our git repos rather than workspaces.

**(B)** In order to set up a workspace using a preexisting `rosinstall` file, we do the following:
1. Create a new workspace
```
mkdir -p catkin_ws/src
```
2. Download the `rosinstall` file and place it in `catkin_ws/src` (let us call this file `example.rosinstall`)
3. Initialize the workspace 
```
cd catkin_ws/src
wstool init
```
4. Merge and update the workspace using a preexisting `rosinstall` file
```
wstool merge example.rosinstall
wstool update
``` 
5. Build the workspace
```
cd ..
catkin_make
```

## Modules and Scripts <a name="modules"></a>
The repository includes the following packages: 
1. `jaco_interaction`: Contains files for interacting with the jaco using the different modules and packages. 
2. `jaco_teleop`: Contains the files necessary for teleoperating the arm and finger joints using different interfaces. 
3. `jaco_pfields`: Contains files for potential field autonomy. 
4. `jaco_blend`: Contains control-blending related files. 
5. `joy_mouse`: Contains files for converting inputs from mouse type interfaces to `sensor_msgs/JOY` topics. 

Submodule: 
* `kinova-ros`: Forked the master branch from `Kinovarobotics/kinova-ros` and included as submodule. Some of the API was changed to suite our lab's requirements.

## Usage Guide <a name="usage"></a>
For basic operation, run the following script with the desired arguments:
```
roslaunch jaco_interaction jaco_base.launch 
```
Possible arguments: 
```
JOY:=true For 3-axis joystick teleoperation 
SNP:=true For sip/puff teleoperation
HA:=true  For headarray teleoperation 
blending:=false  For pure teleoperation with no control blending
```

For MSI Demo: 

Teleoperation with Sip/Puff
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:="j2s7s300"
roslaunch jaco_teleop jaco_teleop.launch SNP:=true
```

Teleoperation with Joystick
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:="j2s7s300"
roslaunch jaco_teleop jaco_teleop.launch JOY3:=true
```

## Useful commands
* Bringup robot
```
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=j2s7s300
```
* Initialize Movieit and Rviz!
```
roslaunch j2s7s300_moveit_config j2s7s300_demo.launch
```
* To home the arm to the default Kinova-defined position:
```
rosservice call /j2s7s300_driver/in/home_arm
```
* To turn on/off force control: 
```
rosservice call /j2s7s300_driver/in/start_force_control
rosservice call /j2s7s300_driver/in/stop_force_control
```
* For velocity control mode:
```
rosservice call /j2s7s300_driver/in/set_control_mode "current_control_mode: 'velocity'"
```
* For trajectory control mode:
```
rosservice call /j2s7s300_driver/in/set_control_mode "current_control_mode: 'trajectory'"
```
* To enable/disable the ROS motion command:
```
rosservice call /'${kinova_robotType}_driver'/in/start
rosservice call /'${kinova_robotType}_driver'/in/stop
```
* To recalibrate torque sensors: move the robot to candle-like pose (all joints 180 deg, robot links points straight up). This configuration ensures zero torques at joints.  This is also the "Vertical" position in the rviz planner. Then,
```
rosservice call /'${kinova_robotType}_driver'/in/set_zero_torques
```

For additional information, refer to the [kinova-ros](https://github.com/Kinovarobotics/kinova-ros#how-to-use-the-stack) repo.
