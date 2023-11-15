# Polybot - linorobot2

![linorobot2](docs/linorobot2.gif)

linorobot2 is a ROS2 port of the [linorobot](https://github.com/linorobot/linorobot) package. If you're planning to build your own custom ROS2 robot (2WD, 4WD, Mecanum Drive) using accessible parts, then this package is for you. This repository contains launch files to easily integrate your DIY robot with Nav2 and a simulation pipeline to run and verify your experiments on a virtual robot in Gazebo.

Once the robot's URDF has been configured in linorobot2_description package, users can easily switch between booting up the physical robot and spawning the virtual robot in Gazebo.

![linorobot2_architecture](docs/linorobot2_launchfiles.png)

Assuming you're using one of the tested sensors, linorobot2 automatically launches the necessary hardware drivers, with the topics being conveniently matched with the topics available in Gazebo. This allows users to define parameters for high level applications (ie. Nav2 SlamToolbox, AMCL) that are common to both virtual and physical robots.

The image below summarizes the topics available after running **bringup.launch.py**.
![linorobot2_microcontroller](docs/microcontroller_architecture.png)

An in-depth tutorial on how to build the robot is available in [linorobot2_hardware](https://github.com/linorobot/linorobot2_hardware).

## Installation

This package requires ros-humble. If you haven't installed ROS2 yet, you can see the guide below. This package and ROS2 should be installed on the robots Raspberry Pi running Ubuntu Server 22.04.3.

### 0. ROS2 Humble

#### 0.1 Install ROS2 Humble

First ensure that the Ubuntu Universe repository is enabled.

    sudo apt install -y software-properties-common
    sudo add-apt-repository -y universe

Now add the ROS 2 GPG key with apt.

    sudo apt update -y
    sudo apt install -y curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

Then add the repository to your sources list.

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

Update repositories

    sudo apt update -y && sudo apt upgrade -y

Install ROS-Base: Communication libraries, message packages, command line tools, No GUI tools. Also install development tools: Compilers and other tools to build ROS packages.

    sudo apt install -y ros-humble-ros-base
    sudo apt install -y ros-dev-tools

Source the setup script

    source /opt/ros/humble/setup.bash

#### 0.2 Configure environment

Now we configure the ROS2 environment. Add this to your _bashrc_ file

    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh" >> ~/.bashrc

Also set a ROS_DOMAIN. Here it is set to 60.

    echo "export ROS_DOMAIN_ID=60 >> ~/.bashrc
    source ~/.bashrc

### 1. Robot Computer - linorobot2 Package

Make a workspace:

    cd $HOME
    mkdir linorobot2_ws

#### 1.1 Laser_sensor - rplidar

    sudo apt install -y ros-$ROS_DISTRO-rplidar-ros
    cd /tmp
    wget https://raw.githubusercontent.com/allenh1/rplidar_ros/ros2/scripts/rplidar.rules
    sudo cp rplidar.rules /etc/udev/rules.d/

#### 1.2 Depth_sensor -realsense

    sudo apt install -y ros-$ROS_DISTRO-realsense2-camera

#### 1.3 micro-ROS

    cd $HOME/linorobot2_ws
    git clone -b $ROS_DISTRO https://github.com/micro-ROS/micro_ros_setup src/micro_ros_setup
    sudo apt install -y python3-vcstool build-essential
    sudo apt update -y && rosdep update
    rosdep install --from-path src --ignore-src -y
    colcon build
    source install/setup.bash

#### 1.4 Setup micro-ROS agent

    ros2 run micro_ros_setup create_agent_ws.sh
    ros2 run micro_ros_setup build_agent.sh
    source install/setup.bash

You can ignore 1 package had stderr output: microxrcedds_agent after building your workspace if it occours.

#### 1.5 Install linorobot2 package

    cd $HOME/linorobot2_ws
    git clone -b $ROS_DISTRO https://github.com/P9-Robuddy/linorobot2 src/linorobot2

If you're installing this on the robot's computer or you don't need to run Gazebo at all, you can skip linorobot2_gazebo package by creating a COLCON_IGNORE file:

    cd $HOME/linorobot2_ws/src/linorobot2/linorobot2_gazebo
    touch COLCON_IGNORE

Now install the linorobot2 package:

    cd $HOME/linorobot2_ws
    rosdep update && rosdep install --from-path src --ignore-src -y --skip-keys microxrcedds_agent
    colcon build
    source install/setup.bash

#### 1.7 Finishing up

Add the following parameters to your _bashrc_ file:

    echo "export LINOROBOT2_BASE=2wd" >> ~/.bashrc
    echo "export LINOROBOT2_DEPTH_SENSOR=realsense" >> ~/.bashrc
    echo "export LINOROBOT2_LASER_SENSOR=rplidar" >> ~/.bashrc
    source ~/.bashrc

### 2. Host Machine / Development Computer - Gazebo Simulation (Optional)

This step is only required if you plan to use Gazebo later. This comes in handy if you want to fine-tune parameters (ie. SLAM Toolbox, AMCL, Nav2) or test your applications on a virtual robot.

#### 2.1 Install linorobot2 Package

Install linorobot2 package on the host machine:

    cd <host_machine_ws>
    git clone -b $ROS_DISTRO https://github.com/P9-Robuddy/linorobot2 src/linorobot2
    rosdep update && rosdep install --from-path src --ignore-src -y --skip-keys microxrcedds_agent --skip-keys micro_ros_agent
    colcon build
    source install/setup.bash

\* microxrcedds_agent and micro_ros_agent dependency checks are skipped to prevent this [issue](https://github.com/micro-ROS/micro_ros_setup/issues/138) of finding its keys. This means that you have to always add `--skip-keys microxrcedds_agent --skip-keys micro_ros_agent` whenever you have to run `rosdep install` on the ROS2 workspace where you installed linorobot2.

#### 2.2 Define Robot Type

Set LINOROBOT2_BASE env variable to the type of robot base used. Available env variables are _2wd_, _4wd_, and _mecanum_. For example:

    echo "export LINOROBOT2_BASE=2wd" >> ~/.bashrc
    source ~/.bashrc

You can skip the next step (Host Machine - RVIZ Configurations) since this package already contains the same RVIZ configurations to visualize the robot.

### 3. Host Machine - RVIZ Configurations

Install [linorobot2_viz](https://github.com/linorobot/linorobot2_viz) package to visualize the robot remotely specifically when creating a map or initializing/sending goal poses to the robot. The package has been separated to minimize the installation required if you're not using the simulation tools on the host machine.

    cd <host_machine_ws>
    git clone https://github.com/linorobot/linorobot2_viz src/linorobot2_viz
    rosdep update && rosdep install --from-path src --ignore-src -y 
    colcon build
    source install/setup.bash

## Hardware and Robot Firmware

All the hardware documentation and robot microcontroller's firmware can be found [here](https://github.com/linorobot/linorobot2_hardware).

## URDF

### 1. Custom URDF - Define robot properties

[linorobot2_description](./linorobot2_description) package has parameterized xacro files that can help you kickstart writing the robot's URDF. Open <robot_type>.properties.urdf.xacro in [linorobot2_description/urdf](./linorobot2_description/urdf) directory and change the values according to the robot's specification/dimensions. All pose definitions must be measured from the `base_link` (center of base) and wheel positions (ie `wheel_pos_x`) are referring to wheel 1.

For custom URDFs, you can change the `urdf_path` in [description.launch.py](./linorobot2_description/launch/description.launch.py) found in linorobot2_description/launch directory.

Robot Orientation:

--------------FRONT--------------

WHEEL1  WHEEL2  (2WD/4WD)

WHEEL3  WHEEL4  (4WD)

--------------BACK--------------

Build the robot computer's workspace to load the new URDF:

    cd <robot_computer_ws>
    colcon build

The same changes must be made on the host machine's <robot_type>.properties.urdf.xacro if you're simulating the robot in Gazebo. Remember to also build the host machine's workspace after editing the xacro file.

    cd <host_machine_ws>
    colcon build

### 2. Visualize the newly created URDF

#### 2.1 Publish the URDF from the robot computer

    ros2 launch linorobot2_description description.launch.py

Optional parameters for simulation on host machine:

- **rviz** - Set to true to visualize the robot in rviz2 and only if you're configuring the URDF from the host machine. For example:

        ros2 launch linorobot2_description description.launch.py rviz:=true

#### 2.2 Visualize the robot from the host machine

The `rviz` argument on description.launch.py won't work on headless setup but you can visualize the robot remotely from the host machine:

    ros2 launch linorobot2_viz robot_model.launch.py

## Quickstart

All commands below are to be run on the robot computer unless you're running a simulation or rviz2 to visualize the robot remotely from the host machine. SLAM and Navigation launch files are the same for both real and simulated robots in Gazebo.

### 1. Booting up the robot

#### 1.1a Using a real robot

    ros2 launch linorobot2_bringup bringup.launch.py

Optional parameters:

- **base_serial_port** - Serial port of the robot's microcontroller. The assumed value is `/dev/ttyACM0`. Otherwise, change the default value to the correct serial port. For example:

    ros2 launch linorobot2_bringup bringup.launch.py base_serial_port:=/dev/ttyACM1

- **joy** - Set to true to run the joystick node in the background. (Tested on Logitech F710).

Always wait for the microROS agent to be connected before running any application (ie. creating a map or autonomous navigation). Once connected, the agent will print:

    | Root.cpp             | create_client     | create
    | SessionManager.hpp   | establish_session | session established

The agent needs a few seconds to get reconnected (less than 30 seconds). Unplug and plug back in the microcontroller if it takes longer than usual.

#### 1.1b Using Gazebo

    ros2 launch linorobot2_gazebo gazebo.launch.py

linorobot2_bringup.launch.py or gazebo.launch.py must always be run on a separate terminal before creating a map or robot navigation when working on a real robot or gazebo simulation respectively.

### 2. Controlling the robot

#### 2.1  Keyboard Teleop

Run [teleop_twist_keyboard](https://index.ros.org/r/teleop_twist_keyboard/) to control the robot using your keyboard:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Press:

- **i** - To drive the robot forward.
- **,** - To reverse the robot.
- **j** - To rotate the robot CCW.
- **l** - To rotate the robot CW.
- **shift + j** - To strafe the robot to the left (for mecanum robots).
- **shift + l** - To strafe the robot to the right (for mecanum robots).
- **u / o / m / .** - Used for turning the robot, combining linear velocity x and angular velocity z.

#### 2.2 Joystick

Pass `joy` argument to the launch file and set it to true to enable the joystick. For example:

    ros2 launch linorobot2_bringup bringup.launch.py joy:=true

- On F710 Gamepad, the top switch should be set to 'X' and the 'MODE' LED should be off.

Press Button/Move Joystick:

- **RB (First top right button)** - Press and hold this button while moving the joysticks to enable control.
- **Left Joystick Up/Down** - To drive the robot forward/reverse.
- **Left Joystick Left/Right** - To strafe the robot to the left/right.
- **Right Joystick Left/Right** - To rotate the robot CW/CCW.

### 3. Creating a map

#### 3.1 Run [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox)

    ros2 launch linorobot2_navigation slam.launch.py

Optional parameters for simulation on host machine:

For example:

    ros2 launch linorobot2_navigation slam.launch.py rviz:=true sim:=true

- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.

#### 3.1 Run rviz2 to visualize the robot from host machine

The `rviz` argument on slam.launch.py won't work on headless setup but you can visualize the robot remotely from the host machine:

    ros2 launch linorobot2_viz slam.launch.py

#### 3.2 Move the robot to start mapping

Drive the robot manually until the robot has fully covered its area of operation. Alternatively, the robot can also receive goal poses to navigate autonomously while mapping:

    ros2 launch nav2_bringup navigation_launch.py

- Pass `use_sim_time:=true` to the launch file when running in simulation.

More info [here](https://navigation.ros.org/tutorials/docs/navigation2_with_slam.html).

#### 3.3 Save the map

    cd linorobot2/linorobot2_navigation/maps
    ros2 run nav2_map_server map_saver_cli -f <map_name> --ros-args -p save_map_timeout:=10000.

### 4. Autonomous Navigation

#### 4.1 Load the map you created

Open linorobot2/linorobot2_navigation/launch/navigation.launch.py and change _MAP_NAME_ to the name of the newly created map. Build the robot computer's workspace once done:

    cd <robot_computer_ws>
    colcon build

Alternatively, `map` argument can be used when launching Nav2 (next step) to dynamically load map files. For example:

    ros2 launch linorobot2_navigation navigation.launch.py map:=<path_to_map_file>/<map_name>.yaml

#### 4.2 Run [Nav2](https://navigation.ros.org/tutorials/docs/navigation2_on_real_turtlebot3.html) package

    ros2 launch linorobot2_navigation navigation.launch.py

Optional parameter for loading maps:

- **map** - Path to newly created map <map_name.yaml>.

Optional parameters for simulation on host machine:

- **sim** - Set to true for simulated robots on the host machine. Default value is false.
- **rviz** - Set to true to visualize the robot in RVIZ. Default value is false.

#### 4.3 Run rviz2 to visualize the robot from host machine

The `rviz` argument for navigation.launch.py won't work on headless setup but you can visualize the robot remotely from the host machine:

    ros2 launch linorobot2_viz navigation.launch.py

Check out Nav2 [tutorial](https://navigation.ros.org/tutorials/docs/navigation2_on_real_turtlebot3.html#initialize-the-location-of-turtlebot-3) for more details on how to initialize and send goal pose.

navigation.launch.py will continue to throw this error `Timed out waiting for transform from base_link to map to become available, tf error: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist` until the robot's pose has been initialized.

## Troubleshooting Guide

### 1. The changes I made on a file are not taking effect on the package configuration/robot's behavior

- You need to build your workspace every time you modify a file:

    cd <ros2_ws>
    colcon build
    #continue what you're doing...

### 2. [`slam_toolbox]: Message Filter dropping message: frame 'laser'`

- Try to up `transform_timeout` by 0.1 in linorobot2_navigation/config/slam.yaml until the warning is gone.

### 3. `target_frame - frame does not exist`

- Check your <robot_type>.properties.urdf.xacro and ensure that there's no syntax errors or repeated decimal points.

### 4. Weird microROS agent behavior after updating the Linux/ROS

- Don't forget to update the microROS agent as well after your updates. Just run:

    bash update_microros.bash

## Useful Resources

- [ROS Navigation setup guide](https://navigation.ros.org/setup_guides/index.html)
- [Gazebo overview](http://gazebosim.org/tutorials/?tut=ros2_overview)
