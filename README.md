# ARCog-NET
Basic packages for simulating a mission

1. Install ROS noetic
2. Install MAVROS (https://docs.px4.io/main/en/ros/mavros_installation.html#install-ros-and-px4)
3. Put the file "multi_uav_mavros_sitl.launch" in ~/catkin_ws/src/Firmware/launch folder
4. Put the folders "gramado", "sun", "helideck" and "turbina_eolica" in ~/catkin_ws/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models folder
5. Put the file "empty_turbinas.world" in ~/catkin_ws/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds folder and rename it as "empty.world"
6. Put the folder "missoes" in ~/catkin_ws/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/
7. Go to ~/catkin_ws/src/Firmware/ and launch simulation with the command "roslaunch px4 multi_uav_mavros_sitl.launch"
8. Then go to ~/catkin_ws/src/Firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/missoes in another terminal and lauch "python3 launch_offboard.py"
9. In another terminal on the same folder, launch "python3 server.py"
