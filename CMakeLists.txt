cmake_minimum_required(VERSION 2.8.3)
project(agrotec_mrcnn)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs message_generation)

catkin_python_setup()

add_message_files(
  FILES
    Result.msg
)

generate_messages(
  DEPENDENCIES std_msgs sensor_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)