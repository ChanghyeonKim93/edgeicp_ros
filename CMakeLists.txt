cmake_minimum_required(VERSION 2.8.3)
project(edgeicp)

find_package(catkin REQUIRED cmake_modules)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
  message_filters
)

catkin_package(
	DEPENDS Eigen
)

set(includeDirs ${includeDirs} ${CMAKE_CURRENT_BINARY_DIR})
set(includeDirs ${includeDirs} include)

set(SRC ${SRC}
	src/main_script.cpp
	src/edgeicp.cpp
	src/Liegroup.cpp
	src/KDTree.cpp
	src/random_function.cpp

	include/edgeicp.h
	include/Liegroup.h
	include/KDTree.h
	include/MyHeaps.h
	include/random_function.h
)


include_directories( ${includeDirs} )
include_directories( ${catkin_INCLUDE_DIRS} )
include_directories( ${Eigen_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -O2 -m64 -Ofast -flto -msse2 -msse3 -march=native -mfpmath=sse")
set(DCMAKE_build_type release)

add_executable (edgeicp_node ${SRC} )
target_link_libraries (edgeicp_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries (edgeicp_node ${Eigen_LIBRARIES} )
