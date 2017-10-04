SET(PROJECT_NAME DSO)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_CONFIGURATION_TYPES  Debug Relase MinSizeRel RelWithDebInfo)
#set(BUILD_TYPE Release)
#set(BUILD_TYPE RelWithDebInfo)

set(INSTALL_PREFIX install)
set(DEVEL_PREFIX devel)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses Installed package
include_directories(${GTSAM_INCLUDE_DIR})
message(STATUS "Gtsam includes ${GTSAM_INCLUDE_DIR}")
# include_directories(/usr/local/include/gtsam/3rdparty/Eigen/)
set(GTSAM_LIBRARIES gtsam)

find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
include(GtsamTesting)

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV 2.4 QUIET)

# ROS
#find_package(sensor_msgs)
#find_package(rosbag)
#find_package(cv_bridge)
#
#set(ROS_INCLUDES ${sensor_msgs_INCLUDE_DIRS} ${rosbag_INCLUDE_DIRS} ${cv_bridge_INCLUDE_DIRS})
#set(ROS_LIBS ${sensor_msgs_LIBRARIES} ${rosbag_LIBRARIES} ${cv_bridge_LIBRARIES})
#message("-- ROS_INCLUDES ${ROS_INCLUDES}")
#message("-- ROS_LIBS ${ROS_LIBS}")

# flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
		# "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
		## no optimization + debugging
		"${SSE_FLAGS} -std=c++0x -march=native -fopenmp -fPIC -O0 -g3 -ggdb"
		#   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)

# remove optimization for debugging
#string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE )
#IF(CMAKE_BUILD_TYPE MATCHES "debug")
#	IF(CMAKE_COMPILER_IS_GNUCC)
#		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#		set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
#		message("-- disabling optimization for debug")
#	ENDIF(CMAKE_COMPILER_IS_GNUCC)
#ENDIF(CMAKE_BUILD_TYPE MATCHES "debug")

if (MSVC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif (MSVC)

# Sources files
set(dso_SOURCE_FILES
		${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystem.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptimize.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemOptPoint.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemDebugStuff.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/FullSystemMarginalize.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/Residuals.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseTracker.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/CoarseInitializer.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/ImmaturePoint.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/HessianBlocks.cpp
		${PROJECT_SOURCE_DIR}/src/FullSystem/PixelSelector2.cpp
		${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctional.cpp
		${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedTopHessian.cpp
		${PROJECT_SOURCE_DIR}/src/OptimizationBackend/AccumulatedSCHessian.cpp
		${PROJECT_SOURCE_DIR}/src/OptimizationBackend/EnergyFunctionalStructs.cpp
		${PROJECT_SOURCE_DIR}/src/util/settings.cpp
		${PROJECT_SOURCE_DIR}/src/util/Undistort.cpp
		${PROJECT_SOURCE_DIR}/src/util/globalCalib.cpp
		${PROJECT_SOURCE_DIR}/src/util/FrameShell.cpp

		# VI
		${PROJECT_SOURCE_DIR}/src/IMU/imudata.cpp
		${PROJECT_SOURCE_DIR}/src/IMU/configparam.cpp
#		${PROJECT_SOURCE_DIR}/src/IMU/MsgSync/MsgSynchronizer.cpp

		# Dataset reading
		${PROJECT_SOURCE_DIR}/src/DatasetIterator/DatasetIterator.cpp src/DatasetIterator/DatasetIterator.h
		${PROJECT_SOURCE_DIR}/src/DatasetIterator/ImuIterator.cpp src/DatasetIterator/ImuIterator.h
		${PROJECT_SOURCE_DIR}/src/GroundTruthIterator/GroundTruthIterator.cpp src/GroundTruthIterator/GroundTruthIterator.h
)

include_directories(
		${PROJECT_SOURCE_DIR}/src
		${PROJECT_SOURCE_DIR}/thirdparty/Sophus
		${PROJECT_SOURCE_DIR}/thirdparty/sse2neon
		${EIGEN3_INCLUDE_DIR}

		# VI
		${PROJECT_SOURCE_DIR}/src/IMU
#		${PROJECT_SOURCE_DIR}/src/IMU/MsgSync
#		${ROS_INCLUDES}

		# Groundtruth
		${PROJECT_SOURCE_DIR}/src/GroundTruthIterator
		${PROJECT_SOURCE_DIR}/src/DatasetIterator
)


# decide if we have pangolin
if (Pangolin_FOUND)
	message("--- found PANGOLIN, compiling dso_pangolin library.")
	include_directories( ${Pangolin_INCLUDE_DIRS} )
	set(dso_pangolin_SOURCE_FILES
			${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
			${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinDSOViewer.cpp)
	set(HAS_PANGOLIN 1)
else ()
	message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
	message("    this means there will be no 3D display / GUI available for dso_dataset.")
	set(dso_pangolin_SOURCE_FILES )
	set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV at ${OpenCV_INCLUDE_DIRS}, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(dso_opencv_SOURCE_FILES
			${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
			${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp)
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV, not compiling dso_opencv library.")
	message("    this means there will be no image display, and image read / load functionality.")
	set(dso_opencv_SOURCE_FILES
			${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageDisplay_dummy.cpp
			${PROJECT_SOURCE_DIR}/src/IOWrapper/ImageRW_dummy.cpp src/main_dso_bag.cpp)
	set(HAS_OPENCV 0)
endif ()

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
	message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
	add_definitions(-DHAS_ZIPLIB=1)
	include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
	message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
	set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories( ${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR})
add_library(dso SHARED ${dso_SOURCE_FILES} ${dso_opencv_SOURCE_FILES} ${dso_pangolin_SOURCE_FILES})
target_link_libraries(dso gtsam boost_system boost_thread cxsparse ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

#set_property( TARGET dso APPEND_STRING PROPERTY COMPILE_FLAGS -Wall )


# build main executable (only if we have both OpenCV and Pangolin)
if (OpenCV_FOUND AND Pangolin_FOUND)
	#	message("--- compiling dso_dataset.")
	#	add_executable(dso_dataset ${PROJECT_SOURCE_DIR}/src/main_dso_pangolin.cpp )
	#    target_link_libraries(dso_dataset dso boost_system boost_thread cxsparse ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

#	message("--- compiling dso_bag.")
#	add_executable(dso_bag ${PROJECT_SOURCE_DIR}/src/main_dso_bag.cpp )
#	target_link_libraries(dso_bag dso boost_system boost_thread cxsparse ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBS})

	message("--- compiling dso_imu.")
	add_executable(dso_imu ${PROJECT_SOURCE_DIR}/src/main_dso_imu.cpp )
	target_link_libraries(dso_imu dso boost_system boost_thread cxsparse ${LIBZIP_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
else()
	message("--- not building dso_dataset, since either don't have openCV or Pangolin.")
endif()

# tests
#add_executable(matrix_assignments_test ${PROJECT_SOURCE_DIR}/src/tests/matrix_assignments.cpp)
#target_link_libraries(matrix_assignments_test dso)
#
#add_executable(imu_errors_test ${PROJECT_SOURCE_DIR}/src/tests/imu_errors.cpp)
#target_link_libraries(imu_errors_test dso gtsam)

#add_executable(two_pose_vision_optimize ${PROJECT_SOURCE_DIR}/src/tests/two_pose_vision_optimize.cpp)
#target_link_libraries(two_pose_vision_optimize dso)

#add_executable(dataset_iterator ${PROJECT_SOURCE_DIR}/src/tests/dataset_iterator.cpp)
#target_link_libraries(dataset_iterator dso)