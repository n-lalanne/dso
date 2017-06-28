/**
 *
 * Class GroundTruthIterator
 * Data structure for getting ground truth readings
 *
 */

#ifndef GROUND_TRUTH_ITERATOR_H
#define GROUND_TRUTH_ITERATOR_H

#include <iostream>

#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <ctime>
#include <random>
#include <fstream>
#include <iostream>

// GTSAM related includes.
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Dense>

// No of reading in ground truth file (not including the timestamp)
// can be either 7 (x,y,z,qw,qx,qy,qz) like VICON or 16 (+ v_x,v_y,v_z,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z) like the state estimate
#define NO_OF_READING_IN_GROUND_TRUTH 16

namespace dso_vi
{

class GroundTruthIterator
{
public:
  typedef struct {
    
    std::time_t timestamp;
    gtsam::Pose3 pose;
    gtsam::Velocity3 velocity;
    Eigen::Vector3d gyroBias;
    Eigen::Vector3d acceleroBias;
    
  } ground_truth_measurement_t;

  GroundTruthIterator(const std::string file_path);

  ~GroundTruthIterator();

  GroundTruthIterator::ground_truth_measurement_t next();
  gtsam::Pose3 getGroundTruthBetween(
    double start_timestamp, double end_timestamp, 
    GroundTruthIterator::ground_truth_measurement_t &start_ground_truth,
    GroundTruthIterator::ground_truth_measurement_t &end_ground_truth
  );
  
  void stash() { mIsStashed = true; }


protected:

  std::ifstream mFileStream;
  bool mIsStashed;
  GroundTruthIterator::ground_truth_measurement_t mStashedMeasurement;
};

} // namespace dso_vi

#endif