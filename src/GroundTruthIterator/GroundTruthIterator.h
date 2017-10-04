//
// Created by rakesh on 03/10/17.
//

#ifndef DSO_GROUNDTRUTHITERATOR_H
#define DSO_GROUNDTRUTHITERATOR_H

#include "DatasetIterator/DatasetIterator.h"

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

namespace dso_vi
{

/**
 *
 * @brief wrapper class for groundtruth from dataset iterator
 */
class GroundTruthIterator : public DatasetIterator {
public:
    typedef struct {

        double timestamp;
        gtsam::Pose3 pose;
        gtsam::Velocity3 velocity;
        Eigen::Vector3d gyroBias;
        Eigen::Vector3d acceleroBias;

    } ground_truth_measurement_t;

    // inherit the damn constructor
    using DatasetIterator::DatasetIterator;

    /**
     *
     * @brief get the groundtruth between two [camera] timestamp
     * @param start_timestamp
     * @param end_timestamp
     * @param start_ground_truth
     * @param end_ground_truth
     * @return relative pose between the two
     */
    gtsam::Pose3 getGroundTruthBetween(
            double start_timestamp, double end_timestamp,
            GroundTruthIterator::ground_truth_measurement_t &start_ground_truth,
            GroundTruthIterator::ground_truth_measurement_t &end_ground_truth
    );

    /**
     *
     * @brief convert the raw vector of measurement of the built-in ground_truth_measurement_t structure
     * @param measurementVector
     * @return ground_truth_measurement_t structure corresponding to the vector
     */
    ground_truth_measurement_t vectorToStruct(std::vector<double> measurementVector);

};

} // namespace dso_vi

#endif //DSO_GROUNDTRUTHITERATOR_H
