//
// Created by rakesh on 03/10/17.
//

// number of data (including timestamp)
// state estimates
#define MAX_READING_IN_GROUND_TRUTH 17
// vicon data
#define MIN_READING_IN_GROUND_TRUTH 8

#include "GroundTruthIterator.h"
#include <cassert>

namespace dso_vi
{

GroundTruthIterator::ground_truth_measurement_t GroundTruthIterator::vectorToStruct(
        std::vector<double> measurementVector
)
{
    assert(measurementVector.size() >= MIN_READING_IN_GROUND_TRUTH);

    ground_truth_measurement_t measurementStruct;
    measurementStruct.timestamp = measurementVector[0];
    measurementStruct.pose = gtsam::Pose3(
      gtsam::Rot3::quaternion(measurementVector[4], measurementVector[5], measurementVector[6], measurementVector[7]),
      gtsam::Point3(measurementVector[1], measurementVector[2], measurementVector[3])
    );

    if (measurementVector.size() >= MAX_READING_IN_GROUND_TRUTH)
    {
        measurementStruct.velocity = gtsam::Velocity3(
                measurementVector[8], measurementVector[9], measurementVector[1]
        );
        measurementStruct.gyroBias << measurementVector[1], measurementVector[1], measurementVector[1];
        measurementStruct.acceleroBias << measurementVector[1], measurementVector[1], measurementVector[1];
    }

    return measurementStruct;
}

gtsam::Pose3 GroundTruthIterator::getGroundTruthBetween(double start_timestamp, double end_timestamp,
                                                        GroundTruthIterator::ground_truth_measurement_t &start_ground_truth,
                                                        GroundTruthIterator::ground_truth_measurement_t &end_ground_truth)
{
    std::vector< std::vector<double> > measurements = getDataBetween(start_timestamp, end_timestamp);

    // invalid initialization
    start_ground_truth.timestamp = 0;
    end_ground_truth.timestamp = 0;

    // sanity checking
    if  (
            measurements.size() < 2 ||
            measurements[0].size() < MIN_READING_IN_GROUND_TRUTH ||
            measurements.back().size() < MIN_READING_IN_GROUND_TRUTH
        )
    {
        return gtsam::Pose3();
    }

    start_ground_truth = vectorToStruct(measurements[0]);
    end_ground_truth = vectorToStruct(measurements.back());

    return start_ground_truth.pose.between(end_ground_truth.pose);
}

}; // namespace dso_vi