//
// Created by rakesh on 03/10/17.
//

#include "ImuIterator.h"

namespace dso_vi
{

IMUData ImuIterator::vectorToStruct(std::vector<double> measurementVector)
{
    assert(measurementVector.size() == 7);
    return IMUData(
        // angular velocities
        measurementVector[1],measurementVector[2],measurementVector[3],
        // linear accelerations
        measurementVector[4],measurementVector[5],measurementVector[6],
        // timestamp
        measurementVector[0]
    );

}

std::vector<IMUData> ImuIterator::getImuBetween(double start_timestamp, double end_timestamp)
{
    std::vector< std::vector<double> > measurements = getDataBetween(start_timestamp, end_timestamp);

    std::vector<IMUData> vImuData;
    vImuData.reserve(10);

    for (std::vector<double> &measurement: measurements)
    {
        if (measurement[0] < start_timestamp)
        {
            continue;
        }
        if (measurement[0] >= end_timestamp)
        {
            break;
        }

        vImuData.push_back(vectorToStruct(measurement));
    }

    return vImuData;
}

}; // namespace dso_vi