//
// Created by rakesh on 03/10/17.
//

#ifndef DSO_IMUITERATOR_H
#define DSO_IMUITERATOR_H

#include "DatasetIterator/DatasetIterator.h"
#include "IMU/imudata.h"

namespace dso_vi
{

/**
 *
 * @brief wrapper class for groundtruth from dataset iterator
 */
class ImuIterator: public DatasetIterator
{
public:
    // inherit the damn constructor
    using DatasetIterator::DatasetIterator;

    /**
     *
     * @brief get the imu between two [camera] timestamp
     * @param start_timestamp
     * @param end_timestamp
     * @return relative pose between the two
     */
    std::vector<IMUData> getImuBetween(
            double start_timestamp, double end_timestamp
    );

    /**
     *
     * @brief convert the raw vector of measurement of the built-in IMUData structure
     * @param measurementVector
     * @return IMUData structure corresponding to the vector
     */
    IMUData vectorToStruct(std::vector<double> measurementVector);
};

} // namespace dso_vi
#endif //DSO_IMUITERATOR_H
