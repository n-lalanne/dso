//
// Created by rakesh on 03/10/17.
//

#ifndef DSO_DATASETITERATOR_H
#define DSO_DATASETITERATOR_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <ctime>
#include <string>
#include <random>
#include <fstream>
#include <vector>

namespace dso_vi
{

/**
 *
 * @brief data structure for reading EUROC dataset
 * @details the data file is a CSV file with comments starting from # symbol
 * @details Eg: the groundtruth state estimate, imu data
 */
class DatasetIterator
{
public:
    /**
     *
     * @param file_path the file that contains CSV data file
     */
    DatasetIterator(const std::string file_path);

    /**
     *
     * @brief destructor
     */
    ~DatasetIterator();

    /**
     *
     * @param start_timestamp starting timestamp (secs)
     * @param end_timestamp end timestamp (secs)
     * @return vector of measurements between the start and end timestamp, each measurement being a vector of doubles
     * @return the first reading in a measurement is always a timestamp (secs)
     */
    std::vector< std::vector<double> >  getDataBetween(
            double start_timestamp, double end_timestamp
    );

    /**
     *
     * @return get the next measurement in the data file
     */
    std::vector<double> next();

    void stash(std::vector<double> measurement) { mStashedMeasurement = measurement; mIsStashed = true; }

protected:
    std::string mFilename;
    std::ifstream mFileStream;
    // a measurement is stashed if it is comes after the end time (so that it can be returned the next time)
    bool mIsStashed;
    std::vector<double> mStashedMeasurement;
};

} // namespace dso_vi


#endif //DSO_DATASETITERATOR_H
