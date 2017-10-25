//
// Created by rakesh on 03/10/17.
//

#include "DatasetIterator.h"
#include <sstream>
#include <cassert>
#include <stdexcept>

// the timestamp is in seconds, we need microsecond precision
#define TIMESTAMP_PRECISION_FACTOR 1e4

namespace dso_vi
{

DatasetIterator::DatasetIterator(const std::string file_path) :
    mIsStashed(false), mFilename(file_path)
{
    mFileStream.open(file_path);

    if (!mFileStream) {
        std::cerr << "Could not open file " << file_path << std::endl;
        throw std::exception();
    }

}

DatasetIterator::~DatasetIterator()
{
    mFileStream.close();
}

std::vector<double> DatasetIterator::next()
{
    if (mIsStashed) {
        mIsStashed = false;

        // Don't read new IMU measurement, give the previous one
        return mStashedMeasurement;
    }

    std::vector<double> measurement;
    measurement.clear();

    if (mFileStream.good()) {
        std::string line;
        std::stringstream line_stream;


        std::getline(mFileStream, line, '\n');

        if (line.empty()) {
            // return empty measurement to denote error
            // TODO: proper error handling
            std::cerr << "empty line in " << mFilename << std::endl;
            return next();

        }
        if (line[0] == '#') {
            // get next measurement
            return next();
        } else {
            std::string str_reading;
            line_stream.clear();
            line_stream << line;

            while (std::getline(line_stream, str_reading, ',')) {
                std::stringstream ss;
                double n_reading;
                ss << str_reading;
                ss >> n_reading;
                measurement.push_back(n_reading);
            }

            if (measurement.size()) {
                // the timestamp we get is in nano seconds, convert to secs
                measurement[0] /= 1e9;
            }
        }

    } else {
        // no measurement
        std::cerr << "End of measurement in file " << mFilename << std::endl;
        throw std::exception();
    }

    return measurement;
}

std::vector< std::vector<double> > DatasetIterator::getDataBetween(double start_timestamp, double end_timestamp)
{
    assert(start_timestamp < end_timestamp);

    // timestamps are in arbitrary units after applying precision factor
    start_timestamp = round(start_timestamp * TIMESTAMP_PRECISION_FACTOR);
    end_timestamp = round(end_timestamp * TIMESTAMP_PRECISION_FACTOR);

    std::vector< std::vector<double> > measurements;
    measurements.reserve(11);

    while (true) {
        std::vector<double> measurement = next();

        if (!measurement.size()) {
            // No measurement
            std::cerr << "Error reading file " << mFilename << std::endl;
            throw std::out_of_range("file overrun");
        }

        double measurement_timestamp = round(measurement[0] * TIMESTAMP_PRECISION_FACTOR);
        if (measurement_timestamp >= start_timestamp && measurement_timestamp < end_timestamp) {
            measurements.push_back(measurement);
        } else if (measurement_timestamp >= end_timestamp) {
            if (measurement_timestamp > end_timestamp) {
                // stash the measurement for next time
                stash(measurement);
            } else {
                // if timestamp of measurement and the end timestamp is same, give it back and stash for next too
                stash(measurement);
                measurements.push_back(measurement);
            }

            return measurements;
        }
    }
}


} // namespace dso_vi
