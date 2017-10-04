//
// Created by rakesh on 03/10/17.
//

#include <iostream>
#include <algorithm>
#include <sstream>
#include "DatasetIterator/DatasetIterator.h"
#include "DatasetIterator/ImuIterator.h"
#include "GroundTruthIterator/GroundTruthIterator.h"
#include "util/DatasetReader.h"

using namespace dso_vi;

double filenameToTimestamp(std::string filename)
{
    // get rid of the extension
    std::string str(filename);
    for (int i = str.length() - 1; i >= 0; i--)
    {
        if (str[i] == '.')
        {
            str = std::string(str.substr(0, i));
            break;
        }
    }
    // get the string after the last '/' to get the filename
    for (int i = str.length() - 2; i >= 0; i--)
    {
        if (str[i] == '/')
        {
            str = std::string(str.substr(i+1));
            break;
        }
    }

    std::stringstream ss;
    ss << str;
    double n;
    ss >> n;
    // timestamp in nano sec, convert to sec
    return n/1e9;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./dataset_iterator <image_dir> <data_file>" << std::endl;
        return 1;
    }

    std::string image_dir(argv[1]);
    std::vector<std::string> image_files;
    getdir(image_dir, image_files);

    for (std::string &image_file: image_files)
    {
        std::cout << image_file << std::endl;
    }

    // find the timestamp from filename
    std::vector<double> image_timestamps;
    std::transform(image_files.begin(), image_files.end(), std::back_inserter(image_timestamps), filenameToTimestamp);

    std::string data_file(argv[2]);
    DatasetIterator dataset_iterator(data_file);
//    GroundTruthIterator groundtruth_iterator(data_file);
    ImuIterator imu_iterator(data_file);

    for (size_t image_idx = 0; image_idx < image_timestamps.size()-1; image_idx++)
    {
        std::vector< std::vector<double> > measurements = dataset_iterator.getDataBetween(image_timestamps[image_idx], image_timestamps[image_idx+1]);

        std::cout << std::fixed <<  std::setprecision(9);
        std::cout << "between " << image_timestamps[image_idx] << " and " << image_timestamps[image_idx+1] << " " << measurements.size() << std::endl;
        for (std::vector<double> &measurement: measurements)
        {
            std::cout << measurement[0] << ", ";
        }
        std::cout << std::endl;

//        GroundTruthIterator::ground_truth_measurement_t start_gt, end_gt;
//        groundtruth_iterator.getGroundTruthBetween(image_timestamps[image_idx], image_timestamps[image_idx+1], start_gt, end_gt);
//        std::cout << start_gt.timestamp << "--" << end_gt.timestamp << std::endl;

        std::vector<IMUData> vImuData = imu_iterator.getImuBetween(image_timestamps[image_idx], image_timestamps[image_idx+1]);
        if (vImuData.size() >= 2)
        {
            std::cout << "IMUs: " << vImuData[0]._t << " -- " << vImuData.back()._t << ": " << vImuData.size() << std::endl;
        }

        std::cout << std::endl << std::endl;
    }
}