/**
 *
 * Class GroundTruthIterator
 * Data structure for getting ground truth readings
 *
 */

#include "GroundTruthIterator.h"
#include <stdexcept>

// the timestamp is in seconds, we need microsecond precision
#define TIMESTAMP_PRECISION_FACTOR 1e4
namespace dso_vi
{

    GroundTruthIterator::GroundTruthIterator(const std::string file_path):
            mIsStashed(false)
    {

      mFileStream.open(file_path);

      // ignore the first line
      std::string tmp_string;
      std::getline(mFileStream, tmp_string);

    }

    GroundTruthIterator::~GroundTruthIterator()
    {
      mFileStream.close();
    }


    GroundTruthIterator::ground_truth_measurement_t GroundTruthIterator::next()
    {

      if (mIsStashed) {
        mIsStashed = false;

        // Don't read new IMU measurement, give the previous one
        return mStashedMeasurement;
      }

      std::string tmp_string;
      std::stringstream ss;

      if (mFileStream.good()) {

        std::getline(mFileStream, tmp_string, ',');

        if (tmp_string.empty()) {

          mStashedMeasurement.timestamp = 0;

        } else {

          ss.clear();
          ss << tmp_string;
          ss >> mStashedMeasurement.timestamp;
          // the groundtruth we get is in nano seconds
          mStashedMeasurement.timestamp /= 1e9;

          Eigen::Matrix<double,NO_OF_READING_IN_GROUND_TRUTH,1> ground_truth = Eigen::Matrix<double,NO_OF_READING_IN_GROUND_TRUTH,1>::Zero();

          for (int i=0; i<(NO_OF_READING_IN_GROUND_TRUTH-1); ++i) {

            std::getline(mFileStream, tmp_string, ',');

            ss.clear();
            ss << tmp_string;
            ss >> ground_truth(i);

          }

          std::getline(mFileStream, tmp_string, '\n');

          ss.clear();
          ss << tmp_string;
          ss >> ground_truth(NO_OF_READING_IN_GROUND_TRUTH-1);

          mStashedMeasurement.pose = gtsam::Pose3 (

                  gtsam::Rot3::quaternion(ground_truth(3), ground_truth(4), ground_truth(5), ground_truth(6)),
                  gtsam::Point3(ground_truth(0), ground_truth(1), ground_truth(2))

          );

          if (ground_truth.size() == 16) {
            mStashedMeasurement.velocity = gtsam::Velocity3 (
                    ground_truth(7), ground_truth(8), ground_truth(9)
            );
            mStashedMeasurement.gyroBias << ground_truth(10), ground_truth(11), ground_truth(12);
            mStashedMeasurement.acceleroBias << ground_truth(13), ground_truth(14), ground_truth(15);
          }

        }

      } else {

        mStashedMeasurement.timestamp = 0;

      }

      return mStashedMeasurement;
    }

    gtsam::Pose3 GroundTruthIterator::getGroundTruthBetween(
            double start_timestamp, double end_timestamp,
            GroundTruthIterator::ground_truth_measurement_t &start_ground_truth,
            GroundTruthIterator::ground_truth_measurement_t &end_ground_truth
    )
    {
      assert(start_timestamp < end_timestamp);

      start_timestamp = round(start_timestamp * TIMESTAMP_PRECISION_FACTOR);
      end_timestamp = round(end_timestamp * TIMESTAMP_PRECISION_FACTOR);

      bool is_first_ground_truth = true;

      GroundTruthIterator::ground_truth_measurement_t previous_ground_truth;

      while (true)
      {
        GroundTruthIterator::ground_truth_measurement_t ground_truth = this->next();
        double ground_truth_timestamp = round(ground_truth.timestamp * TIMESTAMP_PRECISION_FACTOR);

        if (ground_truth.timestamp == 0)
        {
          // No ground truth
          std::cerr << "No ground truth between " << start_timestamp << " - " << end_timestamp << std::endl;
          throw std::out_of_range("Groundtruth file overrun");
        }
        else if (ground_truth_timestamp >= start_timestamp && ground_truth_timestamp < end_timestamp)
        {
          if (is_first_ground_truth)
          {
            is_first_ground_truth = false;
            start_ground_truth = end_ground_truth = ground_truth;
          }

          previous_ground_truth = end_ground_truth;
          end_ground_truth = ground_truth;
        }
        else if (ground_truth_timestamp >= end_timestamp)
        {
          if (is_first_ground_truth)
          {
            // No ground truth
            std::cerr << "No ground before " << start_timestamp << std::endl;
            throw std::out_of_range("Groundtruth file overrun");
          }

          if (ground_truth_timestamp > end_timestamp)
          {
            // we want the previous ground truth value and stash the current one
            end_ground_truth = previous_ground_truth;
            this->stash();
          }
          else
          {
            end_ground_truth = ground_truth;
            this->stash();
          }

          return start_ground_truth.pose.between(end_ground_truth.pose);
        }
      }
    }

} // namespace dso_vi