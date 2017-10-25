//
// Created by sicong, rakesh on 25/08/17.
//
#include <stdlib.h>
#include <ctime>
#include <fstream>
#include <iostream>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>

using namespace gtsam;

PreintegrationType *imu_preintegrated_;

/**
 *
 * @brief Integrate imu measurements between two timestamps
 * @param imu_file
 * @param timestamp1
 * @param timestamp2
 * @param imu_preintegrated_
 * @return success status
 */
bool integrateImuMeasurements(std::ifstream &imu_file, std::time_t timestamp1, std::time_t timestamp2, PreintegrationType *imu_preintegrated_);

/**
 * From groundtruth file, gets two states
 */
void getStates(
        std::ifstream &ground_truth_file,
        unsigned int ground_truth_index_start,
        unsigned int ground_truth_index_length,

        std::time_t &timestamp1,
        Pose3 &pose1,
        Vector3 &velocity1,
        imuBias::ConstantBias &bias1,

        std::time_t &timestamp2,
        Pose3 &pose2,
        Vector3 &velocity2,
        imuBias::ConstantBias &bias2
);


int main(int argc, char **argv)
{
    if (argc < 5) {
        std::cout << "Usage: imu_errors <imu_file_name> <ground_truth_file_name> <ground_truth_index_start> <ground_truth_index_length>" << std::endl;
        return 0;
    }

    // ---------------------------- Parameters start ---------------------------- //

    // We use the sensor specs to build the noise model for the IMU factor.
    double accel_noise_sigma = 0.0003924;
    double gyro_noise_sigma = 0.000205689024915;
    double accel_bias_rw_sigma = 0.004905;
    double gyro_bias_rw_sigma = 0.000001454441043;
    Matrix33 measured_acc_cov = Matrix33::Identity(3,3) * pow(accel_noise_sigma,2);
    Matrix33 measured_omega_cov = Matrix33::Identity(3,3) * pow(gyro_noise_sigma,2);
    Matrix33 integration_error_cov = Matrix33::Identity(3,3)*1e-8; // error committed in integrating position from velocities
    Matrix33 bias_acc_cov = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2);
    Matrix33 bias_omega_cov = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2);
    Matrix66 bias_acc_omega_int = Matrix::Identity(6,6)*1e-5; // error in the bias used for preintegration

    boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedU(0.0);
    // PreintegrationBase params:
    p->accelerometerCovariance = measured_acc_cov; // acc white noise in continuous
    p->integrationCovariance = integration_error_cov; // integration uncertainty continuous
    // should be using 2nd order integration
    // PreintegratedRotation params:
    p->gyroscopeCovariance = measured_omega_cov; // gyro white noise in continuous
    // PreintegrationCombinedMeasurements params:
    p->biasAccCovariance = bias_acc_cov; // acc bias in continuous
    p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
    p->biasAccOmegaInt = bias_acc_omega_int;

    // ---------------------------- Parameters end ---------------------------- //

    // read the command-line arguments
    std::ifstream imu_file(argv[1]);
    std::ifstream ground_truth_file(argv[2]);
    unsigned int ground_truth_index_start = atoi(argv[3]);
    unsigned int ground_truth_index_length = atoi(argv[4]);

    std::time_t timestamp1;
    Pose3 pose1;
    Vector3 velocity1;
    imuBias::ConstantBias bias1;

    std::time_t timestamp2;
    Pose3 pose2;
    Vector3 velocity2;
    imuBias::ConstantBias bias2;

    getStates(
            ground_truth_file,
            ground_truth_index_start,
            ground_truth_index_length,

            timestamp1,
            pose1,
            velocity1,
            bias1,

            timestamp2,
            pose2,
            velocity2,
            bias2
    );

    NavState prev_state = NavState(pose1, velocity1);
    NavState actual_state = NavState(pose2, velocity2);

    std::cout << "---------------State1---------------" << std::endl;
    std::cout << timestamp1 << std::endl;
    std::cout << prev_state << std::endl;
    std::cout << bias1 << std::endl
            ;
    std::cout << "---------------State2---------------" << std::endl;
    std::cout << timestamp2 << std::endl;
    std::cout << actual_state << std::endl;
    std::cout << bias2 << std::endl;

    imu_preintegrated_ = new PreintegratedCombinedMeasurements(p, bias1);
    bool is_imu_measurement = integrateImuMeasurements(imu_file, timestamp1, timestamp2, imu_preintegrated_);

    std::cout << "IMU dt: " << imu_preintegrated_->deltaTij() << std::endl;
    std::cout << "Groundtruth dt: " << std::difftime(timestamp2, timestamp1) * 1e-9 << std::endl;

    if (is_imu_measurement) {


        NavState prop_state = imu_preintegrated_->predict(prev_state, bias1);
        std::cout << "---------------- Predicted: ----------------" << std::endl;
        std::cout << prop_state << std::endl;

        std::cout << "---------------- Actual: ----------------" << std::endl;
        std::cout << prev_state << std::endl;

        // Print out the position and orientation error for comparison.
        Vector3 position_error = prop_state.pose().translation() - actual_state.pose().translation();
        double current_position_error = position_error.norm();
        double position_direction_error = acos(
                prop_state.pose().translation().dot(actual_state.pose().translation()) /
                        (prop_state.pose().translation().norm() * actual_state.pose().translation().norm())
        ) * 180 / M_PI;
        Quaternion quat_error = (actual_state.pose().rotation().inverse() * prop_state.pose().rotation()).toQuaternion();
        quat_error.normalize();
        Vector3 euler_angle_error(quat_error.x()*2,
                                  quat_error.y()*2,
                                  quat_error.z()*2);
        double current_orientation_error = euler_angle_error.norm() * 180 / M_PI;

        // display statistics
        std::cout << "Position error:" << current_position_error << "\t "
                  << "Position direction error: " << position_direction_error << "\t"
                  << "Angular error:" << current_orientation_error << std::endl;
    }

    return 0;

}

bool integrateImuMeasurements(std::ifstream &imu_file, std::time_t timestamp1, std::time_t timestamp2, PreintegrationType *imu_preintegrated_)
{
    // -------------------- Read IMU measurements -------------------- //
    bool is_imu_measurement = false;
    std::string tmp_value;
    std::time_t old_timestamp = timestamp1;
    // skip the first line (just header comment)
    std::getline(imu_file, tmp_value);

    std::cout << "------------------IMU------------------" << std::endl;
    while (imu_file.good()) {
        std::time_t timestamp;
        std::getline(imu_file, tmp_value, ',');
        timestamp = atof(tmp_value.c_str());
        Eigen::Matrix<double,6,1> imu;

        if (timestamp >= timestamp1 && timestamp <= timestamp2) {

            imu = Eigen::Matrix<double,6,1>::Zero();
            for (int i=0; i<5; ++i) {
                std::getline(imu_file, tmp_value, ',');
                imu(i) = atof(tmp_value.c_str());
            }
            std::getline(imu_file, tmp_value, '\n');
            imu(5) = atof(tmp_value.c_str());

            is_imu_measurement = true;
//            std::cout << timestamp << std::endl;

        } else if (timestamp > timestamp2) {
            break;
        } else {
            std::getline(imu_file, tmp_value);
        }

        // Adding the IMU preintegration.
        imu_preintegrated_->integrateMeasurement(
                imu.head<3>(),
                imu.tail<3>(),
                (timestamp - old_timestamp) * 1e-9 // timestamp is in ns (we need seconds)
        );

//        std::cout << "Delta T: " << (timestamp - old_timestamp) << std::endl;

        old_timestamp = timestamp;
    }

    return is_imu_measurement;
}

/*
 * From groundtruth file, gets two top consecutive states
 */
void getStates(
        std::ifstream &ground_truth_file,
        unsigned int ground_truth_index_start,
        unsigned int ground_truth_index_length,

        std::time_t &timestamp1,
        Pose3 &pose1,
        Vector3 &velocity1,
        imuBias::ConstantBias &bias1,

        std::time_t &timestamp2,
        Pose3 &pose2,
        Vector3 &velocity2,
        imuBias::ConstantBias &bias2
)
{
    std::string tmp_value;

    double x, y, z;
    double q_w, q_x, q_y, q_z;
    double v_x, v_y, v_z;
    double bg_x, bg_y, bg_z;
    double ba_x, ba_y, ba_z;

    Point3 t;
    Rot3 R;

    // the first line is a comment, so you are taking line ground_truth_index+1 and ground_truth_index+2
    // skip the lines before required index
    for (unsigned int i = 0; i <= ground_truth_index_start; i++) {
        std::string tmp_value;

        if (!ground_truth_file.good()) {
            std::cout << "Ground truth index out of bounds" << std::endl;
            exit(1);
        }
        std::getline(ground_truth_file, tmp_value);
    }

    // --------------------------for the first pose-------------------------- //
    // read the timestamp
    std::getline(ground_truth_file, tmp_value, ',');
    timestamp1 = atol(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_w = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    ba_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    ba_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value);
    ba_z = atof(tmp_value.c_str());

    t << Point3(x, y, z);
    R = Rot3::Quaternion(
            q_w, q_x, q_y, q_z
    );

    pose1 = Pose3(R, t);
    velocity1 << v_x, v_y, v_z;
    bias1 = imuBias::ConstantBias(
            Velocity3(ba_x, ba_y, ba_z),
            Velocity3(bg_x, bg_y, bg_z)
    );

    // skip the length of poses to get to the last (the first one is already skipped, so need to skip length-1)
    for (unsigned int i = 0; i < ground_truth_index_length - 1; i++) {
        std::string tmp_value;

        if (!ground_truth_file.good()) {
            std::cout << "Ground truth index out of bounds" << std::endl;
            exit(1);
        }
        std::getline(ground_truth_file, tmp_value);
    }

    // --------------------------for the second pose-------------------------- //
    // read the timestamp
    std::getline(ground_truth_file, tmp_value, ',');
    timestamp2 = atol(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_w = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    q_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    v_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    bg_z = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    ba_x = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value, ',');
    ba_y = atof(tmp_value.c_str());

    std::getline(ground_truth_file, tmp_value);
    ba_z = atof(tmp_value.c_str());

    t << Point3(x, y, z);
    R = Rot3::Quaternion(
            q_w, q_x, q_y, q_z
    );

    pose2 = Pose3(R, t);
    velocity2 << v_x, v_y, v_z;
    bias2 = imuBias::ConstantBias(
            Velocity3(ba_x, ba_y, ba_z),
            Velocity3(bg_x, bg_y, bg_z)
    );

}