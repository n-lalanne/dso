#include <GroundTruthIterator/GroundTruthIterator.h>
#include "FrameShell.h"

Mat1515 FrameShell::getIMUcovariance()
{
    PreintegratedCombinedMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedCombinedMeasurements*>(imu_preintegrated_last_frame_);
    return preint_imu->preintMeasCov();
}

Vec3 FrameShell::TWB()
{
    Mat44 Twc = camToWorld.matrix();
    Mat44 Twb = Twc * dso_vi::Tcb.matrix();

    return Twb.block<3,1>(0,3);
}

void FrameShell::linearizeImuFactor(
        gtsam::NavState previous_navstate,
        gtsam::NavState current_navstate,
        gtsam::imuBias::ConstantBias initial_bias
)
{
    PreintegratedCombinedMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedCombinedMeasurements*>(imu_preintegrated_last_frame_);

    delete imu_factor_;
    imu_factor_ = new CombinedImuFactor(
            X(0), V(0),
            X(1), V(1),
            B(0), B(1),
            *preint_imu
    );

    Values initial_values;
    initial_values.insert(X(0), previous_navstate.pose());
    initial_values.insert(X(1), current_navstate.pose());
    initial_values.insert(V(0), previous_navstate.velocity());
    initial_values.insert(V(1), current_navstate.velocity());
    initial_values.insert(B(0), initial_bias);
    initial_values.insert(B(1), this->bias);

    imu_factor_->linearize(initial_values);
}

Vec15 FrameShell::evaluateIMUerrors(
        gtsam::NavState previous_navstate,
        gtsam::NavState current_navstate,
        gtsam::imuBias::ConstantBias initial_bias,
        gtsam::Matrix &J_imu_Rt_i,
        gtsam::Matrix &J_imu_v_i,
        gtsam::Matrix &J_imu_Rt_j,
        gtsam::Matrix &J_imu_v_j,
        gtsam::Matrix &J_imu_bias_i,
        gtsam::Matrix &J_imu_bias_j
)
{

    assert(imu_factor_);
    Vec15 error = imu_factor_->evaluateError(
            previous_navstate.pose(), previous_navstate.velocity(), current_navstate.pose(), current_navstate.velocity(),
            initial_bias, this->bias,
            J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias_i, J_imu_bias_j
    );

    return error;
}

void FrameShell::updateIMUmeasurements(std::vector<dso_vi::IMUData> mvIMUSinceLastF)
{
    // IMU factors from the keyframe to last frame
    std::vector<PreintegrationType*> keyframe2LastFrameFactors;

    std::cout << "last frame/keyframe: " <<  last_frame->id << ", " << last_kf->id << std::endl;
    std::cout << "Frame ID: " << id << std::endl;
    if (id > last_kf->id + 1)
    { // only if the last kf is not the previous frame
        if (last_frame->last_kf && last_frame->last_kf->id == last_kf->id)
        {
            // KF hasn't changed, get the preintegrated values since the last KF from last frame
            keyframe2LastFrameFactors.push_back(last_frame->imu_preintegrated_last_kf_);
            std::cout << "IMU measurements: integrating till keyframe of " << last_frame->id << "(" << last_frame->last_kf->id << ")" << std::endl;
        }
        else
        {
            std::cout << "IMU measurements: integrating frames ";
            // integrate all the imu factors between previous keyframe and previous frame
            // need to to this in reverse order first because we only have pointers to previous frame, not previous keyframe
            for (
                    FrameShell *previousFramePointer = last_frame;
                    previousFramePointer->id > last_kf->id;
                    previousFramePointer = previousFramePointer->last_frame
                )
            {
                std::cout << previousFramePointer->id << ", ";
                keyframe2LastFrameFactors.push_back(previousFramePointer->imu_preintegrated_last_frame_);
                // make sure that the frame ids are going down
                assert(previousFramePointer->id > previousFramePointer->last_frame->id);
            }
            // reverse to get in right order (new keyframe to previous frame)
            std::reverse(keyframe2LastFrameFactors.begin(), keyframe2LastFrameFactors.end());
            std::cout << std::endl;
        }
    }

    printf("Between: %.7f, %.7f", last_frame->viTimestamp, viTimestamp);
    std::cout << "IMU count: " << mvIMUSinceLastF.size() << std::endl;
    for (size_t i = 0; i < mvIMUSinceLastF.size(); i++)
    {
        dso_vi::IMUData imudata = mvIMUSinceLastF[i];

        Mat61 rawimudata;
        rawimudata << 	imudata._a(0), imudata._a(1), imudata._a(2),
                        imudata._g(0), imudata._g(1), imudata._g(2);

        double dt = 0;
        // interpolate readings
        if (i == mvIMUSinceLastF.size() - 1)
        {
            dt = viTimestamp - imudata._t;
        }
        else
        {
            dt = mvIMUSinceLastF[i+1]._t - imudata._t;
        }

        if (i == 0)
        {
            // assuming the missing imu reading between the previous frame and first IMU
            dt += (imudata._t - last_frame->viTimestamp);
        }

        assert(dt >= 0);

        printf("%.7f: %f\n", imudata._t, dt);

        imu_preintegrated_last_frame_->integrateMeasurement(
                rawimudata.block<3,1>(0,0),
                rawimudata.block<3,1>(3,0),
                dt
        );
    }

    Matrix9 H1, H2;

    for (PreintegrationType *&intermediate_factor: keyframe2LastFrameFactors)
    {
        imu_preintegrated_last_kf_->mergeWith(*intermediate_factor, &H1, &H2);
    }

    imu_preintegrated_last_kf_->mergeWith(*imu_preintegrated_last_frame_, &H1, &H2);
    assert(imu_preintegrated_last_kf_->deltaTij() > 0.0);
}


gtsam::NavState FrameShell::PredictPose(gtsam::NavState ref_pose_imu, double lastTimestamp)
{


    // conversion from EUROC reference to our reference (T_dso_euroc = T_dso_ref * T_ref_euroc
    Mat44 T_dso_euroc = ref_pose_imu.pose().matrix() * last_frame->groundtruth.pose.inverse().matrix();
    std::cout << "DSO vs EuRoC rotation: \n" << T_dso_euroc.block<3,3>(0,0) << std::endl;

    ref_pose_imu = gtsam::NavState(
          ref_pose_imu.pose(),
          T_dso_euroc.block<3,3>(0,0) * last_frame->groundtruth.velocity
    );

    // set the velocities to groundtruth
    last_frame->navstate = gtsam::NavState(
            last_frame->navstate.pose(),
            T_dso_euroc.block<3,3>(0,0).transpose() * last_frame->groundtruth.velocity
    );

    std::cout<<"last id: "<<fh->shell->last_frame->id<<std::endl;
    std::cout<<"last_frame->id: "<<last_frame->id<<std::endl;
    std::cout << "GT vel transformed: " << ref_pose_imu.velocity().transpose() << std::endl;
    std::cout << "Vel initialized: " << last_frame->navstate.velocity().transpose() << std::endl;
//    std::cout << "GT Vel initialized: " << last_frame->groundtruth.velocity.transpose() << std::endl;

    gtsam::NavState predicted_pose_imu = imu_preintegrated_last_frame_->predict(last_frame->navstate, bias);

    // set the velocities to groundtruth
//    predicted_pose_imu = gtsam::NavState(
//            predicted_pose_imu.pose(),
//            T_dso_euroc.block<3,3>(0,0) * groundtruth.velocity
//    );

    if  (
            !std::isfinite(predicted_pose_imu.pose().translation().x()) ||
            !std::isfinite(predicted_pose_imu.pose().translation().y()) ||
            !std::isfinite(predicted_pose_imu.pose().translation().z())
        )
    {
        if (!setting_debugout_runquiet)
        {
            std::cout << "IMU prediction nan for translation!!!" << std::endl;
        }
        predicted_pose_imu = gtsam::NavState(
                gtsam::Pose3(ref_pose_imu.pose().rotation(), Vec3::Zero()),
                ref_pose_imu.velocity()
        );
    }

    {
        // run a dummy optimization here
        gtsam::NavState previous_navstate(last_frame->groundtruth.pose, last_frame->groundtruth.velocity);
        gtsam::NavState current_navstate = imu_preintegrated_last_frame_->predict(previous_navstate, bias);

        // add noise to the estimate
        const double NOISE_LEVEL = 0.1;

        previous_navstate = NavState(
                previous_navstate.pose().compose(
                        Pose3::Expmap(
                                (Vec6() << NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL).finished()
                        )
                ),
                previous_navstate.velocity() + (Vec3() << NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL).finished()
        );

        current_navstate = NavState(
                current_navstate.pose().compose(
                        Pose3::Expmap(
                                (Vec6() << NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL).finished()
                        )
                ),
                current_navstate.velocity() + (Vec3() << NOISE_LEVEL, NOISE_LEVEL, NOISE_LEVEL).finished()
        );

        Mat1515 inv_covariance = getIMUcovariance().inverse();

        // useless Jacobians of reference frame (cuz we're not optimizing reference frame)
        gtsam::Matrix J_imu_Rt_i, J_imu_v_i, J_imu_bias_i;
        gtsam::Matrix J_imu_Rt_j, J_imu_v_j, J_imu_bias_j;
        Eigen::Matrix<double, 15, 9> J_i;
        Eigen::Matrix<double, 15, 9> J_j;

        std::cout << "----------------- IMU only optimization -----------------" << std::endl;
        linearizeImuFactor(previous_navstate, current_navstate, bias);
        for (size_t i = 0; i < 8; i++)
        {

            Vec15 res_imu = evaluateIMUerrors(
                    previous_navstate,
                    current_navstate,
                    bias,
                    J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias_i, J_imu_bias_j
            );

            Vec15 r = res_imu;
            // ----------------------- increment previous state ----------------------- //


            J_i.setZero();
            J_i.leftCols(6) = J_imu_Rt_i;
            J_i.rightCols(3) = J_imu_v_i;

            Eigen::Matrix<double, 9, 1> inc_i = (J_i.transpose() * inv_covariance * J_i).ldlt().solve(
                    -J_i.transpose() * inv_covariance * r
            );
            Eigen::Matrix<double, 6, 1> inc_Rt_i = inc_i.head(6);
            Eigen::Matrix<double, 3, 1> inc_v_i = inc_i.tail(3);

//            previous_navstate = NavState(
//                    previous_navstate.pose().compose(Pose3::Expmap(inc_Rt_i)),
//                    previous_navstate.velocity() + inc_v_i
//            );

            // ----------------------- increment previous state ----------------------- //


            J_j.setZero();
            J_j.leftCols(6) = J_imu_Rt_j;
            J_j.rightCols(3) = J_imu_v_j;

            Eigen::Matrix<double, 9, 1> inc_j = (J_j.transpose() * inv_covariance * J_j).ldlt().solve(
                    -J_j.transpose() * inv_covariance * r
            );
            Eigen::Matrix<double, 6, 1> inc_Rt_j = inc_j.head(6);
            Eigen::Matrix<double, 3, 1> inc_v_j = inc_j.tail(3);

            current_navstate = NavState(
                    current_navstate.pose().compose(Pose3::Expmap(inc_Rt_j)),
                    current_navstate.velocity() + inc_v_j
            );

            std::cout << "error: " << r.transpose() << std::endl;
            std::cout << "inc_Rt_i: " << inc_Rt_i.transpose() << std::endl;
            std::cout << "inc_v_i: " << inc_v_i.transpose() << std::endl;
            std::cout << "inc_Rt_j: " << inc_Rt_j.transpose() << std::endl;
            std::cout << "inc_v_j: " << inc_v_j.transpose() << std::endl;

            std::cout << std::endl << std::endl;

            // shuffle the order of the term (t, R, a, b, v, ba, bg)
//            J_i.setZero();
//            J_i.topLeftCorner<15, 3>() = J_imu_Rt_i.block<9, 3>(0, 3);
//            J_i.block<15, 3>(0, 3) = J_imu_Rt_i.topLeftCorner<9, 3>();
//            J_i.block<9, 3>(0, 8) = J_imu_v_i.topLeftCorner<9, 3>();
//
//            J_j.setZero();
//            J_j.topLeftCorner<9, 3>() = J_imu_Rt_j.block<9, 3>(0, 3);
//            J_j.block<9, 3>(0, 3) = J_imu_Rt_j.topLeftCorner<9, 3>();
//            J_j.block<9, 3>(0, 8) = J_imu_v_j.topLeftCorner<9, 3>();
//
//
//            Mat1515 information_matrix = inv_covariance;
//
//            Eigen::Matrix<double, 15, 34> J;
//            J.leftCols(17) = J_i;
//            J.rightCols(17) = J_j;
//
//            Eigen::Matrix<double, 34, 34> H = J.transpose() * J;
//            Eigen::Matrix<double, 34, 1> b = J.transpose() * r;
//            Eigen::Matrix<double, 34, 1> inc = H.ldlt().solve(-b);
//
//            std::cout << "init inc: " << inc.transpose() << std::endl;
//
//            Vec17 inc_i = inc.head<17>();
////            Vec17 inc_j = inc.tail<17>();
//            Vec17 inc_j = (J.rightCols(17).transpose() * J.rightCols(17)).ldlt().solve(
//                    -(J.rightCols(17).transpose() * r ).tail<17>()
//            );
//
////            Mat1717 H_i = J_i.transpose() * information_matrix * J_i;
////            Vec17 b_i = J_i.transpose() * information_matrix * r;
////
////            Mat1717 H_j = J_j.transpose() * information_matrix * J_j;
////            Vec17 b_j = J_j.transpose() * information_matrix * r;
////
////            Vec17 inc_i = H_i.ldlt().solve(-b_i);
////            Vec17 inc_j = H_j.ldlt().solve(-b_j);
//
//            // the sophus se3's first 3 are translation and last 3 are rotation
//            Vec6 inc_i_se3 = inc_i.head<6>();
//            Vec6 inc_j_se3 = inc_j.head<6>();
//
//            SE3 old_pose_i(current_navstate.pose().matrix());
//            SE3 new_pose_i = old_pose_i * SE3::exp(inc_i_se3);
//            Vec3 new_velocity_i = current_navstate.velocity() + inc_i.segment<3>(8);
//
//            SE3 old_pose_j(current_navstate.pose().matrix());
//            SE3 new_pose_j = old_pose_j * SE3::exp(inc_j_se3);
//            Vec3 new_velocity_j = current_navstate.velocity() + inc_j.segment<3>(8);
//
////            previous_navstate = gtsam::NavState(
////                    gtsam::Pose3(new_pose_i.matrix()), new_velocity_i
////            );
////
//            current_navstate = gtsam::NavState(
//              gtsam::Pose3(new_pose_j.matrix()), new_velocity_j
//            );
//
//            std::cout << "J_imu_Rt_i: \n" << J_imu_Rt_i << std::endl;
//            std::cout << "J_imu_v_i: \n" << J_imu_v_i << std::endl;
//            std::cout << "J_imu_Rt_j: \n" << J_imu_Rt_j << std::endl;
//            std::cout << "J_imu_v_j: \n" << J_imu_v_j << std::endl;

//            double final_error = res_imu.transpose() * information_matrix * res_imu;
//            std::cout << "Step " << i << " error: " << r.transpose() << std::endl;
//            std::cout << "Step " << i << " final error: " << final_error << std::endl;
//            std::cout << "Step " << i << " increment_i: " << inc_i_se3.transpose() << std::endl;
//            std::cout << "Step " << i << " increment_j: " << inc_j_se3.transpose() << std::endl << std::endl;
        }
        std::cout << std::endl << std::endl << std::endl;
    }

    return predicted_pose_imu;
}