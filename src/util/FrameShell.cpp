#include "FrameShell.h"

Mat99 FrameShell::getIMUcovariance()
{
    PreintegratedImuMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedImuMeasurements*>(imu_preintegrated_last_frame_);
    return preint_imu->preintMeasCov();
}

Vec3 FrameShell::TWB(Matrix44 Tbc){
    Mat44 Twc;
    Mat44 Tcb;
    Mat44 Twb;
    Twc.setIdentity();
    Tcb.setIdentity();
    Twc.block<3,3>(0,0) = RWC();
    Twc.block<3,1>(0,3) = TWC();
    Tcb.block<3,3>(0,0) = Tbc.block<3,3>(0,0).transpose();
    Tcb.block<3,1>(0,3) = - Tbc.block<3,3>(0,0).transpose() * Tbc.block<3,1>(0,3);
    Twb = Twc * Tcb;
    return Twb.block<3,1>(0,3);
}

Vector9 FrameShell::evaluateIMUerrors(
        SE3 initial_cam_2_world,
        Vec3 initial_velocity,
        SE3 final_cam_2_world,
        Vec3 final_velocity,
        gtsam::imuBias::ConstantBias bias,
        Mat44 Tbc,
        gtsam::Matrix &J_imu_Rt_i,
        gtsam::Matrix &J_imu_v_i,
        gtsam::Matrix &J_imu_Rt_j,
        gtsam::Matrix &J_imu_v_j,
        gtsam::Matrix &J_imu_bias
)
{
    PreintegratedImuMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedImuMeasurements*>(imu_preintegrated_last_frame_);

    ImuFactor imu_factor(X(0), V(0),
                         X(1), V(1),
                         B(0),
                         *preint_imu);

    Values initial_values;
    initial_values.insert(X(0), gtsam::Pose3(initial_cam_2_world.matrix()));
    initial_values.insert(V(0), initial_velocity);
    initial_values.insert(B(0), bias);
    initial_values.insert(X(1), gtsam::Pose3(final_cam_2_world.matrix()));
    initial_values.insert(V(1), final_velocity);

    imu_factor.linearize(initial_values);

    // relative pose wrt IMU
    gtsam::Pose3 initial_imu_2_world(dso_vi::IMUData::convertCamFrame2IMUFrame(
            initial_cam_2_world.matrix(), Tbc
    ));

    gtsam::Pose3 final_imu_2_world(dso_vi::IMUData::convertCamFrame2IMUFrame(
            final_cam_2_world.matrix(), Tbc
    ));

    // temporary stuffs
//	gtsam::Pose3 relative_pose = initial_imu_2_world.inverse().compose(final_imu_2_world);
//
//	return imu_factor.evaluateError(
//			gtsam::Pose3(), initial_velocity, relative_pose, final_velocity, bias,
//			J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias
//	);

    return imu_factor.evaluateError(
            initial_imu_2_world, initial_velocity, final_imu_2_world, final_velocity, bias,
            J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias
    );
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


SE3 FrameShell::PredictPose(SE3 lastPose, Vec3 lastVelocity, double lastTimestamp, Mat44 Tbc)
{
    gtsam::NavState ref_pose_imu = gtsam::NavState(
            gtsam::Pose3( dso_vi::IMUData::convertCamFrame2IMUFrame(lastPose.matrix(), Tbc) ),
            lastVelocity
    );
    gtsam::NavState predicted_pose_imu = imu_preintegrated_last_frame_->predict(ref_pose_imu, bias);

    Mat44 mat_pose_imu = predicted_pose_imu.pose().matrix();
    if (
            !std::isfinite(predicted_pose_imu.pose().translation().x()) ||
            !std::isfinite(predicted_pose_imu.pose().translation().y()) ||
            !std::isfinite(predicted_pose_imu.pose().translation().z())
            )
    {
        if (!setting_debugout_runquiet)
        {
            std::cout << "IMU prediction nan for translation!!!" << std::endl;
        }
        mat_pose_imu.block<3, 1>(0, 3) = Vec3::Zero();
    }

//    Mat44 relative_pose_imu = ref_pose_imu.pose().inverse().compose(predicted_pose_imu.pose()).matrix();
//    Mat44 groundtruth_pose = last_frame->groundtruth.pose.inverse().compose(groundtruth.pose).matrix();
//    Mat44 error = groundtruth_pose.inverse() * relative_pose_imu;
//    Quaternion error_quat = SO3(error.block<3, 3>(0, 0)).unit_quaternion();
//    std::cout << "Tbc \n" << Tbc << std::endl;
//    std::cout << "Error: "
//              << error_quat.x() << ", "
//              << error_quat.y() << ", "
//              << error_quat.z() << ", "
//              << std::endl;

    Mat44 predicted_pose_camera = dso_vi::IMUData::convertIMUFrame2CamFrame(mat_pose_imu, Tbc);
    return SE3(predicted_pose_camera);
}

void FrameShell::setNavstate(Mat33 Rs,Vec3 Ps,Vec3 Vs)
{
    Mat44 w2c;
    w2c.setIdentity();
    w2c.block<3,3>(0,0) = Rs;
    w2c.block<3,1>(0,3) = Ps;
    SE3 worldTocamnew(w2c);
    camToWorld = worldTocamnew.inverse();
    velocity = Vs;
}