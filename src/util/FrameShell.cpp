#include <GroundTruthIterator/GroundTruthIterator.h>
#include "FrameShell.h"

Mat1515 FrameShell::getIMUcovarianceBA()
{
    PreintegratedCombinedMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedCombinedMeasurements*>(imu_preintegrated_last_kf_);
    return preint_imu->preintMeasCov();
}

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

FrameShell::~FrameShell()
{
    if (imu_factor_last_frame_)
    {
        delete imu_factor_last_frame_;
    }
}



void FrameShell::linearizeImuFactorLastKeyFrame(
        gtsam::NavState previouskf_navstate,
        gtsam::NavState current_navstate,
        gtsam::imuBias::ConstantBias previouskf_bias,
        gtsam::imuBias::ConstantBias current_bias
)
{
    if(!fh->needrelin) return;
    PreintegratedCombinedMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedCombinedMeasurements*>(imu_preintegrated_last_kf_);
    if (!imu_factor_last_kf_)
    {
        imu_factor_last_kf_ = new CombinedImuFactor(
                X(0), V(0),
                X(1), V(1),
                B(0), B(1),
                *preint_imu
        );
    }
    else
    {
        *imu_factor_last_kf_ = CombinedImuFactor(
                X(0), V(0),
                X(1), V(1),
                B(0), B(1),
                *preint_imu
        );
    }
    Values initial_values;
    initial_values.insert(X(0), previouskf_navstate.pose());
    initial_values.insert(X(1), current_navstate.pose());
    initial_values.insert(V(0), previouskf_navstate.velocity());
    initial_values.insert(V(1), current_navstate.velocity());
    initial_values.insert(B(0), previouskf_bias);
    initial_values.insert(B(1), current_bias);
    imu_factor_last_kf_->linearize(initial_values);
}


void FrameShell::linearizeImuFactorLastFrame(
        gtsam::NavState previous_navstate,
        gtsam::NavState current_navstate,
        gtsam::imuBias::ConstantBias previous_bias,
        gtsam::imuBias::ConstantBias current_bias
)
{
    PreintegratedCombinedMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedCombinedMeasurements*>(imu_preintegrated_last_frame_);

    if (!imu_factor_last_frame_)
    {
        imu_factor_last_frame_ = new CombinedImuFactor(
                X(0), V(0),
                X(1), V(1),
                B(0), B(1),
                *preint_imu
        );
    }
    else
    {
        *imu_factor_last_frame_ = CombinedImuFactor(
                X(0), V(0),
                X(1), V(1),
                B(0), B(1),
                *preint_imu
        );
    }

    Values initial_values;
    initial_values.insert(X(0), previous_navstate.pose());
    initial_values.insert(X(1), current_navstate.pose());
    initial_values.insert(V(0), previous_navstate.velocity());
    initial_values.insert(V(1), current_navstate.velocity());
    initial_values.insert(B(0), previous_bias);
    initial_values.insert(B(1), current_bias);

    imu_factor_last_frame_->linearize(initial_values);
}

Vec15 FrameShell::evaluateIMUerrorsBA(
        gtsam::NavState previous_navstate,
        gtsam::NavState current_navstate,
        gtsam::imuBias::ConstantBias previous_bias,
        gtsam::imuBias::ConstantBias current_bias,
        gtsam::Matrix &J_imu_Rt_i,
        gtsam::Matrix &J_imu_v_i,
        gtsam::Matrix &J_imu_Rt_j,
        gtsam::Matrix &J_imu_v_j,
        gtsam::Matrix &J_imu_bias_i,
        gtsam::Matrix &J_imu_bias_j
)
{
    Vec15 resreturn = imu_factor_last_kf_->evaluateError(
            previous_navstate.pose(), previous_navstate.velocity(), current_navstate.pose(), current_navstate.velocity(),
            previous_bias, current_bias,
            J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias_i, J_imu_bias_j
    );
//    std::cout<<"by gtsam: "<<std::endl;
//    std::cout<<"J_imu_Rt_i:\n"<<J_imu_Rt_i<<std::endl;
//    std::cout<<"J_imu_v_i:\n"<<J_imu_v_i<<std::endl;
//    std::cout<<"J_imu_Rt_j:\n"<<J_imu_Rt_j<<std::endl;
//    std::cout<<"J_imu_v_j:\n"<<J_imu_v_j<<std::endl;


//    Mat33 Ri = previous_navstate.pose().rotation().matrix();
//    Mat33 Rj = current_navstate.pose().rotation().matrix();
//    Vec3 Vj = current_navstate.velocity();
//    Vec3 Vi = previous_navstate.velocity();
//    Vec3 Pi = previous_navstate.pose().translation().matrix();
//    Vec3 Pj = current_navstate.pose().translation().matrix();
//    double deltaT = imu_preintegrated_last_frame_->deltaTij();
//    double deltaT2 = deltaT*deltaT;
//
//    J_imu_Rt_j.setZero();
//    J_imu_Rt_j.block<3, 3>(0, 0) = (gtsam::SO3::LogmapDerivative(resreturn.segment<3>(0))).inverse();
//    J_imu_Rt_j.block<3, 3>(3, 0) *= 0.0;
//    J_imu_Rt_j.block<3, 3>(6, 0) *= 0.0;
//
//    J_imu_Rt_j.block<3, 3>(0, 3) *= 0.0;
//    J_imu_Rt_j.block<3, 3>(3, 3) = Ri.transpose() * Rj;
//    J_imu_Rt_j.block<3, 3>(6, 3) *= 0.0;
//
//    J_imu_v_j.block<3, 3>(0, 0) *= 0.0;
//    J_imu_v_j.block<3, 3>(3, 0) *= 0.0;
//    J_imu_v_j.block<3, 3>(6, 0) *= Ri.transpose();
//
//    J_imu_Rt_i.setZero();
//    J_imu_Rt_i.block<3, 3>(0, 0) = -(gtsam::SO3::LogmapDerivative(resreturn.segment<3>(0))).inverse() * Rj.transpose() * Ri;
//    J_imu_Rt_i.block<3, 3>(3, 0) = SO3::hat(Ri.transpose() * (Pj - Pi - Vi* deltaT - 0.5 * Vec3(0,0,-9.81) * deltaT2));
//    J_imu_Rt_i.block<3, 3>(6, 0) = SO3::hat(Ri.transpose()*(Vj - Vi - Vec3(0,0,-9.81) * deltaT));
//
//    J_imu_Rt_i.block<3, 3>(0, 3) *= 0.0;
//    J_imu_Rt_i.block<3, 3>(3, 3) = - Mat33::Identity();
//    J_imu_Rt_i.block<3, 3>(6, 3) *= 0.0;
//
//    J_imu_v_i.block<3, 3>(0, 0) *= 0.0;
//    J_imu_v_i.block<3, 3>(3, 0) = - Ri.transpose() * deltaT;
//    J_imu_v_i.block<3, 3>(6, 0) = - Ri.transpose();
//    std::cout<<"by myself: "<<std::endl;
//    std::cout<<"J_imu_Rt_i:\n"<<J_imu_Rt_i<<std::endl;
//    std::cout<<"J_imu_v_i:\n"<<J_imu_v_i<<std::endl;
//    std::cout<<"J_imu_Rt_j:\n"<<J_imu_Rt_j<<std::endl;
//    std::cout<<"J_imu_v_j:\n"<<J_imu_v_j<<std::endl;
//
//
    return resreturn;
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
    Vec15 resreturn = imu_factor_last_frame_->evaluateError(
            previous_navstate.pose(), previous_navstate.velocity(), current_navstate.pose(), current_navstate.velocity(),
            initial_bias, this->bias,
            J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias_i, J_imu_bias_j
    );
//    std::cout<<"by gtsam: "<<std::endl;
//    std::cout<<"J_imu_Rt_i:\n"<<J_imu_Rt_i<<std::endl;
//    std::cout<<"J_imu_v_i:\n"<<J_imu_v_i<<std::endl;
//    std::cout<<"J_imu_Rt_j:\n"<<J_imu_Rt_j<<std::endl;
//    std::cout<<"J_imu_v_j:\n"<<J_imu_v_j<<std::endl;


//    Mat33 Ri = previous_navstate.pose().rotation().matrix();
//    Mat33 Rj = current_navstate.pose().rotation().matrix();
//    Vec3 Vj = current_navstate.velocity();
//    Vec3 Vi = previous_navstate.velocity();
//    Vec3 Pi = previous_navstate.pose().translation().matrix();
//    Vec3 Pj = current_navstate.pose().translation().matrix();
//    double deltaT = imu_preintegrated_last_frame_->deltaTij();
//    double deltaT2 = deltaT*deltaT;
//
//    J_imu_Rt_j.setZero();
//    J_imu_Rt_j.block<3, 3>(0, 0) = (gtsam::SO3::LogmapDerivative(resreturn.segment<3>(0))).inverse();
//    J_imu_Rt_j.block<3, 3>(3, 0) *= 0.0;
//    J_imu_Rt_j.block<3, 3>(6, 0) *= 0.0;
//
//    J_imu_Rt_j.block<3, 3>(0, 3) *= 0.0;
//    J_imu_Rt_j.block<3, 3>(3, 3) = Ri.transpose() * Rj;
//    J_imu_Rt_j.block<3, 3>(6, 3) *= 0.0;
//
//    J_imu_v_j.block<3, 3>(0, 0) *= 0.0;
//    J_imu_v_j.block<3, 3>(3, 0) *= 0.0;
//    J_imu_v_j.block<3, 3>(6, 0) *= Ri.transpose();
//
//    J_imu_Rt_i.setZero();
//    J_imu_Rt_i.block<3, 3>(0, 0) = -(gtsam::SO3::LogmapDerivative(resreturn.segment<3>(0))).inverse() * Rj.transpose() * Ri;
//    J_imu_Rt_i.block<3, 3>(3, 0) = SO3::hat(Ri.transpose() * (Pj - Pi - Vi* deltaT - 0.5 * Vec3(0,0,-9.81) * deltaT2));
//    J_imu_Rt_i.block<3, 3>(6, 0) = SO3::hat(Ri.transpose()*(Vj - Vi - Vec3(0,0,-9.81) * deltaT));
//
//    J_imu_Rt_i.block<3, 3>(0, 3) *= 0.0;
//    J_imu_Rt_i.block<3, 3>(3, 3) = - Mat33::Identity();
//    J_imu_Rt_i.block<3, 3>(6, 3) *= 0.0;
//
//    J_imu_v_i.block<3, 3>(0, 0) *= 0.0;
//    J_imu_v_i.block<3, 3>(3, 0) = - Ri.transpose() * deltaT;
//    J_imu_v_i.block<3, 3>(6, 0) = - Ri.transpose();
//    std::cout<<"by myself: "<<std::endl;
//    std::cout<<"J_imu_Rt_i:\n"<<J_imu_Rt_i<<std::endl;
//    std::cout<<"J_imu_v_i:\n"<<J_imu_v_i<<std::endl;
//    std::cout<<"J_imu_Rt_j:\n"<<J_imu_Rt_j<<std::endl;
//    std::cout<<"J_imu_v_j:\n"<<J_imu_v_j<<std::endl;
//
//
    return resreturn;
}

void FrameShell::updateIMUmeasurements(std::vector<dso_vi::IMUData> mvIMUSinceLastF, std::vector<dso_vi::IMUData> mvIMUSinceLastKF)
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

        if (imudata._t < last_frame->viTimestamp)
        {
            continue;
        }

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
            double interpolated_dt = (imudata._t - last_frame->viTimestamp);
            if (interpolated_dt >= 0)
            {
                dt += interpolated_dt;
            }
            else
            {
                continue;
            }
        }

        assert(dt >= 0);

//        printf("%.7f: %f\n", imudata._t, dt);

        imu_preintegrated_last_frame_->integrateMeasurement(
                rawimudata.block<3,1>(0,0),
                rawimudata.block<3,1>(3,0),
                dt
        );
    }

    for (size_t i = 0; i < mvIMUSinceLastKF.size(); i++)
    {
        dso_vi::IMUData imudata = mvIMUSinceLastKF[i];

        // if imu data is older than the key frame
        if (imudata._t < last_kf->viTimestamp)
        {
            continue;
        }

        Mat61 rawimudata;
        rawimudata << 	imudata._a(0), imudata._a(1), imudata._a(2),
                imudata._g(0), imudata._g(1), imudata._g(2);

        double dt = 0;
        // interpolate readings
        if (i == mvIMUSinceLastKF.size() - 1)
        {
            dt = viTimestamp - imudata._t;
        }
        else
        {
            dt = mvIMUSinceLastKF[i+1]._t - imudata._t;
        }

        if (i == 0)
        {
            // assuming the missing imu reading between the previous frame and first IMU
            double interpolated_dt = (imudata._t - last_kf->viTimestamp);
            if (interpolated_dt >= 0)
            {
                dt += interpolated_dt;
            }
            else
            {
                continue;
            }
        }

        assert(dt >= 0);

        imu_preintegrated_last_kf_->integrateMeasurement(
                rawimudata.block<3,1>(0,0),
                rawimudata.block<3,1>(3,0),
                dt
        );
    }

//    Matrix9 H1, H2;

//    for (PreintegrationType *&intermediate_factor: keyframe2LastFrameFactors)
//    {
//        imu_preintegrated_last_kf_->mergeWith(*intermediate_factor, &H1, &H2);
//    }
//
//    imu_preintegrated_last_kf_->mergeWith(*imu_preintegrated_last_frame_, &H1, &H2);
    assert(imu_preintegrated_last_kf_->deltaTij() > 0.0);
}


gtsam::NavState FrameShell::PredictPose(gtsam::NavState ref_pose_imu, double lastTimestamp)
{


//     //conversion from EUROC reference to our reference
//    Mat44 T_dso_euroc = ref_pose_imu.pose().matrix() * last_frame->groundtruth.pose.inverse().matrix();
//
//    std::cout << "Velocity: " << (last_frame->navstate.velocity()).norm() <<" GT: " << ( last_frame->groundtruth.velocity).norm() << std::endl;
//    std::cout << "DSO vs EuRoC rotation: \n" << T_dso_euroc.block<3,3>(0,0) << std::endl;
//
//    ref_pose_imu = gtsam::NavState(
//          ref_pose_imu.pose(),
//          T_dso_euroc.block<3,3>(0,0) * last_frame->groundtruth.velocity
//    );
//
//    ref_pose_imu = gtsam::NavState(
//            ref_pose_imu.pose(),
//            ref_pose_imu.velocity()
//    );

    last_frame->navstate = ref_pose_imu;

//    std::cout<<"last id: "<<fh->shell->last_frame->id<<std::endl;
//    std::cout<<"last_frame->id: "<<last_frame->id<<std::endl;
//    std::cout << "GT vel transformed: " << ref_pose_imu.velocity().transpose() << std::endl;
//    std::cout << "Vel initialized: " << last_frame->navstate.velocity().transpose() << std::endl;
//    std::cout << "GT Vel initialized: " << last_frame->groundtruth.velocity.transpose() << std::endl;

    gtsam::NavState predicted_pose_imu = imu_preintegrated_last_frame_->predict(ref_pose_imu, bias);

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

    return predicted_pose_imu;
}