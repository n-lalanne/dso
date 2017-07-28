/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "util/NumType.h"
#include "algorithm"
#include "util/settings.h"
#include "FullSystem/HessianBlocks.h"

// Need to add IMU data for each frame
#include "IMU/imudata.h"
#include "GroundTruthIterator/GroundTruthIterator.h"

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

using namespace dso;
using namespace gtsam;

namespace dso
{


class FrameShell
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int id; 			// INTERNAL ID, starting at zero.
	int incoming_id;	// ID passed into DSO
	double timestamp;		// timestamp passed into DSO.
	double viTimestamp;


	// set once after tracking
	SE3 camToTrackingRef;
	FrameShell* trackingRef;

	// constantly adapted.
	SE3 camToWorld;				// Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
	SE3 imuToWorld;
	AffLight aff_g2l;
	bool poseValid;

	// statisitcs
	int statistics_outlierResOnThis;
	int statistics_goodResOnThis;
	int marginalizedAt;
	double movedByOpt;

	//===========================================IMU data ==========================================================
	// data for imu integration
	PreintegrationType *imu_preintegrated_last_frame_;
	PreintegrationType *imu_preintegrated_last_kf_;

	// Predicted pose/biases ;
	gtsam::Velocity3 velocity;
	gtsam::NavState navstate;

	//SE3 prop_pose;

	// the pose of the frame
	// Pose3 pose;
	// Vector3 velocity;

	imuBias::ConstantBias bias;

	// groundtruth measurement
	dso_vi::GroundTruthIterator::ground_truth_measurement_t groundtruth;

	// last keyframe
	FrameShell * last_kf;
	FrameShell * last_frame;

	// corresponding frame hessian
	FrameHessian * fh;

	inline FrameShell(Vec3 acceleroBias, Vec3 gyroBias)
	{
		id=0;
		poseValid=true;
		camToWorld = SE3();
		timestamp=0;
		marginalizedAt=-1;
		movedByOpt=0;
		statistics_outlierResOnThis=statistics_goodResOnThis=0;
		trackingRef=0;
		camToTrackingRef = SE3();

//		gtsam::Vector3 gyroBias(-0.002153, 0.020744, 0.075806); // (0.0,0.0,0.0); //
//		gtsam::Vector3 acceleroBias(-0.013337, 0.103464, 0.093086);

		gtsam::imuBias::ConstantBias biasPrior(acceleroBias, gyroBias);

		bias = biasPrior;

        imu_preintegrated_last_frame_ = new PreintegratedCombinedMeasurements(dso_vi::getIMUParams(), bias);
		imu_preintegrated_last_kf_ = new PreintegratedCombinedMeasurements(dso_vi::getIMUParams(), bias);

	}

	//==========================================IMU related methods==================================================
	/**
	 *
	 * @return transformation from current frame to last frame
	 */
	SE3 PredictPose(SE3 lastPose, Vec3 lastVelocity, double lastTimestamp, Mat44 Tbc);

	Mat99 getIMUcovariance();
	/**
	 *
	 * @param J_imu_Rt_i, J_imu_v_i, J_imu_Rt_j, J_imu_v_j, J_imu_bias: output jacobians
	 * @return IMU residuals (9x1 vector)
	 */
	Vector9 evaluateIMUerrors(
			SE3 initial_cam_2_world,
			Vec3 initial_velocity,
			SE3 final_cam_2_world,
			Vec3 final_velocity,
			gtsam::imuBias::ConstantBias initial_bias,
			Mat44 Tbc,
			gtsam::Matrix &J_imu_Rt_i,
			gtsam::Matrix &J_imu_v_i,
			gtsam::Matrix &J_imu_Rt_j,
			gtsam::Matrix &J_imu_v_j,
			gtsam::Matrix &J_imu_bias_i,
            gtsam::Matrix &J_imu_bias_j
	);

	/**
	 *
	 * @param mvIMUSinceLastKF
	 * @param lastTimestamp timestamp of the last frame from which we want the IMU factor to be (for interpolation)
	 */
	void updateIMUmeasurements(std::vector<dso_vi::IMUData> mvIMUSinceLastF);

	inline Mat33 RWC()
	{
		return camToWorld.matrix().block<3,3>(0,0);
	}

	inline Vec3 TWC()
	{
		return camToWorld.matrix().block<3,1>(0,3);
	}

	inline Mat33 RCW()
	{
		return RWC().transpose();
	}

	inline Vec3 TCW()
	{
		return camToWorld.matrix().inverse().block<3,1>(0,3);
	}

	inline Mat33 RBW(Matrix44 Tbc)
	{
		return Tbc.block<3,3>(0,0) * RCW();
	}
    inline Mat33 RWB(Matrix44 Tbc){
        return RBW(Tbc).transpose();
    }
    inline Vec3 TBW(Matrix44 Tbc){
        return (Tbc * camToWorld.matrix()).block<3,1>(0,3);
    }
    Vec3 TWB(Matrix44 Tbc);

	void setNavstate(Mat33 Rs,Vec3 Ps,Vec3 Vs);

};


}

