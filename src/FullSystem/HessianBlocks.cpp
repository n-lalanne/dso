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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0;


}


void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}


void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;


	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}


void FrameHessian::makeImages(float* color, CalibHessian* HCalib, std::vector<dso_vi::IMUData>& vimuData)
{

	for(int i=0;i<pyrLevelsUsed;i++)
	{
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	dI = dIp[0];


	// make d0
	int w=wG[0];
	int h=hG[0];
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl];
		Eigen::Vector3f* dI_l = dIp[lvl];

		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];



			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy;

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

Mat99 FrameHessian::getIMUcovariance()
{
	PreintegratedImuMeasurements *preint_imu = dynamic_cast<gtsam::PreintegratedImuMeasurements*>(imu_preintegrated_last_frame_);
	return preint_imu->preintMeasCov().inverse();
}

Vector9 FrameHessian::evaluateIMUerrors(
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

void FrameHessian::updateIMUmeasurements(std::vector<dso_vi::IMUData> mvIMUSinceLastF, double lastTimestamp)
{
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
			dt = timestamp - imudata._t;
		}
		else
		{
			dt = mvIMUSinceLastF[i+1]._t - imudata._t;
		}

		if (i == 0)
		{
			// assuming the missing imu reading between the previous frame and first IMU
			dt += (imudata._t - lastTimestamp);
		}

		assert(dt >= 0);

		imu_preintegrated_last_frame_->integrateMeasurement(
				rawimudata.block<3,1>(0,0),
				rawimudata.block<3,1>(3,0),
				dt
		);
	}
}


SE3 FrameHessian::PredictPose(SE3 lastPose, Vec3 lastVelocity, double lastTimestamp, Mat44 Tbc)
{
	gtsam::NavState ref_pose_imu = gtsam::NavState(
			gtsam::Pose3( dso_vi::IMUData::convertCamFrame2IMUFrame(lastPose.matrix(), Tbc) ),
			lastVelocity
	);
	gtsam::NavState predicted_pose_imu = imu_preintegrated_last_frame_->predict(ref_pose_imu, bias1);

	Mat44 mat_pose_imu = predicted_pose_imu.pose().matrix();
	if
	(
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

	Mat44 predicted_pose_camera = dso_vi::IMUData::convertIMUFrame2CamFrame(mat_pose_imu, Tbc);
	return SE3(predicted_pose_camera);
}

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;

	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;


	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

