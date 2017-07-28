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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"


#include <cmath>
#include <GroundTruthIterator/GroundTruthIterator.h>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;

Mat33 ypr2R(const Vec3 &ypr)
{

	double y = ypr(0) / 180.0 * M_PI;
	double p = ypr(1) / 180.0 * M_PI;
	double r = ypr(2) / 180.0 * M_PI;

	Mat33 Rz;
	Rz << cos(y), -sin(y), 0,
			sin(y), cos(y), 0,
			0, 0, 1;

	Mat33 Ry;
	Ry << cos(p), 0., sin(p),
			0., 1., 0.,
			-sin(p), 0., cos(p);

	Mat33 Rx;
	Rx << 1., 0., 0.,
			0., cos(r), -sin(r),
			0., sin(r), cos(r);

	return Rz * Ry * Rx;
}

Vec3 R2ypr(const Eigen::Matrix3d &R)
{
	Eigen::Vector3d n = R.col(0);
	Eigen::Vector3d o = R.col(1);
	Eigen::Vector3d a = R.col(2);

	Eigen::Vector3d ypr(3);
	double y = atan2(n(1), n(0));
	double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
	double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
	ypr(0) = y;
	ypr(1) = p;
	ypr(2) = r;

	return ypr / M_PI * 180.0;
}

Mat33 g2R(const Eigen::Vector3d &g)
{
	Eigen::Matrix3d R0;
	Eigen::Vector3d ng1 = g.normalized();
	Eigen::Vector3d ng2{0, 0, 1.0};
	R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
	double yaw = R2ypr(R0).x();
	R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
	// R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
	return R0;
}

FullSystem::FullSystem()
{
	dso_vi::initializeIMUParams();
	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0], this);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0], this);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;

    isLocalBADone = true;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}


Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

	for(IOWrap::Output3DWrapper* ow : outputWrapper)
		ow->pushLiveFrame(fh);



	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	SE3 slast_2_sprelast;
	SE3 lastF_2_slast;
	SE3 const_vel_lastF_2_fh;
	SE3 fh_2_slast;
	SE3 lastF_2_world;
	SE3 slast_2_world;
	Vec3 slast_velocity;

	if(allFrameHistory.size() == 2)
	{
		for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) lastF_2_fh_tries.push_back(SE3());
	}
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];

		// get last delta-movement.
		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.push_back(SE3());
		}
		else
        {
            double slast_timestamp;
            {    // lock on global pose consistency!
                boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
                lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
                aff_last_2_l = slast->aff_g2l;
                lastF_2_world = lastF->shell->camToWorld;
                slast_2_world = slast->camToWorld;
                slast_velocity = slast->velocity;
                slast_timestamp = slast->viTimestamp;
            }

            fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.
            const_vel_lastF_2_fh = fh_2_slast.inverse() * lastF_2_slast;


            SE3 groundtruth_lastF_2_fh(dso_vi::IMUData::convertIMUFrame2CamFrame(
                    fh->shell->groundtruth.pose.inverse().compose(lastF->shell->groundtruth.pose).matrix(), getTbc()
            ));

            //just use the initial pose from IMU
            //when we determine the last key frame, we will propagate the pose by using the preintegration measurement and the pose of the last key frame
            SE3 prop_fh_2_world = fh->shell->PredictPose(slast_2_world, slast_velocity, slast_timestamp, getTbc());
            SE3 prop_lastF_2_fh_r = prop_fh_2_world.inverse() * lastF_2_world;
            SE3 prop_slast_2_fh = prop_fh_2_world.inverse() * slast_2_world;
            SE3 prop_lastF_2_slast = slast_2_world.inverse() * lastF_2_world;
            if (false) {
                lastF_2_fh_tries.push_back(prop_lastF_2_fh_r);
                lastF_2_fh_tries.push_back(
                        prop_slast_2_fh * prop_slast_2_fh *  prop_lastF_2_slast);    // assume double motion (frame skipped)
                lastF_2_fh_tries.push_back(
                        SE3::exp(prop_slast_2_fh.log() * 0.5) *
                                prop_lastF_2_slast); // assume half motion.
                lastF_2_fh_tries.push_back(prop_lastF_2_slast); // assume zero motion.
                lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

                for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh_r *
                                               SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                }

            }
            else
            {
                // get the translation independent of current frame estimated rotation
                Vec3 const_vel_translation = const_vel_lastF_2_fh.inverse().matrix().block<3, 1>(0, 3);
                SE3 prop_lastF_2_fh(
                        prop_lastF_2_fh_r.rotationMatrix(),
                        -prop_lastF_2_fh_r.rotationMatrix() * const_vel_translation
                        // Vec3::Zero()
                        // const_vel_lastF_2_fh.matrix().block<3, 1>(0, 3)
                );

                lastF_2_fh_tries.push_back(prop_lastF_2_fh);

//			std::cout << "KF: " << coarseTracker->lastRef->frameID << " Fh: " << fh->frameID << std::endl;
//			std::cout << "IMUs: " << mvIMUSinceLastKF.size() << std::endl;
//			std::cout << "Const: \n" << const_vel_lastF_2_fh.matrix() << std::endl;
//			std::cout << "Pred: \n" << prop_pose.matrix() << std::endl;
				std::cout<<"slast->id = "<<slast->id<<"\n pose:\n"<< slast->camToWorld.matrix() <<std::endl;
				std::cout<<"prelast->id = "<< sprelast->id<<"\n pose:\n"<< sprelast->camToWorld.matrix() <<std::endl;
				std::cout<<"KF->id = "<<lastF->shell->id << "\n pose:\n" << lastF->shell->camToWorld.matrix() << std::endl;

				std::cout<< " GT/prediction = \n"<<groundtruth_lastF_2_fh.matrix() * prop_lastF_2_fh.inverse().matrix()<<std::endl;
                Vec3 groundtruth_translation = groundtruth_lastF_2_fh.inverse().matrix().block<3, 1>(0, 3);
                SE3 groundtruth_slast_2_sprelast(dso_vi::IMUData::convertIMUFrame2CamFrame(
                        sprelast->groundtruth.pose.inverse().compose(slast->groundtruth.pose).matrix(),
                        getTbc()
                ));

				if (IMUinitialized)
				{
					// check the translation prediction the translation
					SE3 estimatedRelativePose = slast->camToWorld.inverse() * prop_fh_2_world;
					SE3 groundtruthRelativePose(dso_vi::IMUData::convertIMUFrame2CamFrame(
						slast->groundtruth.pose.inverse().compose(fh->shell->groundtruth.pose).matrix(),
						getTbc()
					));

					std::cout << "Estimated translation: " << estimatedRelativePose.translation().transpose() << std::endl;
					std::cout << "Groundtruth translation: " << groundtruthRelativePose.translation().transpose() << std::endl;

				}
//                std::cout << "Groundtruth translation: " << groundtruth_slast_2_sprelast.translation() << std::endl;
//                std::cout << "DSO translation: " << slast_2_sprelast.translation() << std::endl;

                double translation_direction_error = acos(
                        slast_2_sprelast.translation().normalized().dot(groundtruth_slast_2_sprelast.translation().normalized())
                ) * 180 / M_PI;

                std::cout << "Scale: "
                          << groundtruth_slast_2_sprelast.translation().norm() / slast_2_sprelast.translation().norm()
                          << " Translation dir error: " << translation_direction_error
//                          << groundtruth_slast_2_sprelast.translation()(0) / slast_2_sprelast.translation()(0) << ", "
//                          << groundtruth_slast_2_sprelast.translation()(1) / slast_2_sprelast.translation()(1) << ", "
//                          << groundtruth_slast_2_sprelast.translation()(2) / slast_2_sprelast.translation()(2)
                          << std::endl;

                // relative rotation wrt to KF compared to groundtruth
//			SE3 rotation_error = groundtruth_lastF_2_fh.inverse() * prop_lastF_2_fh;
//			Quaternion quaternion_error = rotation_error.unit_quaternion();
//
//			std::cout << "Rotation KF error imu: " << quaternion_error.x() << ", "
//					  << quaternion_error.y() << ", "
//					  << quaternion_error.z()
//					  << std::endl;
//
//			rotation_error = groundtruth_lastF_2_fh.inverse() * const_vel_lastF_2_fh;
//			quaternion_error = rotation_error.unit_quaternion();
//			std::cout << "Rotation kF error cvm: " << quaternion_error.x() << ", "
//					  << quaternion_error.y() << ", "
//					  << quaternion_error.z()
//					  << std::endl;
//
//			// relative rotation wrt to previous frame compared to groundtruth
//			SE3 groundtruth_slast_2_fh(dso_vi::IMUData::convertIMUFrame2CamFrame(
//					fh->shell->groundtruth.pose.inverse().compose(slast->groundtruth.pose).matrix(), getTbc()
//			));
//
//			rotation_error = groundtruth_slast_2_fh.inverse() * prop_slast_2_fh;
//			quaternion_error = rotation_error.unit_quaternion();
//			std::cout << "Rotation error imu: " << quaternion_error.x() << ", "
//					  << quaternion_error.y() << ", "
//					  << quaternion_error.z()
//					  << std::endl;
//
//			rotation_error = groundtruth_slast_2_fh.inverse() * fh_2_slast.inverse();
//			quaternion_error = rotation_error.unit_quaternion();
//			std::cout << "Rotation error cvm : " << quaternion_error.x() << ", "
//					  << quaternion_error.y() << ", "
//					  << quaternion_error.z()
//					  << std::endl;

                lastF_2_fh_tries.push_back(
                        prop_slast_2_fh * prop_slast_2_fh * lastF_2_slast);    // assume double motion (frame skipped)
                lastF_2_fh_tries.push_back(
                        SE3::exp(prop_slast_2_fh.inverse().log() * 0.5).inverse() *
                        lastF_2_slast); // assume half motion.
                lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
                lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

                for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0,
                                                                                                                  0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0,
                                                                                                                   0)));            // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                                                                     Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                    lastF_2_fh_tries.push_back(prop_lastF_2_fh *
                                               SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                                                   Vec3(0, 0, 0)));    // assume constant motion.
                }

//			lastF_2_fh_tries.push_back(const_vel_lastF_2_fh);    // assume constant motion.
//			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() *
//									   lastF_2_slast);    // assume double motion (frame skipped)
//			lastF_2_fh_tries.push_back(
//					SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
//			lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
//			lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.
//
////			 just try a TON of different initializations (all rotations). In the end,
////			 if they don't work they will only be tried on the coarsest level, which is super fast anyway.
////			 also, if tracking rails here we loose, so we really, really want to avoid that.
//
//
//			for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
//											   Vec3(0, 0, 0)));            // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
//										   SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
//											   Vec3(0, 0, 0)));    // assume constant motion.
//			}
            }
        }

		if(!lastF->shell->poseValid )
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		SE3 slast_2_fh_this = lastF_2_fh_this * lastF_2_slast.inverse();
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, slast_2_fh_this,  aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
				   i,
				   i, pyrLevelsUsed-1,
				   aff_g2l_this.a,aff_g2l_this.b,
				   achievedRes[0],
				   achievedRes[1],
				   achievedRes[2],
				   achievedRes[3],
				   achievedRes[4],
				   coarseTracker->lastResiduals[0],
				   coarseTracker->lastResiduals[1],
				   coarseTracker->lastResiduals[2],
				   coarseTracker->lastResiduals[3],
				   coarseTracker->lastResiduals[4]);
		}


		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


		if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
			break;

	}

	if(!haveOneGood)
	{
		printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

	std::cout << "After tracking pose: \n" << fh->shell->camToWorld.matrix() << std::endl;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

	if(!setting_debugout_runquiet)
		printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
							 << fh->shell->id << " "
							 << fh->shell->timestamp << " "
							 << fh->ab_exposure << " "
							 << fh->shell->camToWorld.log().transpose() << " "
							 << aff_g2l.a << " "
							 << aff_g2l.b << " "
							 << achievedRes[0] << " "
							 << tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}


void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{

	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
//				immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}


	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}

bool FullSystem::SolveScale(Vec3 &g, Eigen::VectorXd &x)
{
//	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	Eigen::Vector3d G{0.0, 0.0, 9.8};
	int n_state = WINDOW_SIZE * 3 + 3 + 1;

	Eigen::MatrixXd A{n_state, n_state};
	A.setZero();
	Eigen::VectorXd b{n_state};
	b.setZero();
	int firstindex = allKeyFramesHistory.size()-WINDOW_SIZE;

	for (int i = 0; i < WINDOW_SIZE - 1; i++)
	{
		FrameShell *frame_i = allKeyFramesHistory[i+firstindex];
		FrameShell *frame_j = allKeyFramesHistory[i+firstindex+1];

		Eigen::MatrixXd tmp_A(6, 10);
		tmp_A.setZero();
		Eigen::VectorXd tmp_b(6);
		tmp_b.setZero();

		double dt = frame_j->imu_preintegrated_last_kf_->deltaTij();
		std::cout << "Frame ID: " << frame_j->id << " to " << frame_i->id <<", estimated dt: "<<dt<<" groundtruth dt: "<<frame_j->viTimestamp-frame_i->viTimestamp<<std::endl;

		tmp_A.block<3, 3>(0, 0) = -dt * Mat33::Identity();
		tmp_A.block<3, 3>(0, 6) = frame_i->RBW(getTbc()) * dt * dt / 2 * Mat33::Identity();
		tmp_A.block<3, 1>(0, 9) = frame_i->RBW(getTbc()) * ( frame_j->TWC() - frame_i->TWC()) / 100.0;
		tmp_b.block<3, 1>(0, 0) = frame_j->imu_preintegrated_last_kf_->deltaPij() + frame_i->RBW(getTbc()) * frame_j->RBW(getTbc()).transpose() * getTbc().block<3,1>(0,3) - getTbc().block<3,1>(0,3);
		//cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
		tmp_A.block<3, 3>(3, 0) = -Mat33::Identity();
		tmp_A.block<3, 3>(3, 3) = frame_i->RBW(getTbc()) * frame_j->RBW(getTbc()).transpose();
		tmp_A.block<3, 3>(3, 6) = frame_i->RBW(getTbc()) * dt * Mat33::Identity();
		tmp_b.block<3, 1>(3, 0) = frame_j->imu_preintegrated_last_kf_->deltaVij();

//		Mat44 fi_TIB = frame_i->groundtruth.pose.matrix();
//		Mat44 fi_TWC = dso_vi::IMUData::convertIMUFrame2CamFrame(fi_TIB, getTbc());
//		Mat44 fi_TWB = fi_TWC * getTbc().inverse();
//		Mat33 fi_RBW = fi_TWB.block<3,3>(0,0).transpose();
//		Vec3 fi_tWC = fi_TWC.block<3,1>(0,3);
//
//		Mat44 fj_TIB = frame_j->groundtruth.pose.matrix();
//		Mat44 fj_TWC = dso_vi::IMUData::convertIMUFrame2CamFrame(fj_TIB, getTbc());
//		Mat44 fj_TWB = fj_TWC * getTbc().inverse();
//		Mat33 fj_RBW = fj_TWB.block<3,3>(0,0).transpose();
//		Vec3 fj_tWC = fj_TWC.block<3,1>(0,3);
//
//
//		tmp_A.block<3, 3>(0, 0) = -dt * Mat33::Identity();
//		tmp_A.block<3, 3>(0, 6) = fi_RBW * dt * dt / 2 * Mat33::Identity();
//		tmp_A.block<3, 1>(0, 9) = fi_RBW * ( fj_tWC - fi_tWC) / 100.0;
//		tmp_b.block<3, 1>(0, 0) = frame_j->imu_preintegrated_last_kf_->deltaPij() + fi_RBW * fj_RBW.transpose() * getTbc().block<3,1>(0,3) - getTbc().block<3,1>(0,3);
//		//cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
//		tmp_A.block<3, 3>(3, 0) = -Mat33::Identity();
//		tmp_A.block<3, 3>(3, 3) = fi_RBW * fj_RBW.transpose();
//		tmp_A.block<3, 3>(3, 6) = fi_RBW * dt * Mat33::Identity();
//		tmp_b.block<3, 1>(3, 0) = frame_j->imu_preintegrated_last_kf_->deltaVij();

		//cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;
//        std::cout<<"dt:\n"<<dt<<"\ntem_A: \n"<<tmp_A<<"\ntem_b:\n"<<tmp_b<<std::endl;
		Mat66 cov_inv;
		//cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
		//MatrixXd cov_inv = cov.inverse();
		cov_inv.setIdentity();

		Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
		Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

		A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
		b.segment<6>(i * 3) += r_b.head<6>();

		A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
		b.tail<4>() += r_b.tail<4>();

		A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
		A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
	}
//    A = A * 1000.0;
//    b = b * 1000.0;
	x = A.ldlt().solve(b);
	double s = x(n_state - 1) / 100.0;
	std::cout<<"estimated scale:"<<s<<std::endl;
	g = x.segment<3>(n_state - 4);
	std::cout<<" result g.norm:" << g.norm() << " g:" << g.transpose()<<std::endl;
	//" result g     " << g.norm() << " " << g.transpose());
	if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
	{
		return false;
	}

	// compute the gravity direction from groundtruth
	Mat33 R_b0_bx = dso_vi::IMUData::convertCamFrame2IMUFrame(allKeyFramesHistory[0]->camToWorld.matrix(), getTbc()).block<3,3>(0, 0);
	Mat33 R_I_bx = allKeyFramesHistory[0]->groundtruth.pose.rotation().matrix();
	Mat33 R_I_b0 =  R_I_bx * R_b0_bx.transpose();

	Vec3 g_b0 = getRbc() * g;
	Vec3 g_I = R_I_b0 * g_b0;

	std::cout << "Before refinement Inertial g: " << g_I << std::endl;

	for (int i = 0; i < WINDOW_SIZE-1; i++)
	{
		Vec3 velocity = x.segment<3>(i * 3);
		FrameShell *frame_i = allKeyFramesHistory[i+firstindex];
		std::cout << "GT v: " << frame_i->groundtruth.velocity.norm() << " calc v: " << velocity.norm() << std::endl;
	}



	RefineGravity(g, x);
	s = (x.tail<1>())(0) / 100.0;
	(x.tail<1>())(0) = s;

	for (int i = 0; i < WINDOW_SIZE-1; i++)
	{
		Vec3 velocity = x.segment<3>(i * 3);
		FrameShell *frame_i = allKeyFramesHistory[i+firstindex];
		std::cout << "GT v: " << frame_i->groundtruth.velocity.norm() << " calc v: " << velocity.norm() << std::endl;
	}

	//ROS_DEBUG("refine estimated scale: %f", s);
	//ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
	if(s > 0.0 )
	{
		//ROS_DEBUG("initial succ!");
	}
	else
	{
		//ROS_DEBUG("initial fail");
		return false;
	}
	return true;
}

MatXX TangentBasis(Vec3 &g0)
{
	Vec3 b, c;
	Vec3 a = g0.normalized();
	Vec3 tmp(0, 0, 1);
	if(a == tmp)
		tmp << 1, 0, 0;
	b = (tmp - a * (a.transpose() * tmp)).normalized();
	c = a.cross(b);
	MatXX bc(3, 2);
	bc.block<3, 1>(0, 0) = b;
	bc.block<3, 1>(0, 1) = c;
	return bc;
}

void FullSystem::RefineGravity(Vec3 &g, VecX &x)
{
	Eigen::Vector3d G{0.0, 0.0, 9.8};
	Vec3 g0 = g.normalized() * G.norm();
	//VectorXd x;
	int n_state = WINDOW_SIZE * 3 + 2 + 1;

	Eigen::MatrixXd A{n_state, n_state};
	A.setZero();
	Eigen::VectorXd b{n_state};
	b.setZero();
	Matrix69 tmp_A;
	Vec6 tmp_b;

	for(int k = 0; k < 4; k++)
	{
		Eigen::MatrixXd lxly(3, 2);
		lxly = TangentBasis(g0);

		int firstindex = allKeyFramesHistory.size()-WINDOW_SIZE;
		for (int i = 0; i < WINDOW_SIZE-1; i++)
		{

			FrameShell *frame_i = allKeyFramesHistory[i+firstindex];
			FrameShell *frame_j = allKeyFramesHistory[i+firstindex+1];

			tmp_A.setZero();

			tmp_b.setZero();

			double dt = frame_j->imu_preintegrated_last_kf_->deltaTij();

			tmp_A.block<3, 3>(0, 0) = -dt * Mat33::Identity();
			tmp_A.block<3, 2>(0, 6) = frame_i->RBW(getTbc()) * dt * dt / 2 * Mat33::Identity()* lxly;
			tmp_A.block<3, 1>(0, 8) = frame_i->RBW(getTbc()) * ( frame_j->TWC() - frame_i->TWC()) / 100.0;
			tmp_b.block<3, 1>(0, 0) = frame_j->imu_preintegrated_last_kf_->deltaPij() + frame_i->RBW(getTbc()) * frame_j->RBW(getTbc()).transpose() * getTbc().block<3,1>(0,3) - getTbc().block<3,1>(0,3) - frame_i->RBW(getTbc()) * dt * dt / 2 * g0;
			//cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
			tmp_A.block<3, 3>(3, 0) = -Mat33::Identity();
			tmp_A.block<3, 3>(3, 3) = frame_i->RBW(getTbc()) * frame_j->RBW(getTbc()).transpose();
			tmp_A.block<3, 2>(3, 6) = frame_i->RBW(getTbc()) * dt * Mat33::Identity()* lxly;
			tmp_b.block<3, 1>(3, 0) = frame_j->imu_preintegrated_last_kf_->deltaVij() - frame_i->RBW(getTbc()) * dt * Mat33::Identity() * g0;


			Mat66 cov_inv;
			//cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
			//MatrixXd cov_inv = cov.inverse();
			cov_inv.setIdentity();

			Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
			Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

			A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
			b.segment<6>(i * 3) += r_b.head<6>();

			A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
			b.tail<3>() += r_b.tail<3>();

			A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
			A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();

//			std::cout << "A: \n" << A << std::endl;
//			std::cout << "rA: \n" << r_A << std::endl;
//			std::cout << "i: " << i << " n_state: " << n_state << std::endl;
//			std::cout << "A_block: \n" << A.block<3, 6>(n_state - 3, i * 3) << std::endl;
//			std::cout << "rA_block: \n" << r_A.bottomLeftCorner<3, 6>() << std::endl;

//			std::cout<<"goes here!"<<std::endl;
		}
		A = A * 1000.0;
		b = b * 1000.0;
		x = A.ldlt().solve(b);
		VecX dg = x.segment<2>(n_state - 3);
		g0 = (g0 + lxly * dg).normalized() * G.norm();
		//double s = x(n_state - 1);

		std::cout<<"Refined g: \n"<< g0<<std::endl;

		// compute the gravity direction from groundtruth
		Mat33 R_b0_bx = dso_vi::IMUData::convertCamFrame2IMUFrame(allKeyFramesHistory[0]->camToWorld.matrix(), getTbc()).block<3,3>(0, 0);
		Mat33 R_I_bx = allKeyFramesHistory[0]->groundtruth.pose.rotation().matrix();
		Mat33 R_I_b0 =  R_I_bx * R_b0_bx.transpose();

		Vec3 g_b0 = getRbc() * g0;
		Vec3 g_I = R_I_b0 * g_b0;

		std::cout << "Final Inertial gravity: " << g_I << std::endl;
	}
	g = g0;
	gravity = g;
//    std::cout<<"Refined g: \n"<< g<<std::endl;
//
//	// compute the gravity direction from groundtruth
//    Mat33 R_b0_bx = dso_vi::IMUData::convertCamFrame2IMUFrame(allKeyFramesHistory[0]->camToWorld.matrix(), getTbc()).block<3,3>(0, 0);
//    Mat33 R_I_bx = allKeyFramesHistory[0]->groundtruth.pose.rotation().matrix();
//    Mat33 R_I_b0 =  R_I_bx * R_b0_bx.transpose();
//
//    Vec3 g_b0 = getRbc() * g;
//    Vec3 g_I = R_I_b0 * g_b0;
//
//    std::cout << "Final Inertial gravity: " << g_I << std::endl;

}

void FullSystem::solveGyroscopeBiasbyGTSAM()
{
	for (int iter_idx = 0; iter_idx < 1; iter_idx++)
	{
		Mat33 A;
		Vec3 b;
		Vec3 delta_bg;
		A.setZero();
		b.setZero();
		for (int index_i = allKeyFramesHistory.size() - 1;
			 index_i >= allKeyFramesHistory.size() - WINDOW_SIZE + 1; index_i--)
		{
			Mat33 resR;
			FrameShell *frame_i = allKeyFramesHistory[index_i - 1];
			FrameShell *frame_j = allKeyFramesHistory[index_i];
			SE3 Ti = frame_i->camToWorld;
			SE3 Tj = frame_j->camToWorld;
			SE3 Tij = frame_i->camToWorld.inverse() * frame_j->camToWorld;
			Mat33 Ri = dso_vi::IMUData::convertCamFrame2IMUFrame(Ti.matrix(), getTbc()).block<3, 3>(0, 0);
			Mat33 Rj = dso_vi::IMUData::convertCamFrame2IMUFrame(Tj.matrix(), getTbc()).block<3, 3>(0, 0);
			Mat33 R_ij = Ri.inverse() * Rj;

			Mat33 tmp_A(3, 3);
			tmp_A.setZero();
			Vec3 tmp_b(3);
			tmp_b.setZero();

			//============================================for the jacobian of rotation=================================================
			PreintegratedImuMeasurements *preint_imu = dynamic_cast<PreintegratedImuMeasurements *>(frame_j->imu_preintegrated_last_kf_);
			ImuFactor imu_factor(X(0), V(0),
								 X(1), V(1),
								 B(0),
								 *preint_imu);

			// relative pose wrt IMU
			gtsam::Pose3 relativePose(dso_vi::IMUData::convertCamFrame2IMUFrame(Tij.matrix(), getTbc()));

			gtsam::Vector3 velocity;
			velocity << 0, 0, 0;

			Values initial_values;
			initial_values.insert(X(0), gtsam::Pose3::identity());
			initial_values.insert(V(0), gtsam::zero(3));
			initial_values.insert(B(0), frame_i->bias);
			initial_values.insert(X(1), relativePose);
			initial_values.insert(V(1), velocity);

			//boost::shared_ptr<GaussianFactor> linearFactor =
			imu_factor.linearize(initial_values);
			// useless Jacobians of reference frame (cuz we're not optimizing reference frame)
			gtsam::Matrix J_imu_Rt_i, J_imu_v_i, J_imu_Rt, J_imu_v, J_imu_bias;
			Vector9 res_imu = imu_factor.evaluateError(
					gtsam::Pose3::identity(), gtsam::zero(3), relativePose, velocity, frame_i->bias,
					J_imu_Rt_i, J_imu_v_i, J_imu_Rt, J_imu_v, J_imu_bias
			);
			//=======================================================================================================================

			tmp_A = J_imu_bias.block<3, 3>(0, 3);
			tmp_A = tmp_A;
//			std::cout << "J_imu_bias bg =  \n" << J_imu_bias << std::endl;
//			std::cout << "J_imu_bias bg =  " << std::endl << tmp_A << std::endl;
			tmp_b = res_imu.block<3, 1>(0, 0);
//			std::cout << "the realtive rotation se3 is :\n" << tmp_b << std::endl;
			A += tmp_A.transpose() * tmp_A;
			b -= tmp_A.transpose() * tmp_b;

		}
		delta_bg = A.ldlt().solve(b);

		Vec6 agbias;
		agbias.setZero();
		agbias.tail<3>() = delta_bg;
		for (int index_i = allKeyFramesHistory.size() - 1;
			 index_i >= allKeyFramesHistory.size() - WINDOW_SIZE + 1; index_i--)
		{
			FrameShell *frame_i = allKeyFramesHistory[index_i];
			frame_i->bias = frame_i->bias + agbias;
			frame_i->imu_preintegrated_last_kf_->biasCorrectedDelta(frame_i->bias);
		}

        gyroBiasEstimate = allKeyFramesHistory.back()->bias.gyroscope();

//		std::cout << "\"gyroscope bias initial calibration::::::; " << delta_bg.transpose() << std::endl;
		std::cout << "\"gyroscope bias initial calibration::::::; " << gyroBiasEstimate.transpose() << std::endl;




	}
}

void FullSystem::UpdateState(Vec3 &g, VecX &x)
{

	// here Rs, Ps in vins are all from imu to world;
	std::vector<Mat33 >Rs;

	std::vector<Vec3> Ps;
	std::vector<Vec3> Vs;
    double scale = (x.tail<1>())(0);

	SE3 TBC(getTbc());
	SE3 TCB = TBC.inverse();

	for (int i = 0; i <allKeyFramesHistory.size() ; i++)
    {
        SE3 Twci = allKeyFramesHistory[i]->camToWorld;
        SE3 Twbi = Twci * TCB;
		Ps.push_back(Twbi.translation());
		Rs.push_back(Twbi.rotationMatrix());
		//Vs[kv] = frame_i->second.R * x.segment<3>(i * 3)
		Vs.push_back(Twbi.rotationMatrix() * x.segment<3>(i * 3));
	}

	std::cout << "Pre-update" << std::endl;
	for (int i = 1; i < allKeyFramesHistory.size(); i++)
	{
		SE3 groundtruth_T0(allKeyFramesHistory[i-1]->groundtruth.pose.matrix());
		SE3 estimated_T0(Rs[i-1], Ps[i-1]);

		SE3 groundtruth_Ti(allKeyFramesHistory[i]->groundtruth.pose.matrix());
		SE3 estimated_Ti(Rs[i], Ps[i]);

		SE3 groundtruth_relative_pose = groundtruth_T0.inverse() * groundtruth_Ti;
		SE3 estimated_relative_pose = estimated_T0.inverse() * estimated_Ti;

		double translation_direction_error = acos(
				estimated_relative_pose.translation().normalized().dot(groundtruth_relative_pose.translation().normalized())
		) * 180 / M_PI;

		std::cout 	<< "Scale: "
					 << groundtruth_relative_pose.translation().norm() / estimated_relative_pose.translation().norm() << ", "
					 << "translation dir error: " << translation_direction_error
					 << std::endl;
	}

	Mat33 R0 = g2R(g);
	std::cout<< "g: "<<g<<"\nR0: "<<R0<<std::endl;

	double yaw = R2ypr(R0 * Rs[0]).x();
	R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
	g = R0 * g;
    gravity = g;
	std::cout<< "yaw: "<<yaw << "\n R0: \n"<<R0<<std::endl;
	std::cout<< "g: "<<g<<std::endl;
	Mat33 rot_diff = R0;

	Mat33 initial_R = Rs[0];
	Vec3 initial_P = Ps[0];

    // Rescale the camera position (origin is now at keyframe 0, but KF 0 has orientation wrt inertial frame)
	for (int i = 0; i < allFrameHistory.size(); i++)
	{
		std::cout<<"the pose of frame: "<<allFrameHistory[i]->id <<" before update:\n "<<allFrameHistory[i]->camToWorld.matrix()<<std::endl;

		SE3 imu2World = allFrameHistory[i]->camToWorld * TCB;

		// rescale and center at first keyframe
		allFrameHistory[i]->imuToWorld = SE3(
				imu2World.rotationMatrix(), scale * (imu2World.translation() - Ps[0])
		);

		// make the orientation wrt inertial frame
		allFrameHistory[i]->imuToWorld = SE3(
				rot_diff * allFrameHistory[i]->imuToWorld.rotationMatrix(),
				rot_diff * allFrameHistory[i]->imuToWorld.translation()
		);

		allFrameHistory[i]->camToWorld = allFrameHistory[i]->imuToWorld * TBC;
		std::cout<<"the pose of frame: "<<allFrameHistory[i]->id <<" after update:\n "<<allFrameHistory[i]->camToWorld.matrix()<<std::endl;
	}

    for (int i = allKeyFramesHistory.size(); i >= 0; i--)
	{
//		Ps[i] = scale * Ps[i] - Rs[i] * getTbc().block<3, 1>(0, 3) -
//				(scale * Ps[0] - Rs[0] * getTbc().block<3, 1>(0, 3));
		Ps[i] = scale * Ps[i] - scale * Ps[0];
	}

	for (int i = 0; i <allKeyFramesHistory.size() ; i++)
	{
		Ps[i] = rot_diff * Ps[i];
		Rs[i] = rot_diff * Rs[i];
		Vs[i] = rot_diff * Vs[i];
		//allKeyFramesHistory[i]->setNavstate(Rs[i],Ps[i],Vs[i]);
	}

	std::cout << "First KF: " << Rs[0] << ", " << Ps[0].transpose() << std::endl;

	// verify with the groundtruth that the scaling is right

	for (int i = 1; i < allKeyFramesHistory.size(); i++)
	{
		SE3 groundtruth_T0(allKeyFramesHistory[i-1]->groundtruth.pose.matrix());
		SE3 estimated_T0(Rs[i-1], Ps[i-1]);

		SE3 groundtruth_Ti(allKeyFramesHistory[i]->groundtruth.pose.matrix());
		SE3 estimated_Ti(Rs[i], Ps[i]);

		SE3 groundtruth_relative_pose = groundtruth_T0.inverse() * groundtruth_Ti;
		SE3 estimated_relative_pose = estimated_T0.inverse() * estimated_Ti;

		double translation_direction_error = acos(
				estimated_relative_pose.translation().normalized().dot(groundtruth_relative_pose.translation().normalized())
		) * 180 / M_PI;

		std::cout 	<< "Scale: "
					 << groundtruth_relative_pose.translation().norm() / estimated_relative_pose.translation().norm() << ", "
					 << "translation dir error: " << translation_direction_error
					 << std::endl;

		std::cout << "Velocity GT VS Our: \n"
				  << allKeyFramesHistory[i]->groundtruth.velocity.transpose() << std::endl
				  << Vs[i].transpose()
				  << std::endl;
	}

	std::cout <<"----------------------normal frames--------------------------"<<std::endl;
	for (int i = 1; i < allFrameHistory.size(); i++)
	{
		SE3 groundtruth_T0(allFrameHistory[i-1]->groundtruth.pose.matrix());
		SE3 estimated_T0 = allFrameHistory[i-1]->imuToWorld;

		SE3 groundtruth_Ti(allFrameHistory[i]->groundtruth.pose.matrix());
		SE3 estimated_Ti = allFrameHistory[i]->imuToWorld;

		SE3 groundtruth_relative_pose = groundtruth_T0.inverse() * groundtruth_Ti;
		SE3 estimated_relative_pose = estimated_T0.inverse() * estimated_Ti;

		double translation_direction_error = acos(
				estimated_relative_pose.translation().normalized().dot(groundtruth_relative_pose.translation().normalized())
		) * 180 / M_PI;

		std::cout 	 << "Frame: " << i << " Scale: "
					 << groundtruth_relative_pose.translation().norm() / estimated_relative_pose.translation().norm() << ", "
					 << "translation dir error: " << translation_direction_error
					 << std::endl;
	}


	std::cout<<"scale is : "<<scale<<std::endl;
	// Rescale the depth

	// keep a log of rescaled points (TODO: find if we can just read the points without any duplicates)
	std::map<PointHessian*, bool> rescaled_points;
	std::map<ImmaturePoint*, bool> rescaled_immature_points;

	for (int fh_idx = frameHessians.size()-1; fh_idx > 0  ; fh_idx--)
	{
		//if(allKeyFramesHistory[i]->fh == NULL) continue;
//		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

		//std::cout<<"i: "<< i <<std::endl;

		FrameHessian *fh = frameHessians[fh_idx];

		for (int i = 0; i < allKeyFramesHistory.size(); i++)
		{
			if (allKeyFramesHistory[i]->fh == fh)
			{
				int shell_idx = i;
				SE3 imuToWorld = SE3(Rs[shell_idx], Ps[shell_idx]);
				SE3 camToWorld = imuToWorld * TBC;

				fh->shell->camToWorld = camToWorld;
				fh->shell->velocity = Vs[shell_idx];
				fh->PRE_camToWorld = camToWorld;
				fh->PRE_worldToCam = camToWorld.inverse();
				fh->worldToCam_evalPT = fh->PRE_worldToCam;

				// relinearize the states
				fh->setState(Vec10::Zero());
				fh->setStateZero(Vec10::Zero());

				size_t change_points_count = 0;
				for(PointHessian* ph : fh->pointHessians)
				{
					if ( rescaled_points.find(ph) != rescaled_points.end() )
					{
						// the point has already been rescaled before
						continue;
					}

					// the point hasn't been rescaled before
					// the depth is in the co-ordinate of the host frame. But the scaling is required in global frame

					float old_depth = ph->idepth;
					float new_depth = ph->idepth / scale;
					ph->setIdepth(new_depth);
					ph->setIdepthZero(new_depth);
					ph->idepth_backup = new_depth;

					rescaled_points.insert(std::make_pair(ph, true));
					assert(old_depth != ph->idepth);

					change_points_count++;
				}

				for(PointHessian* ph : fh->pointHessiansMarginalized)
				{
					if ( rescaled_points.find(ph) != rescaled_points.end() )
					{
						// the point has already been rescaled before
						continue;
					}

					// the point hasn't been rescaled before
					// the depth is in the co-ordinate of the host frame. But the scaling is required in global frame

					float old_depth = ph->idepth;
					float new_depth = ph->idepth / scale;
					ph->setIdepth(new_depth);
					ph->setIdepthZero(new_depth);
					ph->idepth_backup = new_depth;

					rescaled_points.insert(std::make_pair(ph, true));
					assert(old_depth != ph->idepth);

					change_points_count++;
				}

				for(PointHessian* ph : fh->pointHessiansOut)
				{
					if ( rescaled_points.find(ph) != rescaled_points.end() )
					{
						// the point has already been rescaled before
						continue;
					}

					// the point hasn't been rescaled before
					// the depth is in the co-ordinate of the host frame. But the scaling is required in global frame

					float old_depth = ph->idepth;
					float new_depth = ph->idepth / scale;
					ph->setIdepth(new_depth);
					ph->setIdepthZero(new_depth);
					ph->idepth_backup = new_depth;

					rescaled_points.insert(std::make_pair(ph, true));
					assert(old_depth != ph->idepth);

					change_points_count++;
				}

				for(ImmaturePoint* ip : fh->immaturePoints)
				{
					if ( rescaled_immature_points.find(ip) != rescaled_immature_points.end() )
					{
						// the point has already been rescaled before
						continue;
					}

					// the point hasn't been rescaled before
					// the depth is in the co-ordinate of the host frame. But the scaling is required in global frame
					if(ip)
					{
						ip->idepth_max = ip->idepth_max / scale;
						ip->idepth_min = ip->idepth_min / scale;

						change_points_count++;

						rescaled_immature_points.insert(std::make_pair(ip, true));
					}
				}

				std::cout << "Changed points: " << change_points_count << std::endl;

				break;
			}
		}


		for (PointHessian *ph : fh->pointHessians)
		{
			for(PointFrameResidual* r : ph->residuals)
			{
				r->linearize(&Hcalib);
			}

			for (size_t resIdx = 0; resIdx < 2; resIdx++)
			{
				if (ph->lastResiduals[resIdx].first != 0 && ph->lastResiduals[resIdx].second == ResState::IN)
				{
					PointFrameResidual *r = ph->lastResiduals[resIdx].first;
					r->linearize(&Hcalib);
				}
			}
		}
	}

	coarseTracker->makeCoarseDepthL0(frameHessians);
	coarseTracker_forNewKF->makeCoarseDepthL0(frameHessians);

	// keep a log of rescaled points (TODO: find if we can just read the points without any duplicates)
//    std::map<PointHessian*, bool> rescaled_points;
//    for (int i = 0; i < 10  ; i++)
//    {
//        //if(allKeyFramesHistory[i]->fh == NULL) continue;
//        //boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
//        //std::cout<<"i: "<< i <<std::endl;
//        //assert( frameHessians[i] == allKeyFramesHistory[i]->fh );
//
//        FrameHessian *fh = allKeyFramesHistory[i]->fh;
//
//        SE3 imuToWorld = SE3(Rs[i], Ps[i]);
//        SE3 camToWorld = imuToWorld * TBC;
//
////        fh->shell->camToWorld = camToWorld;
////        fh->shell->velocity = Vs[i];
//        fh->PRE_camToWorld = camToWorld;
//        fh->PRE_worldToCam = camToWorld.inverse();
//
////        for(IOWrap::Output3DWrapper* ow : outputWrapper)
////            ow->resetKeyframes(fh->frameID,fh,&Hcalib,scale);
//		std::cout<<"update "<<i<<" frame!!"<<std::endl;
////        for(PointHessian* ph : fh->pointHessians)
////        {
////            if ( rescaled_points.find(ph) != rescaled_points.end() )
////            {
////                // the point has already been rescaled before
////                continue;
////            }
////
////            // the point hasn't been rescaled before
////            // the depth is in the co-ordinate of the host frame. But the scaling is required in global frame
////
////            float new_depth = ph->idepth * scale;
////            ph->setIdepth(new_depth);
////            ph->setIdepthZero(new_depth);
////
////            rescaled_points.insert(std::make_pair(ph, true));
////        }
//    }


//	std::map<PointHessian*, bool> rescaled_points;
//	for (int i = frameHessians.size()-1; i > 0  ; i--)
//	{
//        //if(allKeyFramesHistory[i]->fh == NULL) continue;
//		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
//        //std::cout<<"i: "<< i <<std::endl;
//		FrameHessian *fh = frameHessians[i];
//
//		SE3 imuToWorld = SE3(Rs[i], Ps[i]);
//		SE3 camToWorld = imuToWorld * TBC;
//
//		fh->shell->camToWorld = camToWorld;
//        fh->shell->velocity = Vs[i];
//		fh->PRE_camToWorld = camToWorld;
//		fh->PRE_worldToCam = camToWorld.inverse();
//		for(PointHessian* ph : fh->pointHessians)
//		{
//			if ( rescaled_points.find(ph) != rescaled_points.end() )
//			{
//				// the point has already been rescaled before
//				continue;
//			}
//
//			// the point hasn't been rescaled before
//			// the depth is in the co-ordinate of the host frame. But the scaling is required in global frame
//
//			float new_depth = ph->idepth * scale;
//			ph->setIdepth(new_depth);
//			ph->setIdepthZero(new_depth);
//
//			rescaled_points.insert(std::make_pair(ph, true));
//		}
//	}

//	for (int i = 0; i <= frame_count; i++)
//	{
//		Mat33 Ri = allFrameHistory[i]->RCW();
//		Vec3 Pi = allFrameHistory[i]->TCW();
//		Ps[i] = Pi;
//		Rs[i] = Ri;
//		all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
//
//		Matrix3d R0 = Utility::g2R(g);
//		double yaw = Utility::R2ypr(R0 * Rs[0]).x();
//		R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
//		g = R0 * g;
//	}

	return;
}

void FullSystem::addActiveFrame( ImageAndExposure* image, int id , std::vector<dso_vi::IMUData> vimuData, double ftimestamp,
								 dso_vi::ConfigParam &config , dso_vi::GroundTruthIterator::ground_truth_measurement_t groundtruth )
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);

	if(!IMUinitialized && allKeyFramesHistory.size() >= WINDOW_SIZE)
    {
		// wait for the mapping to end
		while (!isLocalBADone.load());

		// lock everything
		boost::unique_lock<boost::mutex> lockMap(mapMutex);
		boost::unique_lock<boost::mutex> lockTrackMap(trackMapSyncMutex);
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

        Vec3 g;
        Eigen::VectorXd initialstates;
		solveGyroscopeBiasbyGTSAM();
        if(SolveScale(g, initialstates))
		{
			UpdateState(g,initialstates);
			IMUinitialized = true;
		}
		else
		{
			std::cout << "Failed to solve scale!!!" << std::endl;
			exit(0);
		}
    }

	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell(accBiasEstimate, gyroBiasEstimate);
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->viTimestamp = ftimestamp;
    shell->incoming_id = id;
    shell->groundtruth = groundtruth;
	shell->fh = fh;
	fh->shell = shell;
	allFrameHistory.push_back(shell);

	//============================ accumulate  IMU data=====================================

	mvIMUSinceLastF = vimuData;
	for(dso_vi::IMUData rawimudata : vimuData)
	{
		mvIMUSinceLastKF.push_back(rawimudata);
	}

	if (initialized)
	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		shell->last_kf = (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
						 ? coarseTracker_forNewKF->lastRef->shell
						 : coarseTracker->lastRef->shell;
	}
	else
	{
		shell->last_kf = NULL;
	}

	if (allFrameHistory.size() > 1)
	{
		std::cout << "Frame: " << shell->id << std::endl;

		shell->last_frame = allFrameHistory[allFrameHistory.size()-2];

		assert(shell->last_frame);
		assert(shell->id - shell->last_frame->id == 1);

		if (!shell->last_kf)
		{
			// this condition is true when there is just one keyframe (frame id 0) in the keyframe history
			std::cout << "Last KF swap" << std::endl;
			shell->last_kf = shell->last_frame->last_kf;
		}

		if (shell->last_kf)
		{ // this condition is false for the first tracked frame
			std::cout << "Frame/Keyframe: " << shell->id << "/" << shell->last_kf->id << std::endl;
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			shell->updateIMUmeasurements(mvIMUSinceLastF);
		}
	}


	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib, mvIMUSinceLastKF);

	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{

			coarseInitializer->setFirst(&Hcalib, fh);
			shell->last_kf = shell;
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		Vec4 tres = trackNewCoarse(fh);
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }

		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||
					2*coarseTracker->firstCoarseRMSE < tres[0];

		}




        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);




		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		mvIMUSinceLastF.clear();
		return;
	}
}
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{

	isLocalBADone = false;
	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh);
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
	isLocalBADone = true;
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);

				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();



	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}




	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();




	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);





	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;




	// =========================== REMOVE OUTLIER =========================
	removeOutliers();




	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");






	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	ef->marginalizePointsF();



	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);





    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }



	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(frameHessians[i]); i=0;}



	printLogLine();
    //printEigenValLine();

// 	=========================== Clear the IMU buffer for next round ===========
	// This is wrong. The new keyframe will not be the current frame, but something between the previous keyframe and the current frame
	// Hence, the IMU measurements since last keyframe is not empty
	mvIMUSinceLastKF.clear();
}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }


		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->iR*rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}



	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}



void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}





}
