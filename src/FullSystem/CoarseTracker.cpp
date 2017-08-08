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

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <GroundTruthIterator/GroundTruthIterator.h>


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}


CoarseTracker::CoarseTracker(int ww, int hh, FullSystem* _fullsystem) : lastRef_aff_g2l(0,0), fullSystem(_fullsystem)
{
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
        int hl = hh>>lvl;

        idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        pc_u[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_v[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_color[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

	}

	// warped buffers
	buf_warped_rx = allocAligned<4,float>(ww*hh, ptrToDelete);
	buf_warped_ry = allocAligned<4,float>(ww*hh, ptrToDelete);
	buf_warped_rz = allocAligned<4,float>(ww*hh, ptrToDelete);
	buf_warped_lpc_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dx = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dy = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_residual = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_weight = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(ww*hh, ptrToDelete);


	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
//	std::cout<<"fullSystem->getTbc() :\n"<<fullSystem->getTbc()<<std::endl;
//	camtoimu = SE3(fullSystem->getTbc());
//	std::cout<<"camtoimu is initialized as :\n"<<camtoimu.matrix()<<std::endl;
//	imutocam = camtoimu.inverse();
}
CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}



void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians)
{
	// make coarse tracking templates for latstRef.
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
	memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

	for(FrameHessian* fh : frameHessians)
	{
		for(PointHessian* ph : fh->pointHessians)
		{
			if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
			{
				PointFrameResidual* r = ph->lastResiduals[0].first;
				assert(r->efResidual->isActive() && r->target == lastRef);
				int u = r->centerProjectedTo[0] + 0.5f;
				int v = r->centerProjectedTo[1] + 0.5f;
				float new_idepth = r->centerProjectedTo[2];
				float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

				idepth[0][u+w[0]*v] += new_idepth *weight;
				weightSums[0][u+w[0]*v] += weight;
			}
		}
	}


	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		float* idepth_l = idepth[lvl];
		float* weightSums_l = weightSums[lvl];

		float* idepth_lm = idepth[lvlm1];
		float* weightSums_lm = weightSums[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
			{
				int bidx = 2*x   + 2*y*wlm1;
				idepth_l[x + y*wl] = 		idepth_lm[bidx] +
											idepth_lm[bidx+1] +
											idepth_lm[bidx+wlm1] +
											idepth_lm[bidx+wlm1+1];

				weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
											weightSums_lm[bidx+1] +
											weightSums_lm[bidx+wlm1] +
											weightSums_lm[bidx+wlm1+1];
			}
	}


    // dilate idepth by 1.
	for(int lvl=0; lvl<2; lvl++)
	{
		int numIts = 1;


		for(int it=0;it<numIts;it++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for(int i=w[lvl];i<wh;i++)
			{
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
					if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
					if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
					if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
					if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
				}
			}
		}
	}


	// dilate idepth by 1 (2 on lower levels).
	for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
	{
		int wh = w[lvl]*h[lvl]-w[lvl];
		int wl = w[lvl];
		float* weightSumsl = weightSums[lvl];
		float* weightSumsl_bak = weightSums_bak[lvl];
		memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
		float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
										// read values with weightSumsl>0, and write ones with weightSumsl<=0.
		for(int i=w[lvl];i<wh;i++)
		{
			if(weightSumsl_bak[i] <= 0)
			{
				float sum=0, num=0, numn=0;
				if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
				if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
				if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
				if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
				if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
			}
		}
	}


	// normalize idepths and weights.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		float* weightSumsl = weightSums[lvl];
		float* idepthl = idepth[lvl];
		Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

		int wl = w[lvl], hl = h[lvl];

		int lpc_n=0;
		float* lpc_u = pc_u[lvl];
		float* lpc_v = pc_v[lvl];
		float* lpc_idepth = pc_idepth[lvl];
		float* lpc_color = pc_color[lvl];


		for(int y=2;y<hl-2;y++)
			for(int x=2;x<wl-2;x++)
			{
				int i = x+y*wl;

				if(weightSumsl[i] > 0)
				{
					idepthl[i] /= weightSumsl[i];
					lpc_u[lpc_n] = x;
					lpc_v[lpc_n] = y;
					lpc_idepth[lpc_n] = idepthl[i];
					lpc_color[lpc_n] = dIRefl[i][0];



					if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
					{
						idepthl[i] = -1;
						continue;	// just skip if something is wrong.
					}
					lpc_n++;
				}
				else
					idepthl[i] = -1;

				weightSumsl[i] = 1;
			}

		pc_n[lvl] = lpc_n;
	}

}

void CoarseTracker::calcGSSSESingleIMU(int lvl, Mat1717 &H_out, Vec17 &b_out, const gtsam::NavState navState_, AffLight aff_g2l)
{
	acc.initialize();
	int n = buf_warped_n;
	SE3 Tib(navState_.pose().matrix());
	SE3 Tw_reffromNAV = SE3(lastRef->shell->navstate.pose().matrix()) * imutocam().inverse();
	SE3 Tw_ref = lastRef->shell->camToWorld;
	Mat33 Rcb = imutocam().rotationMatrix();
	SE3 Trb = Tw_ref.inverse() * Tib;

	// refToNew for debug
	SE3 refToNew = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
			(navState_.pose().inverse() * lastRef->shell->navstate.pose()).matrix()
	));
//	std::cout << "GSSSE ref2New: \n" << refToNew.matrix() << std::endl;
//	std::cout << "GSSSE Trb: \n" << Trb.matrix() << std::endl;
//	std::cout << "GSSSE Tib: \n" << Tib.matrix()<<std::endl;
//	std::cout << "GSSSE Tw_ref:\n" << Tw_ref.matrix()<<std::endl;
//	std::cout << "GSSSE Tw_reffromNAV:\n" << Tw_reffromNAV.matrix()<<std::endl;

	for(int i=0;i<n;i++)
	{
		Vec3 Pr,PI;
		Vec6 Jab;
		Vec2 dxdy;
		dxdy(0) = *(buf_warped_dx+i);
		dxdy(1)	= *(buf_warped_dy+i);
		float b0 = lastRef_aff_g2l.b;
		float a = (float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]);

		float id = *(buf_warped_idepth+i);
		float u = *(buf_warped_u+i);
		float v = *(buf_warped_v+i);
		Pr(0) = *(buf_warped_rx+i);
		Pr(1) = *(buf_warped_ry+i);
		Pr(2) = *(buf_warped_rz+i);
		PI = Tw_ref * Pr;
		// Jacobian of camera projection
		Matrix23 Maux;
		Maux.setZero();
		Maux(0,0) = fx[lvl];
		Maux(0,1) = 0;
		Maux(0,2) = -u *fx[lvl];
		Maux(1,0) = 0;
		Maux(1,1) = fy[lvl];
		Maux(1,2) = -v * fy[lvl];
		Matrix23 Jpi = Maux * id;
		Jab.head(3) = dxdy.transpose() * Jpi * (-Rcb);
		Jab.tail(3) = dxdy.transpose() * Jpi * Rcb * SO3::hat(Tib.inverse() * PI); //Rrb.transpose()*(Pr-Prb));

		acc.updateSingleWeighted(
				(float)Jab(0),
				(float)Jab(1),
				(float)Jab(2),
				(float)Jab(3),
				(float)Jab(4),
				(float)Jab(5),
				a*(b0-buf_warped_refColor[i]),
				-1.0,
				buf_warped_residual[i],
				buf_warped_weight[i]
		);

//		if (fullSystem->isIMUinitialized()&& i<10 && lvl == 4)
//		{
//			// debug
//			std::cout << i << " J_r: " << Jab.tail(3).transpose() << std::endl;
//			std::cout << i << " Pb: " << (Tib.inverse() * PI).transpose() << std::endl;
//			std::cout << i << " Pr: " << Pr.transpose()<<std::endl;
//		}

	}

	acc.finish();

    H_out.setZero();
    b_out.setZero();
	H_out.block<8,8>(0,0) = acc.H.topLeftCorner<8,8>().cast<double>()  * (1.0f/n);
	b_out.segment<8>(0) = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);
	//std::cout<<"H_out:\n"<<H_out.block<8,8>(0,0)<<std::endl;
	//std::cout<<"b_out:\n"<<b_out.segment<8>(0)<<std::endl;
	// here rtvab means rotation, translation, affine, velocity and biases

	std::cout << "H_out: \n" << H_out.topLeftCorner<6,6>() << std::endl;
	std::cout << "b_out: \n" << b_out.segment<6>(0).transpose() << std::endl;

    Mat1717 H_imu_rtavb;
	Vec17 b_imu_rtavb;
	Mat1517 J_imu_rtavb;

	J_imu_rtavb.setZero();
	// for rotation and translation
//	J_imu_rtavb.topLeftCorner<9, 3>() = J_imu_Rt.block<9, 3>(0,3);
//	J_imu_rtavb.block<9, 3>(0, 3) = J_imu_Rt.topLeftCorner<9, 3>();
//    J_imu_rtavb.block<9, 3>(0, 8) = J_imu_v.topLeftCorner<9, 3>();

	J_imu_rtavb.block<6, 3>(0, 3) = J_imu_Rt.topLeftCorner<6, 3>();
//	J_imu_rtavb.block<6, 3>(0, 3) = J_imu_Rt.topLeftCorner<6, 3>();
	J_imu_rtavb.topLeftCorner<6, 3>() = J_imu_Rt.block<6, 3>(0, 3);
	res_imu.segment<9>(6) = Eigen::Matrix<double,9,1>::Zero();

    information_imu.topLeftCorner<3, 3>() /= SCALE_IMU_R;
    information_imu.block<3,3>(3, 3) /= SCALE_IMU_T;

	H_imu_rtavb = J_imu_rtavb.transpose() * information_imu * J_imu_rtavb;
	b_imu_rtavb = J_imu_rtavb.transpose() * information_imu * res_imu;
	//H_imu_rtavb /= 1000.0;
	//b_imu_rtavb /= 1000.0;

    std::cout << "J_imu: \n " << J_imu_rtavb << std::endl;
	std::cout << "H_imu: \n" << H_imu_rtavb.topLeftCorner<6,6>() << std::endl;
	std::cout << "b_imu: \n" << b_imu_rtavb.segment<6>(0).transpose() << std::endl;

	H_out += H_imu_rtavb;
	b_out += b_imu_rtavb;

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}

void CoarseTracker::calcGSSSESingle(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{

        acc.initialize();
        int n = buf_warped_n;
		SE3 Trb;
		Mat33 Rcb,Rrb;
		Vec3 Prb;
        SE3 Tref_new = refToNew.inverse();
		Rcb = imutocam().rotationMatrix();
		Trb = Tref_new * imutocam();
//	std::cout << "GSSSE ref2New: \n" << refToNew.matrix() << std::endl;
//	std::cout << "GSSSE Trb: \n" << Trb.matrix() << std::endl;
//	std::cout << "GSSSE Tib: \n" << Tib.matrix()<<std::endl;
//	std::cout << "GSSSE Tw_ref:\n" << Tw_ref.matrix()<<std::endl;
		SE3 Tib,Tw_ref;
		Tib = lastRef->shell->camToWorld * Tref_new * SE3(imutocam());
		Tw_ref = lastRef->shell->camToWorld;
//		std::cout << "no imu ref2New: \n" << refToNew.matrix() << std::endl;
//		std::cout << "no imu Trb: \n" << Trb.matrix() << std::endl;
//		std::cout << "no imu Tib: \n" << Tib.matrix() << std::endl;
//		std::cout << "no imu Tib from nav\n"<<lastRef->shell->navstate.pose().matrix()<<std::endl;
//		std::cout << "no imu Tw_ref: \n" << Tw_ref.matrix() << std::endl;

        for(int i=0;i<n;i++)
        {
                Vec3 Pr;
                Vec6 Jab;
                Vec2 dxdy;
                dxdy(0) = *(buf_warped_dx+i);
				dxdy(1)	= *(buf_warped_dy+i);
                float b0 = lastRef_aff_g2l.b;
                float a = (float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]);

                float id = *(buf_warped_idepth+i);
                float u = *(buf_warped_u+i);
                float v = *(buf_warped_v+i);
				Pr(0) = *(buf_warped_rx+i);
				Pr(1) = *(buf_warped_ry+i);
				Pr(2) = *(buf_warped_rz+i);

			    // Jacobian of camera projection
                Matrix23 Maux;
                Maux.setZero();
                Maux(0,0) = fx[lvl];
                Maux(0,1) = 0;
                Maux(0,2) = -u *fx[lvl];
                Maux(1,0) = 0;
                Maux(1,1) = fy[lvl];
                Maux(1,2) = -v * fy[lvl];
                Matrix23 Jpi = Maux * id;
				Jab.head(3) = dxdy.transpose() * Jpi * (-Rcb);
                Jab.tail(3) = dxdy.transpose() * Jpi * Rcb * SO3::hat(Trb.inverse() * Pr); //Rrb.transpose()*(Pr-Prb));
//			Vector3d Paux = Rcb*Rwb.transpose()*(Pw-Pwb);
//			//Matrix3d J_Pc_dRwb = Sophus::SO3::hat(Paux) * Rcb;
//			Matrix<double,2,3> JdRwb = - Jpi * (Sophus::SO3::hat(Paux) * Rcb);
                acc.updateSingleWeighted(
                                (float)Jab(0),
                                (float)Jab(1),
                                (float)Jab(2),
                                (float)Jab(3),
                                (float)Jab(4),
                                (float)Jab(5),
                                a*(b0-buf_warped_refColor[i]),
                                -1.0,
                                buf_warped_residual[i],
                                buf_warped_weight[i]
                );

//				if (fullSystem->isIMUinitialized()&& i <10 && lvl ==4)
//				{
//					// debug
//					std::cout << i << " J_r: " << Jab.tail(3).transpose() << std::endl;
//					std::cout << i << " Pb: " << (Trb.inverse() * Pr).transpose() << std::endl;
//					std::cout << i << " Pr: " << Pr.transpose()<<std::endl;
//				}

        }

        acc.finish();

        H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
        b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);
//		std::cout<<"H_out: \n"<<H_out<<std::endl;
//	    std::cout<<"b_out: \n"<<b_out<<std::endl;

//        Mat33 J_imu_rot,H_imu_rot;
//        Vec3 r_imu_rot,b_imu_rot;
//
//        J_imu_rot = J_imu_Rt.block<3, 3>(0, 0);
//        r_imu_rot = res_imu.segment(0, 2);
//
//        Mat33 information_r = information_imu.block<3, 3>(6, 6);
//
//        H_imu_rot.noalias() = J_imu_rot.transpose() * information_r * J_imu_rot;
//        b_imu_rot.noalias() = J_imu_rot.transpose() * information_r * r_imu_rot;

    	//H_out.block<3,3>(0,0) += H_imu_rot;
    	//b_out.segment<3>(0) += b_imu_rot;

        H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
        H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
        H_out.block<8,1>(0,6) *= SCALE_A;
        H_out.block<8,1>(0,7) *= SCALE_B;
        H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
        H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
        H_out.block<1,8>(6,0) *= SCALE_A;
        H_out.block<1,8>(7,0) *= SCALE_B;
        b_out.segment<3>(0) *= SCALE_XI_ROT;
        b_out.segment<3>(3) *= SCALE_XI_TRANS;
        b_out.segment<1>(6) *= SCALE_A;
        b_out.segment<1>(7) *= SCALE_B;
}


//void CoarseTracker::calcGSSSEstr(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
//{
//	acc.initialize();
//
//	float fxl = fx[lvl];
//	float fyl = fy[lvl];
//	float b0 = lastRef_aff_g2l.b;
//	float a = (float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]);
//	int n = buf_warped_n;
//	assert(n%4==0);
//
//	for(int i=0;i<n;i++)
//	{
//		float dx = (*(buf_warped_dx+i)) * fxl;
//		float dy = (*(buf_warped_dy+i)) * fxl;
//		float u = *(buf_warped_u+i);
//		float v = *(buf_warped_v+i);
//		float id = *(buf_warped_idepth+i);
//
//		acc.updateSingleWeighted(
//				id*dx,
//				id*dy,
//				-id*(u*dx+v*dy),
//				-(u*v*dx+dy*(1+v*v)),
//				u*v*dy+dx*(1+u*u),
//				u*dy-v*dx,
//				a*(b0-buf_warped_refColor[i]),
//				-1.0,
//				buf_warped_residual[i],
//				buf_warped_idepth[i]
//		);
//	}
//
//	acc.finish();
//
//	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
//	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);
//
//	Mat33 J_imu_rot,H_imu_rot;
//	Vec3 r_imu_rot,b_imu_rot;
//
//	J_imu_rot = J_imu_Rt.block<3, 3>(0, 0);
//	r_imu_rot = res_imu.segment(0, 2);
//
//	Mat33 information_r = Mat33::Identity(); // information_imu.block<3, 3>(6, 6);
//
//	H_imu_rot.noalias() = J_imu_rot.transpose() * information_r * J_imu_rot;
//	b_imu_rot.noalias() = J_imu_rot.transpose() * information_r * r_imu_rot;
//
//	H_out.block<3,3>(0,0) += H_imu_rot;
//	b_out.segment<3>(0) += b_imu_rot;
//
//	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
//	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
//	H_out.block<8,1>(0,6) *= SCALE_A;
//	H_out.block<8,1>(0,7) *= SCALE_B;
//	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
//	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
//	H_out.block<1,8>(6,0) *= SCALE_A;
//	H_out.block<1,8>(7,0) *= SCALE_B;
//	b_out.segment<3>(0) *= SCALE_XI_ROT;
//	b_out.segment<3>(3) *= SCALE_XI_TRANS;
//	b_out.segment<1>(6) *= SCALE_A;
//	b_out.segment<1>(7) *= SCALE_B;
//}

void CoarseTracker::calcGSSSEst(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	acc.initialize();

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float b0 = lastRef_aff_g2l.b;
	float a = (float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]);
	int n = buf_warped_n;
	assert(n%4==0);

	for(int i=0;i<n;i++)
	{
		float dx = (*(buf_warped_dx+i)) * fxl;
		float dy = (*(buf_warped_dy+i)) * fxl;
		float u = *(buf_warped_u+i);
		float v = *(buf_warped_v+i);
		float id = *(buf_warped_idepth+i);

		acc.updateSingleWeighted(
				id*dx,
				id*dy,
				-id*(u*dx+v*dy),
				-(u*v*dx+dy*(1+v*v)),
				u*v*dy+dx*(1+u*u),
				u*dy-v*dx,
				a*(b0-buf_warped_refColor[i]),
				-1.0,
				buf_warped_residual[i],
				buf_warped_idepth[i]
		);
	}

	acc.finish();

	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);

	Mat33 J_imu_rot,H_imu_rot;
	Vec3 r_imu_rot,b_imu_rot;

	J_imu_rot = J_imu_Rt.block<3, 3>(0, 0);
	r_imu_rot = res_imu.segment(0, 2);

	Mat33 information_r = Mat33::Identity(); // information_imu.block<3, 3>(6, 6);

	H_imu_rot.noalias() = J_imu_rot.transpose() * information_r * J_imu_rot;
	b_imu_rot.noalias() = J_imu_rot.transpose() * information_r * r_imu_rot;

	H_out.block<3,3>(0,0) += H_imu_rot;
	b_out.segment<3>(0) += b_imu_rot;

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}

void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	acc.initialize();

	__m128 fxl = _mm_set1_ps(fx[lvl]);
	__m128 fyl = _mm_set1_ps(fy[lvl]);
	__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
	__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);

	int n = buf_warped_n;
	assert(n%4==0);
	for(int i=0;i<n;i+=4)
	{
		__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl);
		__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);
		__m128 u = _mm_load_ps(buf_warped_u+i);
		__m128 v = _mm_load_ps(buf_warped_v+i);
		__m128 id = _mm_load_ps(buf_warped_idepth+i);


		acc.updateSSE_eighted(
				_mm_mul_ps(id,dx),
				_mm_mul_ps(id,dy),
				_mm_sub_ps(zero, _mm_mul_ps(id,_mm_add_ps(_mm_mul_ps(u,dx), _mm_mul_ps(v,dy)))),
				_mm_sub_ps(zero, _mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dx),
						_mm_mul_ps(dy,_mm_add_ps(one, _mm_mul_ps(v,v))))),
				_mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dy),
						_mm_mul_ps(dx,_mm_add_ps(one, _mm_mul_ps(u,u)))),
				_mm_sub_ps(_mm_mul_ps(u,dy), _mm_mul_ps(v,dx)),
				_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),
				minusOne,
				_mm_load_ps(buf_warped_residual+i),
				_mm_load_ps(buf_warped_weight+i));
    }

	acc.finish();

	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);

	Mat33 J_imu_rot,H_imu_rot;
	Vec3 r_imu_rot,b_imu_rot;

	J_imu_rot = J_imu_Rt.block<3, 3>(0, 0);
	r_imu_rot = res_imu.segment(0, 2);

	Mat33 information_r = Mat33::Identity(); // information_imu.block<3, 3>(6, 6);

	H_imu_rot.noalias() = J_imu_rot.transpose() * information_r * J_imu_rot;
	b_imu_rot.noalias() = J_imu_rot.transpose() * information_r * r_imu_rot;

	H_out.block<3,3>(0,0) += H_imu_rot;
	b_out.segment<3>(0) += b_imu_rot;

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}



Vec15 CoarseTracker::calcIMURes(gtsam::NavState current_navstate, Vec6 bias)
{
	information_imu = newFrame->shell->getIMUcovariance().inverse();

	// useless Jacobians of reference frame (cuz we're not optimizing reference frame)
	gtsam::Matrix  J_imu_Rt_i, J_imu_v_i, J_imu_bias_i;
	//newFrame->shell->velocity << 1, 1, 1;

	res_imu = newFrame->shell->evaluateIMUerrors(
			newFrame->shell->last_frame->navstate,
			current_navstate,
            newFrame->shell->bias,
			J_imu_Rt_i, J_imu_v_i, J_imu_Rt, J_imu_v, J_imu_bias_i, this->J_imu_bias
	);

	// in gtsam the error in Rtv due to bias is calcuated with respect to bias_i (of previous frame)
	this->J_imu_bias.block<9, 6>(0, 0) = J_imu_bias_i.block<9, 6>(0, 0);

	return res_imu;
}



Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, const SE3 &previousToNew, AffLight aff_g2l, float cutoffTH)
{
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();


	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.


	MinimalImageB3* resImage = 0;
	if(debugPlot)
	{
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(255,255,255));
	}

	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

//	std::cout << "calcRes: " << std::endl;
//	std::cout << "ref2cam: \n" << refToNew.matrix() << std::endl;

	for(int i=0;i<nl;i++)
	{
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

		Vec3f pr = Ki[lvl] * Vec3f(x, y, 1) / id;
		Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = fxl * u + cxl;
		float Kv = fyl * v + cyl;
		float new_idepth = id/pt[2];

//		if (fullSystem->isIMUinitialized() && i < 10 && lvl == 4)
//		{
//			std::cout << i << " pr: " << pr.transpose() << std::endl;
//		}

		if(lvl==0 && i%32==0)
		{
			// translation only (positive)
			Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
			float uT = ptT[0] / ptT[2];
			float vT = ptT[1] / ptT[2];
			float KuT = fxl * uT + cxl;
			float KvT = fyl * vT + cyl;

			// translation only (negative)
			Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
			float uT2 = ptT2[0] / ptT2[2];
			float vT2 = ptT2[1] / ptT2[2];
			float KuT2 = fxl * uT2 + cxl;
			float KvT2 = fyl * vT2 + cyl;

			//translation and rotation (negative)
			Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
			float u3 = pt3[0] / pt3[2];
			float v3 = pt3[1] / pt3[2];
			float Ku3 = fxl * u3 + cxl;
			float Kv3 = fyl * v3 + cyl;

			//translation and rotation (positive)
			//already have it.

			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

//		if (fullSystem->isIMUinitialized() && lvl==4 && i==0)
//		{
//			std::cout 	<< Ku << " "
//						<< Kv << " "
//						<< new_idepth << " "
//						<< wl << " "
//						<< hl << " "
//						<< std::endl;
//		}

		if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0)) continue;

		float refColor = lpc_color[i];
		Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
		if(!std::isfinite((float)hitColor[0])) continue;
		float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

//		if (fullSystem->isIMUinitialized() && lvl==4 && i==0)
//		{
//			std::cout << "Point error: " << residual
//					  << " Hw: " << hw
//					  << std::endl;
//		}

		if(fabs(residual) > cutoffTH)
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
			E += maxEnergy;
			numTermsInE++;
			numSaturated++;
		}
		else
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));

			E += hw *residual*residual*(2-hw);
			numTermsInE++;
			buf_warped_rx[numTermsInWarped] = pr(0);
			buf_warped_ry[numTermsInWarped] = pr(1);
			buf_warped_rz[numTermsInWarped] = pr(2);
			buf_warped_lpc_idepth[numTermsInWarped] = id;
			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;
		}
	}

	while(numTermsInWarped%4!=0)
	{
		buf_warped_rx[numTermsInWarped] = 0;
		buf_warped_ry[numTermsInWarped] = 0;
		buf_warped_rz[numTermsInWarped] = 0;
		buf_warped_lpc_idepth[numTermsInWarped] = 0;
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;



	//std::cout << "Normalized Residue: " << E / numTermsInE <<" numTermsInE:" <<numTermsInE<<" nl: " <<nl<< std::endl;

//    E += IMUenergy;
//=============================================================================================================
	if(debugPlot)
	{
		IOWrap::displayImage("RES", resImage, false);
		IOWrap::waitKey(0);
		delete resImage;
	}

	Vec6 rs;
	rs[0] = E;
	rs[1] = numTermsInE;
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
	rs[5] = numSaturated / (float)numTermsInE;

	return rs;
}



Vec6 CoarseTracker::calcResIMU(int lvl,const gtsam::NavState current_navstate, AffLight aff_g2l,const Vec6 biases, float cutoffTH)
{
	SE3 refToNew;
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];
	refToNew = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
			(current_navstate.pose().inverse() * lastRef->shell->navstate.pose()).matrix()
	));

	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.


	MinimalImageB3* resImage = 0;
	if(debugPlot)
	{
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(255,255,255));
	}

	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

//	std::cout << "CalcResIMU: " << std::endl;
//	std::cout << "ref2cam: \n" << refToNew.matrix() << std::endl;

	for(int i=0;i<nl;i++)
	{
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

		Vec3f pr = Ki[lvl] * Vec3f(x, y, 1) / id;
		Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = fxl * u + cxl;
		float Kv = fyl * v + cyl;
		float new_idepth = id/pt[2];

//		if (fullSystem->isIMUinitialized() && i < 10 && lvl == 4)
//		{
//			std::cout << i << " pr: " << pr.transpose() << std::endl;
//		}

		if(lvl==0 && i%32==0)
		{
			// translation only (positive)
			Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
			float uT = ptT[0] / ptT[2];
			float vT = ptT[1] / ptT[2];
			float KuT = fxl * uT + cxl;
			float KvT = fyl * vT + cyl;

			// translation only (negative)
			Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
			float uT2 = ptT2[0] / ptT2[2];
			float vT2 = ptT2[1] / ptT2[2];
			float KuT2 = fxl * uT2 + cxl;
			float KvT2 = fyl * vT2 + cyl;

			//translation and rotation (negative)
			Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
			float u3 = pt3[0] / pt3[2];
			float v3 = pt3[1] / pt3[2];
			float Ku3 = fxl * u3 + cxl;
			float Kv3 = fyl * v3 + cyl;

			//translation and rotation (positive)
			//already have it.

			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

//		if (fullSystem->isIMUinitialized() && lvl==4 && i==0)
//		{
//			std::cout 	<< Ku << " "
//						 << Kv << " "
//						 << new_idepth << " "
//						 << wl << " "
//						 << hl << " "
//						 << std::endl;
//		}

		if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0)) continue;



		float refColor = lpc_color[i];
		Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
		if(!std::isfinite((float)hitColor[0])) continue;
		float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

//		if (fullSystem->isIMUinitialized() && lvl==4 && i==0)
//		{
//			std::cout << "Point error: " << residual
//					  << " Hw: " << hw
//					  << std::endl;
//		}

		if(fabs(residual) > cutoffTH)
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
			E += maxEnergy;
			numTermsInE++;
			numSaturated++;
		}
		else
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));
			// information matrix (weight) based on pyramid level
			float lvl_info = 1.0;// / pow(2.0, (double)lvl);
			//residual * =lvl_info;
			E += (hw *residual*residual*lvl_info*lvl_info*(2-hw)) ;
			numTermsInE++;
			buf_warped_rx[numTermsInWarped] = pr(0);
			buf_warped_ry[numTermsInWarped] = pr(1);
			buf_warped_rz[numTermsInWarped] = pr(2);
			buf_warped_lpc_idepth[numTermsInWarped] = id;
			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw * lvl_info;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;
		}
	}

	while(numTermsInWarped%4!=0)
	{
		buf_warped_rx[numTermsInWarped] = 0;
		buf_warped_ry[numTermsInWarped] = 0;
		buf_warped_rz[numTermsInWarped] = 0;
		buf_warped_lpc_idepth[numTermsInWarped] = 0;
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;



	Vec15 imu_error = calcIMURes(current_navstate, biases);
	std::cout << "Before IMU error: " << imu_error.head<3>().transpose() << std::endl;
	imu_error.segment<9>(6) = Eigen::Matrix<double,9,1>::Zero();

	double IMUenergy = imu_error.transpose() * information_imu * imu_error;
	std::cout << "IMUenergy: " << IMUenergy << std::endl;
    // TODO: make threshold a setting
	float imu_huberTH = 21.666; // 1e5;
    if (IMUenergy > imu_huberTH)
    {
        float hw_imu = fabs(IMUenergy) < imu_huberTH ? 1 : imu_huberTH / fabs(IMUenergy);
        IMUenergy = hw_imu * IMUenergy * (2 - hw_imu);
        information_imu *= hw_imu;
    }


//	std::cout << "Normalized Residue: " << E / numTermsInE <<" numTermsInE:" <<numTermsInE<<" nl: " <<nl<<" IMUerror: "<<IMUenergy<< std::endl;

	std::cout<<"Unnormaliezd E is :"<<E << std::endl;
	E/=numTermsInE;
	IMUenergy/=SCALE_IMU_T;

	// orientation error of previous pose
	FrameShell *last_frame = fullSystem->getAllFrameHistory().back()->last_frame;
	gtsam::Rot3 rot3_diff = last_frame->groundtruth.pose.rotation().inverse().compose(
		last_frame->navstate.pose().rotation()
	);

	Vec3 ypr_diff = rot3_diff.ypr();
	double angle_error = sqrt( pow(ypr_diff(1), 2) + pow(ypr_diff(2), 2) ) * 100;

	std::cout<<"information_imu :\n"<<information_imu.diagonal().transpose()<<std::endl;
	std::cout<<"imu_error:\n"<<imu_error.transpose()<<std::endl;
	std::cout<<"number of points:"<<numTermsInE<<std::endl;
	std::cout<<"E vs IMU_error is :"
			 <<E <<" "
			 <<IMUenergy<<" "
			 <<angle_error<<" "
			 << lvl << std::endl;
	E += IMUenergy;
//=============================================================================================================
	if(debugPlot)
	{
		IOWrap::displayImage("RES", resImage, false);
		IOWrap::waitKey(0);
		delete resImage;
	}

	Vec6 rs;
	rs[0] = E;
	rs[1] = numTermsInE;
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
	rs[5] = numSaturated / (float)numTermsInE;

	return rs;
}


void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians)
{
	assert(frameHessians.size()>0);
	lastRef = frameHessians.back();
	makeCoarseDepthL0(frameHessians);



	refFrameID = lastRef->shell->id;
	lastRef_aff_g2l = lastRef->aff_g2l();

	firstCoarseRMSE=-1;

}
bool CoarseTracker::trackNewestCoarse(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out, SE3 &previousToNew_out, AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);


	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;

	SE3 previousToNew_current = previousToNew_out;
	SE3 refToNew_current = lastToNew_out;
	SE3 IMUToref_current = refToNew_current.inverse() * imutocam();
	SE3 previousToref = refToNew_current.inverse() * previousToNew_current;
	//SE3 IMUToref_current = refToIMU_current.inverse();
	//SE3 test = imutocam * IMUToref_current.inverse();
	//std::cout<<"test: \n"<< test.matrix()<<"refToNew_current:\n"<<refToNew_current.matrix()<<std::endl;
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;


	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		//std::cout<<"level: "<<lvl<<std::endl;
		Mat88 H; Vec8 b;
		float levelCutoffRepeat=1;
		Vec6 resOld = calcRes(lvl, refToNew_current, previousToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);
		//std::cout << "threshold: " << setting_coarseCutoffTH*levelCutoffRepeat << std::endl;
		//std::cout<<"resOld is: "<<resOld.transpose()<<std::endl;
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			//std::cout<<"cut off "<<std::endl;
			levelCutoffRepeat*=2;
			resOld = calcRes(lvl, refToNew_current, previousToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

            if(!setting_debugout_runquiet)
                printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}
		calcGSSSESingle(lvl, H, b, refToNew_current, aff_g2l_current);

		float lambda = 0.01;

		if(debugPrint)
		{
			Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					 0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
		}


		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			Mat88 Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Vec8 inc = Hl.ldlt().solve(-b);

//			std::cout << "Hl: \n" << Hl << std::endl;
//			std::cout << "b: \n" << b << std::endl;
//			std::cout << "lambda: " << lambda << std::endl;
//			std::cout << "increment: " << inc.transpose() << std::endl;

//			inc.setZero();
//			inc.head<8>() = Hl.topLeftCorner<8,8>().ldlt().solve(-b.head<8>());
//
//			std::cout << "increment: " << inc.transpose() << std::endl;

			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)	// fix a, b
			{
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
			 	inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
			{
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
			 	inc.tail<1>().setZero();
			}
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))	// fix a
			{
				Mat88 HlStitch = Hl;
				Vec8 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}




			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec8 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

			//std::cout<<"increment: \n"<<incScaled.transpose()<<std::endl;


            if(!std::isfinite(incScaled.sum())) incScaled.setZero();


			SE3 IMUToref_new = IMUToref_current * SE3::exp((Vec6)(incScaled.head<6>()));
			SE3 refToIMU_new = IMUToref_new.inverse();
			SE3 refToNew_new = imutocam() * refToIMU_new;
			SE3 previousToNew_new = refToNew_new * previousToref;

			//SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];

//			std::cout <<"lastRef->shell->navstate.pose()\n"<<lastRef->shell->navstate.pose().matrix()<<std::endl;
//			std::cout << "ref2New optimized (no imu): \n" << refToNew_new.matrix() << std::endl;

			Vec6 resNew = calcRes(lvl, refToNew_new, previousToNew_new, aff_g2l_new, setting_coarseCutoffTH*levelCutoffRepeat);

			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

			if(debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
			}
			if(accept)
			{
				calcGSSSESingle(lvl, H, b, refToNew_new, aff_g2l_new);
				resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				refToNew_current = refToNew_new;
				IMUToref_current = IMUToref_new;
				previousToNew_current = previousToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}

			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}

		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		lastFlowIndicators = resOld.segment<3>(2);
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;


		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}

	// set!
	lastToNew_out = refToNew_current;
	aff_g2l_out = aff_g2l_current;

	SE3 T_ref_new = refToNew_current.inverse();
	SE3 T_world_ref = lastRef->shell->camToWorld;
	SE3 T_world_new = T_world_ref * T_ref_new;
	SE3 nav_state = T_world_new * SE3(fullSystem->getTbc()).inverse();

//	std::cout<<" IMU free version: affine a: "<<aff_g2l_out.a<< " affine b: "<<aff_g2l_out.b<<std::endl;
//	std::cout << "IMU free navstate: " << nav_state.matrix() << std::endl;

	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	|| (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	|| (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;



	if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

	return true;
}

bool CoarseTracker::trackNewestCoarsewithIMU(
		FrameHessian* newFrameHessian, gtsam::NavState &navstate_out,
		Vec6 &biases_out , AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	newFrameHessian->shell->linearizeImuFactor(
			newFrameHessian->shell->last_frame->navstate,
			newFrameHessian->shell->navstate,
			newFrameHessian->shell->last_frame->bias
	);

	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

	gtsam::NavState navbak = navstate_out;
	AffLight aff_g2l_bak = aff_g2l_out;
	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;

	Vec6 biases_current = biases_out;
	gtsam::NavState navstate_current = navstate_out;
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;

//	std::cout<<"in tracking: lastRef id : "<<lastRef->shell->id<<std::endl;
//	std::cout<<"lastRef->Tib: "<<lastRef->shell->navstate.pose().matrix()<<std::endl;
//	std::cout<<"lastRef->Tib from camtoworld: "<<(lastRef->shell->camToWorld * SE3(fullSystem->getTbc()).inverse()).matrix()<<std::endl;

	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
//		std::cout<<"level: "<<lvl<<std::endl;
		Mat1717 H; Vec17 b;
		float levelCutoffRepeat=1;
		Vec6 resOld = calcResIMU(lvl, navstate_current, aff_g2l_current, biases_current, setting_coarseCutoffTH*levelCutoffRepeat);

		//std::cout << "threshold: " << setting_coarseCutoffTH*levelCutoffRepeat << std::endl;
		//std::cout<<"resOld is: "<<resOld.transpose()<<std::endl;
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			//std::cout<<"cut off!"<<std::endl;
			levelCutoffRepeat*=2;
			resOld = calcResIMU(lvl, navstate_current, aff_g2l_current, biases_current, setting_coarseCutoffTH*levelCutoffRepeat);

			if(!setting_debugout_runquiet)
				printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}
		//std::cout<<"resOld is: "<<resOld<<std::endl;
		calcGSSSESingleIMU(lvl, H, b, navstate_current, aff_g2l_current);

		float lambda = 0.01;


		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			Mat1717 Hl = H;
			for(int i=0;i<17;i++) Hl(i,i) *= (1+lambda);
			Vec17 inc;// = Hl.ldlt().solve(-b);

			// solve only vision terms
            inc.setZero();
			inc.head<11>() = Hl.topLeftCorner<11,11>().ldlt().solve(-b.head<11>());


			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)	// fix a, b
			{
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
				inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
			{
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
				inc.tail<1>().setZero();
			}
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))	// fix a
			{
				Mat1717 HlStitch = Hl;
				Vec17 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}




			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec17 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

			std::cout<<"increment: \n"<<incScaled.transpose()<<std::endl;


			if(!std::isfinite(incScaled.sum())) incScaled.setZero();

			SE3 IMUTow_new = SE3(navstate_current.pose().matrix()) * SE3::exp((Vec6)(incScaled.head<6>()));
			//std::cout<<"increment of velocity: "<<incScaled.segment<3>(8).transpose()<<std::endl;
			Vec3 velocity_new = navstate_current.velocity() + incScaled.segment<3>(8);
			gtsam::NavState navstate_new = gtsam::NavState(
					gtsam::Pose3(IMUTow_new.matrix()),
					velocity_new
			);

			// calculate relative pose with ref frame
			SE3 refToNew_new = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
					( SE3(lastRef->shell->navstate.pose().inverse().matrix()) * IMUTow_new ).matrix()
			)).inverse();
			//std::cout <<"lastRef->shell->navstate.pose()\n"<<lastRef->shell->navstate.pose().matrix()<<std::endl;
			//std::cout << "ref2new optimized: \n" << refToNew_new.matrix() << std::endl;

			//std::cout<<"increment of biases: "<<incScaled.tail<6>().transpose()<<std::endl;
			Vec6 biases_new = biases_current + incScaled.tail<6>();

//			SE3 wToIMU_new = IMUTow_new.inverse();
//			SE3 wToNew_new = imutocam * wToIMU_new;
//			SE3 refToNew_new = wToNew_new * Tw_ref;
//			SE3 previousToNew_new = wToNew_new * previousTow;
//			std::cout<<"IMUTow_new:\n"<<IMUTow_new.matrix()<<"\nImuTow_current:\n"<<ImuTow_current.matrix()<<std::endl;
//			//std::cout<<"wToIMU_new:\n"<<wToIMU_new.matrix()<<"\nwToIMU_current:\n"<<wToIMU_current.matrix()<<std::endl;
//			std::cout<<"wToNew_new:\n"<<wToNew_new.matrix()<<"\nwToNew_current:\n"<<wToNew_new.matrix()<<std::endl;
//			std::cout<<"refToNew_new:\n"<<refToNew_new.matrix()<<"\nrefToNew_old:\n"<<refToNew_current.matrix()<<std::endl;


			//SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];

			Vec6 resNew = calcResIMU(lvl, navstate_new, aff_g2l_new, biases_new, setting_coarseCutoffTH*levelCutoffRepeat);

			bool accept = resNew[0] < resOld[0];
					//= (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

			if(accept)
			{
				calcGSSSESingleIMU(lvl, H, b, navstate_new, aff_g2l_new);
				resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				biases_current = biases_new;
				navstate_current = navstate_new;
				lambda *= 0.5;

//				refToNew_current = refToNew_new;
//				ImuTow_current = IMUTow_new;
//				previousToNew_current = previousToNew_new;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}

			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}

		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		lastFlowIndicators = resOld.segment<3>(2);
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;


		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}

	// set!


//	lastToNew_out = refToNew_current;
	navstate_out = navstate_current;
	aff_g2l_out = aff_g2l_current;
	biases_out = biases_current;

//	std::cout<<" IMU version: affine a: "<< aff_g2l_out.a << "affine b: "<< aff_g2l_out.b<<std::endl;
//	std::cout<<" NAVSTATE: \n" << navstate_current.pose().matrix()<<std::endl;
//	std::cout<<" after affine a: " << aff_g2l_out.a << "affine b: "<< aff_g2l_out.b<<std::endl;
//	std::cout<< "\npreToworld:=\n "<<newFrame->shell->last_frame->navstate.pose().matrix()<<std::endl;
//	std::cout<< "\n(before optimization)NewToworld:=\n "<<navbak.pose().matrix()<<std::endl;
//	std::cout<< "\n(after optimization)NewToworld:=\n "<<navstate_out.pose().matrix()<<std::endl;
//	std::cout<< "\n(before optimization)perviousToNew_out:=\n "<<navbak.pose().inverse().matrix() * newFrame->shell->last_frame->navstate.pose().matrix()<<std::endl;
//	std::cout<< "\n(after optimization)perviousToNew_out:=\n "<<navstate_out.pose().inverse().matrix() * newFrame->shell->last_frame->navstate.pose().matrix()<<std::endl;
//	std::cout<<" \nperviousToworld_GT:=\n "<< newFrame->shell->last_frame->groundtruth.pose.matrix()<<std::endl;
//	std::cout<<" \nnewToworld_GT:=\n "<< newFrame->shell->groundtruth.pose.matrix()<<std::endl;
//	std::cout<<" \nperviousToNew_GT:=\n "<< newFrame->shell->groundtruth.pose.inverse().matrix() * newFrame->shell->last_frame->groundtruth.pose.matrix()<<std::endl;


	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	   || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	   || (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;



	if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

	return true;
}

void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;


	int lvl = 0;

	{
		std::vector<float> allID;
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			if(idepth[lvl][i] > 0)
				allID.push_back(idepth[lvl][i]);
		}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		if (n<=0) return;

		float minID_new = allID[(int)(n*0.05)];
		float maxID_new = allID[(int)(n*0.95)];

		float minID, maxID;
		minID = minID_new;
		maxID = maxID_new;
		if(minID_pt!=0 && maxID_pt!=0)
		{
			if(*minID_pt < 0 || *maxID_pt < 0)
			{
				*maxID_pt = maxID;
				*minID_pt = minID;
			}
			else
			{

				// slowly adapt: change by maximum 10% of old span.
				float maxChange = 0.3*(*maxID_pt - *minID_pt);

				if(minID < *minID_pt - maxChange)
					minID = *minID_pt - maxChange;
				if(minID > *minID_pt + maxChange)
					minID = *minID_pt + maxChange;


				if(maxID < *maxID_pt - maxChange)
					maxID = *maxID_pt - maxChange;
				if(maxID > *maxID_pt + maxChange)
					maxID = *maxID_pt + maxChange;

				*maxID_pt = maxID;
				*minID_pt = minID;
			}
		}


		MinimalImageB3 mf(w[lvl], h[lvl]);
		mf.setBlack();
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			int c = lastRef->dIp[lvl][i][0]*0.9f;
			if(c>255) c=255;
			mf.at(i) = Vec3b(c,c,c);
		}
		int wl = w[lvl];
		for(int y=3;y<h[lvl]-3;y++)
			for(int x=3;x<wl-3;x++)
			{
				int idx=x+y*wl;
				float sid=0, nid=0;
				float* bp = idepth[lvl]+idx;

				if(bp[0] > 0) {sid+=bp[0]; nid++;}
				if(bp[1] > 0) {sid+=bp[1]; nid++;}
				if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
				if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
				if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

				if(bp[0] > 0 || nid >= 3)
				{
					float id = ((sid / nid)-minID) / ((maxID-minID));
					mf.setPixelCirc(x,y,makeJet3B(id));
					//mf.at(idx) = makeJet3B(id);
				}
			}
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


        for(IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

		if(debugSaveImages)
		{
			char buf[1000];
			snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
			IOWrap::writeImage(buf,&mf);
		}

	}
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);


	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}





void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame)
{
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;


	// make coarse tracking templates for latstRef.
	int numItems = 0;

	for(FrameHessian* fh : frameHessians)
	{
		if(frame == fh) continue;

		SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
		Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
		Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

		for(PointHessian* ph : fh->pointHessians)
		{
			assert(ph->status == PointHessian::ACTIVE);
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled;
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;
			if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
			fwdWarpedIDDistFinal[u+w1*v]=0;
			bfsList1[numItems] = Eigen::Vector2i(u,v);
			numItems++;
		}
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}



void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++)
	{
		int bfsNum2 = bfsNum;
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2);
		bfsNum=0;

		if(k%2==0)
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}


void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

}
