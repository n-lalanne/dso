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
#include <cmath>
#include <boost/tuple/tuple.hpp>
#include <opencv2/core/eigen.hpp>
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

void CoarseTracker::setUnscaledGaussNewtonSys(Mat3232 H, Vec32 b)
{
	H_unscaled = H;
	b_unscaled = b;
}

void CoarseTracker::splitHessianForMarginalization(
		Mat1515 &Hbb,
		Mat1517 &Hbm,
		Mat1717 &Hmm,
		Vec15 &bb,
		Vec17 &bm
)
{
	Hmm.topLeftCorner<2, 2>() = H_unscaled.block<2, 2>(6, 6);
	Hmm.bottomRightCorner<15, 15>() = H_unscaled.bottomRightCorner<15, 15>();
	Hmm.bottomLeftCorner<15, 2>() = H_unscaled.block<15, 2>(17, 6);
	Hmm.topRightCorner<2, 15>() = H_unscaled.block<2, 15>(6, 17);

	Hbb.topLeftCorner<6, 6>() = H_unscaled.topLeftCorner<6, 6>();
	Hbb.bottomRightCorner<9, 9>()= H_unscaled.block<9, 9>(8, 8);
	Hbb.bottomLeftCorner<9, 6>() = H_unscaled.block<9, 6>(8, 0);
	Hbb.topRightCorner<6, 9>() = H_unscaled.block<6, 9>(0, 8);

	Hbm.topLeftCorner<6, 2>() = H_unscaled.block<6, 2>(0, 6);
	Hbm.bottomLeftCorner<9, 2>() = H_unscaled.block<9, 2>(8, 6);
	Hbm.topRightCorner<6, 15>() = H_unscaled.block<6, 15>(0, 17);
	Hbm.bottomRightCorner<9, 15>() = H_unscaled.block<9, 15>(8, 17);

	bm.head(2) = b_unscaled.segment<2>(6);
	bm.tail(15) = b_unscaled.tail(15);

	bb.head(6) = b_unscaled.head(6);
	bb.tail(9) = b_unscaled.segment<9>(8);
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
		//std::cout<<"produced "<<lpc_n<<"pixels on level"<<lvl<<std::endl;
	}

}

// order of states: t_j, R_j, a_j, b_j, v_j, ba_j, bg_j, t_i, R_i, v_i, ba_i, bg_i
void CoarseTracker::calcGSSSEDoubleIMU(int lvl, Mat3232 &H_out, Vec32 &b_out, const gtsam::NavState navState_previous, const gtsam::NavState navState_current, AffLight aff_g2l)
{
	acc.initialize();
	int n = buf_warped_n;
	SE3 Tib(navState_current.pose().matrix());
	SE3 Tw_reffromNAV = SE3(lastRef->shell->navstate.pose().matrix()) * imutocam().inverse();
	SE3 Tw_ref = lastRef->shell->camToWorld;
	Mat33 Rcb = imutocam().rotationMatrix();
	SE3 Trb = Tw_ref.inverse() * Tib;

	// refToNew for debug
	SE3 refToNew = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
			(navState_current.pose().inverse() * lastRef->shell->navstate.pose()).matrix()
	));

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
	}

	acc.finish();

	H_out.setZero();
	b_out.setZero();
	H_out.block<8,8>(0,0) = acc.H.topLeftCorner<8,8>().cast<double>();// * (1.0f/n);
	b_out.segment<8>(0) = acc.H.topRightCorner<8,1>().cast<double>();// * (1.0f/n);



    Mat1515 H_previous;
    Vec15 b_previous;
    Mat1515 J_imu_travb_previous;

    J_imu_travb_previous.setZero();
    J_imu_travb_previous.block<15, 3>(0, 0) = J_imu_Rt_previous.block<15, 3>(0, 3);
    J_imu_travb_previous.block<15, 3>(0, 3) = J_imu_Rt_previous.block<15, 3>(0, 0);
    J_imu_travb_previous.block<15, 3>(0, 6) = J_imu_v_previous.block<15, 3>(0, 0);
	J_imu_travb_previous.block<15, 6>(0, 9) = J_imu_bias_previous;

	// ------------------ don't ignore the cross terms in hessian between i and jth poses ------------------
    Mat1517 J_imu_travb_current;
    J_imu_travb_current.setZero();
    J_imu_travb_current.block<15, 3>(0, 0) = J_imu_Rt.block<15, 3>(0, 3);
    J_imu_travb_current.block<15, 3>(0, 3) = J_imu_Rt.block<15, 3>(0, 0);
    J_imu_travb_current.block<15, 3>(0, 8) = J_imu_v.block<15, 3>(0, 0);
	J_imu_travb_previous.block<15, 6>(0, 11) = J_imu_bias;

    Mat1532 J_imu_complete;
    J_imu_complete.leftCols(17) = J_imu_travb_current;
    J_imu_complete.rightCols(15) = J_imu_travb_previous;
//	std::cout<<"H_photometric of current pose:\n"<<H_out.block<8,8>(0,0)<<std::endl;
//	std::cout<<"H_imu of current pose:\n"<<(J_imu_complete.transpose() * information_imu * J_imu_complete).block<11, 11>(0,0)<<std::endl;
//	std::cout<<"H_imu of pervious pose:\n"<<(J_imu_complete.transpose() * information_imu * J_imu_complete).block<9, 9>(17,17)<<std::endl;

    H_out += J_imu_complete.transpose() * information_imu * J_imu_complete;
    b_out += J_imu_complete.transpose() * information_imu * res_imu;

    std::cout << "J_imu_bias_previous\n" << J_imu_bias_previous << std::endl;
    std::cout << "information_bias_previous\n" << information_imu.bottomRightCorner<6, 6>() << std::endl;
    std::cout << "hessian_bias_previous: \n" << H_out.rightCols<6>() << std::endl;

    std::cout << "J_imu_bias\n" << J_imu_bias << std::endl;
    std::cout << "hessian_bias: \n" << H_out.block<32, 6>(0, 11) << std::endl;

	// ----------- Prior factor ----------- //
	Mat1515 H_prior = J_prior.transpose() * information_prior * J_prior;
	Vec15 b_prior = J_prior.transpose() * information_prior * res_prior;

//	Mat1515 H_prior = fullSystem->Hprior;
//	Vec15 b_prior = fullSystem->bprior + fullSystem->Hprior * res_imu;


	// Becareful!! the block for pervious pose still contains affine a and b!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//std::cout<<"H_imu+photometric of pervious pose:\n"<<H_out.block<9, 9>(17,17)<<std::endl;
	if(fullSystem->addprior)
	{
		//std::cout<<"prior added here!"<<std::endl;
		H_out.bottomRightCorner<15, 15>() +=  H_prior;
		b_out.tail(15) += b_prior;
	}

//	std::cout	<< "H_prior: \n" << H_prior.topLeftCorner<9, 9>() << std::endl
//				<< "b_prior: " << b_prior.head(9).transpose() << std::endl
//				<< "prior infomation: " << information_prior.diagonal().head(9).transpose() << std::endl;
//
//	std::cout<<"H_imu+photometric+prior of pervious pose:\n"<<H_out.block<9, 9>(17,17)<<std::endl;

    // ------------------ ignore the cross terms in hessian between i and jth poses ------------------
//    H_previous.noalias() = J_imu_travb_previous.transpose() * information_imu * J_imu_travb_previous;
//    b_previous.noalias() = J_imu_travb_previous.transpose() * information_imu * res_imu;
//
//	Mat1717 H_current = J_imu_travb_current.transpose() * information_imu * J_imu_travb_current;
//	Vec17 b_current = J_imu_travb_current.transpose() * information_imu * res_imu;
//
//	H_out.topLeftCorner<17, 17>() += H_current;
//    H_out.bottomRightCorner<15, 15>() += H_previous;
//
//    b_out.head<17> += b_current;
//    b_out.tail<15> += b_previous;

    H_unscaled = H_out;
    b_unscaled = b_out;

    H_out.block<32,3>(0,0) *= SCALE_XI_ROT;
    H_out.block<32,3>(0,3) *= SCALE_XI_TRANS;
    H_out.block<32,1>(0,6) *= SCALE_A;
    H_out.block<32,1>(0,7) *= SCALE_B;
    H_out.block<32,3>(0,8) *= SCALE_IMU_V;
	H_out.block<32,3>(0,11) *= SCALE_IMU_ACCE;
	H_out.block<32,3>(0,14) *= SCALE_IMU_GYRO;
    H_out.block<32,3>(0,17) *= SCALE_XI_ROT;
    H_out.block<32,3>(0,20) *= SCALE_XI_TRANS;
    H_out.block<32,3>(0,23) *= SCALE_IMU_V;
	H_out.block<32,3>(0,26) *= SCALE_IMU_ACCE;
	H_out.block<32,3>(0,29) *= SCALE_IMU_GYRO;


	H_out.block<3,32>(0,0) *= SCALE_XI_ROT;
    H_out.block<3,32>(3,0) *= SCALE_XI_TRANS;
    H_out.block<1,32>(6,0) *= SCALE_A;
    H_out.block<1,32>(7,0) *= SCALE_B;
    H_out.block<3,32>(8,0) *= SCALE_IMU_V;
	H_out.block<3,32>(11,0) *= SCALE_IMU_ACCE;
	H_out.block<3,32>(14,0) *= SCALE_IMU_GYRO;
    H_out.block<3,32>(17,0) *= SCALE_XI_ROT;
    H_out.block<3,32>(20,0) *= SCALE_XI_TRANS;
    H_out.block<3,32>(23,0) *= SCALE_IMU_V;
	H_out.block<3,32>(26,0) *= SCALE_IMU_ACCE;
	H_out.block<3,32>(29,0) *= SCALE_IMU_GYRO;

	b_out.segment<3>(0) *= SCALE_XI_ROT;
    b_out.segment<3>(3) *= SCALE_XI_TRANS;
    b_out.segment<1>(6) *= SCALE_A;
    b_out.segment<1>(7) *= SCALE_B;
    b_out.segment<3>(8) *= SCALE_IMU_V;
	b_out.segment<3>(11) *= SCALE_IMU_ACCE;
	b_out.segment<3>(14) *= SCALE_IMU_GYRO;

    b_out.segment<3>(17) *= SCALE_XI_ROT;
    b_out.segment<3>(20) *= SCALE_XI_TRANS;
    b_out.segment<3>(23) *= SCALE_IMU_V;
	b_out.segment<3>(26) *= SCALE_IMU_ACCE;
	b_out.segment<3>(29) *= SCALE_IMU_GYRO;
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
//		}11

	}

	acc.finish();

    H_out.setZero();
    b_out.setZero();
	H_out.block<8,8>(0,0) = acc.H.topLeftCorner<8,8>().cast<double>();// * (1.0f/n);
	b_out.segment<8>(0) = acc.H.topRightCorner<8,1>().cast<double>();// * (1.0f/n);
	//std::cout<<"H_out:\n"<<H_out.block<8,8>(0,0)<<std::endl;
	//std::cout<<"b_out:\n"<<b_out.segment<8>(0)<<std::endl;
	// here rtvab means rotation, translation, affine, velocity and biases

//	std::cout << "H_out: \n" << H_out.topLeftCorner<11,11>() << std::endl;
//	std::cout << "b_out: \n" << b_out.segment<11>(0).transpose() << std::endl;

    Mat1717 H_imu_travb;
	Vec17 b_imu_travb;
	Mat1517 J_imu_travb;

	J_imu_travb.setZero();
	// delta r/ navstate j trv
	J_imu_travb.block<15, 3>(0, 0) = J_imu_Rt.block<15, 3>(0, 3);
	J_imu_travb.block<15, 3>(0, 3) = J_imu_Rt.block<15, 3>(0, 0);
    J_imu_travb.block<15, 3>(0, 8) = J_imu_v.block<15, 3>(0, 0);
	J_imu_travb.block<15, 6>(0, 11) = J_imu_bias_previous; // J_imu_bias;

//	// delta t/ navstate j trv
//	J_imu_travb.block<3, 3>(3, 0) = J_imu_Rt.block<3, 3>(3, 3);
//	J_imu_travb.block<3, 3>(3, 3) = J_imu_Rt.block<3, 3>(3, 0);
//	J_imu_travb.block<3, 3>(3, 8) = J_imu_v.block<3, 3>(3, 0);
//
//	// delta v/ navstate j trv
//	J_imu_travb.block<3, 3>(6, 0) = J_imu_Rt.block<3, 3>(6, 3);
//	J_imu_travb.block<3, 3>(6, 3) = J_imu_Rt.block<3, 3>(6, 0);
//	J_imu_travb.block<3, 3>(6, 8) = J_imu_v.block<3, 3>(6, 0);

	// set the res vector


    //computing blocks for velocity independently
//    Mat33 H_imu_v = J_imu_v.transpose() * information_imu * J_imu_v;
//    Vec3 b_imu_v = J_imu_v.transpose() * information_imu * res_imu;

	//computing b vector and hessian matrix


    //information_imu.topLeftCorner<3, 3>() /= SCALE_IMU_R;
    //information_imu.block<3,3>(3, 3) /= SCALE_IMU_T;

	H_imu_travb = J_imu_travb.transpose() * information_imu * J_imu_travb;
//    H_imu_travb.block<3,3>(6,6) = H_imu_v;
	b_imu_travb = J_imu_travb.transpose() * information_imu * res_imu;
//    b_imu_travb.segment<3>(6) = b_imu_v;

//  H_imu_travb.block<8,3>(0,0) *= SCALE_IMU_T;
//  H_imu_travb.block<8,3>(0,3) *= SCALE_IMU_R;
//  H_imu_travb.block<3,8>(0,0) *= SCALE_IMU_T;
//  H_imu_travb.block<3,8>(3,0) *= SCALE_IMU_R;
//	H_imu_travb.block<3,8>(8,0) *= SCALE_IMU_V;
//	H_imu_travb.block<8,3>(0,8) *= SCALE_IMU_V;
//    H_imu_travb.block<8,8>(0,0) *= 0.01;
//    b_imu_travb.segment<8>(0) *= 0.01;
//
//    H_imu_travb.block<3,8>(8,0) *= SCALE_IMU_V;
//	H_imu_travb.block<8,3>(0,8) *= SCALE_IMU_V;
//	b_imu_travb.segment<3>(8) *= SCALE_IMU_V;

	//H_imu_rtavb /= 1000.0;
	//b_imu_rtavb /= 1000.0;

//    std::cout << "J_imu: \n " << J_imu_travb << std::endl;
    //std::cout<<" res_imu:\n"<<res_imu.transpose()<<std::endl;
    //std::cout<<" information_imu: \n"<<information_imu.diagonal().transpose()<<std::endl;
//	std::cout<<"J_imu\n"<<J_imu_travb.transpose()<<std::endl;
//	std::cout<<"information:\n"<<information_imu.diagonal().transpose()<<std::endl;
//	std::cout << "H_imu: \n" << H_imu_travb.topLeftCorner<11,11>() << std::endl;
//	std::cout << "b_imu: \n" << b_imu_travb.segment<11>(0).transpose() << std::endl;
//
//    std::cout << "before adding: H_out: \n" << H_out.topLeftCorner<11,11>() << std::endl;
//    std::cout << "before adding: b_out: \n" << b_out.segment<11>(0).transpose() << std::endl;

	if(fullSystem->addimu)
	{
		H_out += H_imu_travb;
		b_out += b_imu_travb;
	}
	//std::cout << "before rescale: H_out: \n" << H_out.topLeftCorner<11,11>() << std::endl;
	//std::cout << "before rescale: b_out: \n" << b_out.segment<11>(0).transpose() << std::endl;

    H_unscaled.topLeftCorner<17, 17>() = H_out;
    b_unscaled.head(17) = b_out;

	H_out.block<17,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<17,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<17,1>(0,6) *= SCALE_A;
	H_out.block<17,1>(0,7) *= SCALE_B;
	H_out.block<3,17>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,17>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,17>(6,0) *= SCALE_A;
	H_out.block<1,17>(7,0) *= SCALE_B;
	H_out.block<3,17>(8,0) *= SCALE_IMU_V;
	H_out.block<17,3>(0,8) *= SCALE_IMU_V;
	H_out.block<17,3>(0,11) *= SCALE_IMU_ACCE;
	H_out.block<17,3>(0,14) *= SCALE_IMU_GYRO;
	H_out.block<3,17>(11,0) *= SCALE_IMU_ACCE;
	H_out.block<3,17>(14,0) *= SCALE_IMU_GYRO;

	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
	b_out.segment<3>(8) *= SCALE_IMU_V;
	b_out.segment<3>(11) *= SCALE_IMU_ACCE;
	b_out.segment<3>(14) *= SCALE_IMU_GYRO;


	std::cout << "H_out: \n" << H_out << std::endl;
	std::cout << "b_out: \n" << b_out.transpose() << std::endl;


//	H_out += H_imu_travb;
//	b_out += b_imu_travb;

//	std::cout << "H_final: \n" << H_out.topLeftCorner<11,11>() << std::endl;
//	std::cout << "b_final: \n" << b_out.segment<11>(0).transpose() << std::endl;
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



Vec15 CoarseTracker::calcIMURes(gtsam::NavState previous_navstate, gtsam::NavState current_navstate, Vec6 prev_bias, Vec6 bias)
{
	information_imu = newFrame->shell->getIMUcovariance().inverse();

	// useless Jacobians of reference frame (cuz we're not optimizing reference frame)
	gtsam::Matrix  J_imu_Rt_i, J_imu_v_i, J_imu_bias_i;
	//newFrame->shell->velocity << 1, 1, 1;
	gtsam::imuBias::ConstantBias last_bias(prev_bias);
	gtsam::imuBias::ConstantBias curr_bias(bias);

    res_imu = newFrame->shell->evaluateIMUerrors(
			previous_navstate,
			current_navstate,
			last_bias,
			curr_bias,
			J_imu_Rt_previous, J_imu_v_previous, J_imu_Rt, J_imu_v, J_imu_bias_previous, this->J_imu_bias
	);

    return res_imu;
}

Vec15 CoarseTracker::calcPriorRes(gtsam::NavState previous_navstate, Vec6 previous_bias)
{
	res_prior.setZero();
	J_prior.setZero();
	information_prior.setZero();

	//return res_prior;

	//delta T
	res_prior.segment<3>(0) = fullSystem->navstatePrior.pose().translation() - previous_navstate.pose().translation();
	//delta R
	res_prior.segment<3>(3) = gtsam::Rot3::Logmap(fullSystem->navstatePrior.pose().rotation().inverse() * previous_navstate.pose().rotation());
	//delta V
	res_prior.segment<3>(6) = fullSystem->navstatePrior.velocity() - previous_navstate.velocity();
    res_prior.segment<6>(9) = fullSystem->biasPrior - previous_bias;
	// eR = log(R_prior^-1 * R_est)

	//J_prior = Matrix<double,9,9>::Zero();
	J_prior.block<3,3>(0,0) = -previous_navstate.pose().rotation().matrix();
	J_prior.block<3,3>(3,3) = Rot3::LogmapDerivative(res_prior.segment<3>(3));
	J_prior.block<3,3>(6,6) = -Mat33::Identity();
    J_prior.block<6,6>(9,9) = -Mat66::Identity();

	information_prior = fullSystem->Hprior; // .diagonal().asDiagonal();

	return res_prior;

//	gtsam::Matrix3 D_dR_R, D_dt_R, D_dv_R;
//	const Rot3 dR = previous_navstate.pose().rotation().between(
//			fullSystem->navstatePrior.pose().rotation(),
//			&D_dR_R
//	);
//	gtsam::Point3 dt = previous_navstate.pose().rotation().unrotate(
//			fullSystem->navstatePrior.pose().translation() - previous_navstate.pose().translation(),
//			&D_dt_R
//	);
//	gtsam::Vector dv = previous_navstate.pose().rotation().unrotate(
//			fullSystem->navstatePrior.velocity() - previous_navstate.velocity(),
//			&D_dv_R
//	);
//
//	gtsam::Vector9 xi;
//	gtsam::Matrix3 D_xi_R;
//	res_prior.head(3) = dt.vector();
//	res_prior.segment<3>(3) = gtsam::Rot3::Logmap(dR, &D_xi_R);
//	res_prior.segment<3>(6) = dv;
//
//    // Separate out derivatives
//    // Note that doing so requires special treatment of velocities, as when treated as
//    // separate variables the retract applied will not be the semi-direct product in NavState
//    // Instead, the velocities in nav are updated using a straight addition
//    // This is difference is accounted for by the R().transpose calls below
//
//    // diagonals
//	J_prior.topLeftCorner<3, 3>() = -Mat33::Identity();
//	J_prior.block<3, 3>(3, 3) = D_xi_R * D_dR_R;
//	J_prior.block<3, 3>(6, 6) = -previous_navstate.R().transpose();
//	// off-diagonals
//	// t, R
//	J_prior.block<3, 3>(0, 3) = D_dt_R;
//	// v, R
//	J_prior.block<3, 3>(6, 3) = D_dv_R;
//
//	// TODO: bias error
//
//	information_prior = fullSystem->Hprior.diagonal().asDiagonal();
//	// more reduction for R ant t
////	information_prior.topLeftCorner<6, 6>() = setting_priorFactorWeight * information_prior.topLeftCorner<6, 6>();
////	information_prior.block<3, 3>(0, 3) = setting_priorFactorWeight * information_prior.block<3, 3>(0, 3);
////	information_prior.block<3, 3>(3, 0) = setting_priorFactorWeight * information_prior.block<3, 3>(3, 0);
//
//	return res_prior;
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



Vec6 CoarseTracker::calcResIMU(int lvl, const gtsam::NavState previous_navstate, const gtsam::NavState current_navstate, AffLight aff_g2l, const Vec6 prev_biases, const Vec6 biases, float cutoffTH)
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

	SE3 refToNew_gt = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
			(newFrame->shell->groundtruth.pose.inverse() * lastRef->shell->groundtruth.pose).matrix()
	));

	//std::cout << "Ref to new error: " << (refToNew_gt.inverse() * refToNew).log().transpose() << std::endl;

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

	if(lvl == 0 && nl < 50){
		std::cout<<"not enough depth!"<<std::endl;
		exit(1);
	}

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
			//residual * =lvl_info;
			E += (setting_visionFactorWeight * hw * residual * residual * (2-hw)) ;
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
			buf_warped_weight[numTermsInWarped] = hw * setting_visionFactorWeight;
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



	std::cout<<"----------------------------------------------------------------"<<std::endl;
	Vec15 imu_error = calcIMURes(previous_navstate, current_navstate, prev_biases, biases);
	//std::cout << "Before IMU error: " << imu_error.transpose() << std::endl;
	//imu_error.segment<6>(9) = Eigen::Matrix<double,6,1>::Zero();

	double IMUenergy = imu_error.transpose() * information_imu * imu_error;
	//std::cout << "imu_error: " << imu_error.transpose() << std::endl;
    // TODO: make threshold a setting
	float imu_huberTH = 50;//21.66;
	//std::cout<<"IMUenergy(uncut): "<<IMUenergy<<std::endl;
	//std::cout<<"information_imu:(uncut)"<<information_imu.diagonal().transpose()<<std::endl;

	if(fabs(imu_error(8))>0.5||fabs(imu_error(7))>0.5||fabs(imu_error(6))>0.5){
		std::cout<<" wrong imu_error!!!!"<<std::endl;
//		exit(0);
	}

    if (IMUenergy > imu_huberTH)
    {
        float hw_imu = fabs(IMUenergy) < imu_huberTH ? 1 : imu_huberTH / fabs(IMUenergy);
        IMUenergy = hw_imu * IMUenergy * (2 - hw_imu);
        information_imu *= hw_imu;
    }

	if (!std::isfinite(IMUenergy))
	{
		IMUenergy = 0;
		information_imu.setZero();
		//std::cout << "undefined imu energy" << std::endl;
	}

	// -------------------------------------------------- Prior factor -------------------------------------------------- //
	std::cout<<"last v:"<<previous_navstate.v().transpose()<<std::endl;
	std::cout<<"current v:"<<current_navstate.v().transpose()<<std::endl;
	res_prior = calcPriorRes(previous_navstate, prev_biases);


	//std::cout << "Res prior: " << res_prior.transpose() << std::endl;
	//std::cout<<"information_prior:(uncut)"<<information_prior.diagonal().transpose()<<std::endl;

//	Eigen::LLT<Eigen::MatrixXd> lltOfA(information_prior); // compute the Cholesky decomposition of A
//	if(lltOfA.info() == Eigen::NumericalIssue)
//	{
//		std::cout<<"Possibly non semi-positive definitie information matrix!"<<std::endl;
//	}

	double priorEnergy = res_prior.transpose() * information_prior * res_prior;
	double priorr = res_prior.head(3).transpose()*information_prior.block<3,3>(0,0) * res_prior.head(3);
	double priort = res_prior.segment<3>(3).transpose()*information_prior.block<3,3>(3,3) * res_prior.segment<3>(3);
	double priorv = res_prior.segment<3>(6).transpose()*information_prior.block<3,3>(6,6) * res_prior.segment<3>(6);
	std::cout<<"er "<<priorr<<" et "<<priort<<" priorv "<<priorv<<std::endl;
	if(priorEnergy<0.0)std::cout<<"priorEnergy is negative!!!"<<std::endl;
	// TODO: make threshold a setting
	float prior_huberTH = 21.66;//50;
	std::cout<<"information(uncut): "<<information_prior.diagonal().transpose()<<std::endl;
	std::cout<<"res_prior: "<<res_prior<<std::endl;
	std::cout<<"priorEnergy(uncut): "<<priorEnergy<<std::endl;

	if (priorEnergy > prior_huberTH)
	{
		//std::cout<<"information_prior needs to be cut off"<<std::endl;
		float hw_res = fabs(priorEnergy) < prior_huberTH ? 1 : prior_huberTH / fabs(priorEnergy);
		priorEnergy = hw_res * priorEnergy * (2 - hw_res);
		information_prior *= hw_res;
	}
	std::cout<<"priorEnergy(cut): "<<priorEnergy<<std::endl;

	if (!std::isfinite(priorEnergy))
	{
		priorEnergy = 0;
		information_prior.setZero();
		//std::cout << "undefined prior energy" << std::endl;
	}

	if(fabs(res_prior(8))>0.5||fabs(res_prior(7))>0.5||fabs(res_prior(6))>0.5){
		std::cout<<" wrong res_prior!!!!"<<std::endl;
//		exit(0);
	}

//	std::cout << "Normalized Residue: " << E / numTermsInE <<" numTermsInE:" <<numTermsInE<<" nl: " <<nl<<" IMUerror: "<<IMUenergy<< std::endl;
	// -------------------------------------------------- Prior factor -------------------------------------------------- //
    //E/=numTermsInE;
	//std::cout<<nl<<"pixels with depth avaliable :"<<numTermsInE<<" inliers, vision error is "<< E << std::endl;
	//IMUenergy/=SCALE_IMU_T;


	//std::cout<<"information_imu :\n"<<information_imu.diagonal().transpose()<<std::endl;
	//std::cout<<"imu_error:\n"<<imu_error.transpose()<<std::endl;
	//std::cout<<"number of points:"<<numTermsInE<<std::endl;
	std::cout<<"E vs IMU_error vs priorEnergy  is :"<<E <<" "<<IMUenergy<< " " <<priorEnergy<<" "<< lvl << std::endl;

	// calculate error wrt gt
	Vec6 pose_error_current =  gtsam::Pose3::Logmap(gtsam::Pose3(
			(fullSystem->T_dsoworld_eurocworld * newFrame->shell->groundtruth.pose.matrix()).inverse() * current_navstate.pose().matrix()
	));
	Vec3 velocity_error_current = current_navstate.velocity() - fullSystem->T_dsoworld_eurocworld.topLeftCorner(3, 3) * newFrame->shell->groundtruth.velocity;

	Vec6 pose_error_previous =  gtsam::Pose3::Logmap(gtsam::Pose3(
			(fullSystem->T_dsoworld_eurocworld * newFrame->shell->last_frame->groundtruth.pose.matrix()).inverse() * previous_navstate.pose().matrix()
	));
	Vec3 velocity_error_previous = previous_navstate.velocity() - fullSystem->T_dsoworld_eurocworld.topLeftCorner(3, 3) * newFrame->shell->last_frame->groundtruth.velocity;

	//std::cout << "error_j: " << pose_error_current.transpose() << " " << velocity_error_current.transpose() << std::endl;
	//std::cout << "error_i: " << pose_error_previous.transpose() << " " << velocity_error_previous.transpose() << std::endl;


    //std::cout<<"----------------------------------------------------------------"<<std::endl;
//	E += IMUenergy;
	if(fullSystem->addimu)
	{
		E += IMUenergy;
	}
	if(fullSystem->addprior)
	{
		E += priorEnergy;
	}
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
		FrameHessian* newFrameHessian, gtsam::NavState &navstate_i_out, gtsam::NavState &navstate_out, Vec6 &pbiases_out,
		Vec6 &biases_out , AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

	AffLight aff_g2l_bak = aff_g2l_out;
	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;
	std::cout<<"The bias before tracking"<<biases_out<<std::endl;
	Vec6 biases_current = biases_out;
	Vec6 pbiases_current = pbiases_out;
	gtsam::imuBias::ConstantBias pbiases_first_estimate(pbiases_out);
	gtsam::imuBias::ConstantBias biases_current_estimate(biases_out);

	gtsam::NavState navstate_j_current_bak = navstate_out;
	gtsam::NavState navstate_i_current_bak = navstate_i_out;

	gtsam::NavState navstate_j_current = navstate_out;
	gtsam::NavState navstate_i_current = navstate_i_out;
	gtsam::NavState navstate_i_first_estimate = navstate_i_current;
	AffLight aff_g2l_current = aff_g2l_out;

	gtsam::NavState navstate_i_gt = gtsam::NavState(
			gtsam::Pose3(fullSystem->T_dsoworld_eurocworld).compose(newFrame->shell->last_frame->groundtruth.pose),
			fullSystem->T_dsoworld_eurocworld.topLeftCorner(3, 3) * newFrame->shell->last_frame->groundtruth.velocity
	);
	gtsam::NavState navstate_j_gt = gtsam::NavState(
			gtsam::Pose3(fullSystem->T_dsoworld_eurocworld).compose(newFrame->shell->groundtruth.pose),
			fullSystem->T_dsoworld_eurocworld.topLeftCorner(3, 3) * newFrame->shell->groundtruth.velocity
	);

	gtsam::NavState navstate_last_ref_gt = gtsam::NavState(
			gtsam::Pose3(fullSystem->T_dsoworld_eurocworld).compose(lastRef->shell->groundtruth.pose),
			fullSystem->T_dsoworld_eurocworld.topLeftCorner(3, 3) * lastRef->shell->groundtruth.velocity
	);
	std::cout << "Last ref error: "
			  << gtsam::Pose3::Logmap(navstate_last_ref_gt.pose().inverse().compose(lastRef->shell->navstate.pose())).transpose()
			  << " " << (lastRef->shell->navstate.velocity() - navstate_last_ref_gt.velocity()).transpose()
			  << std::endl;

	bool haveRepeated = false;

//	std::cout<<"in tracking: lastRef id : "<<lastRef->shell->id<<std::endl;
//	std::cout<<"lastRef->Tib: "<<lastRef->shell->navstate.pose().matrix()<<std::endl;
//	std::cout<<"lastRef->Tib from camtoworld: "<<(lastRef->shell->camToWorld * SE3(fullSystem->getTbc()).inverse()).matrix()<<std::endl;

	Mat3232 H; Vec32 b;
	Mat1717 H17; Vec17 b17;

    // optimize only one pose if we the last frame is the last keyframe (map recently updated)
    bool isOptimizeSingle = (lastRef->shell == newFrameHessian->shell->last_frame);

	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
//		std::cout<<"level: "<<lvl<<std::endl;
		// linearize the imu factor (for first estimate jacobian)
		newFrame->shell->linearizeImuFactorLastFrame(
				navstate_i_first_estimate,
				navstate_j_current,
				pbiases_first_estimate,
				biases_current_estimate
		);
        H.setZero(); b.setZero();

		float levelCutoffRepeat=1;
		Vec6 resOld = calcResIMU(lvl, navstate_i_current, navstate_j_current, aff_g2l_current, pbiases_current, biases_current, setting_coarseCutoffTH*levelCutoffRepeat);

		//std::cout << "threshold: " << setting_coarseCutoffTH*levelCutoffRepeat << std::endl;
		//std::cout<<"resOld is: "<<resOld.transpose()<<std::endl;
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			//std::cout<<"cut off!"<<std::endl;
			levelCutoffRepeat*=2;
			resOld = calcResIMU(lvl, navstate_i_current, navstate_j_current, aff_g2l_current, pbiases_current, biases_current, setting_coarseCutoffTH*levelCutoffRepeat);

			if(!setting_debugout_runquiet)
				printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		//std::cout<<"resOld is: "<<resOld<<std::endl;
        if (isOptimizeSingle||!fullSystem->addimu)
        {
            calcGSSSESingleIMU(lvl, H17, b17, navstate_j_current, aff_g2l_current);
            H.topLeftCorner<17, 17>() = H17;
            b.head<17>() = b17;
        }
        else
        {
            calcGSSSEDoubleIMU(lvl, H, b, navstate_i_current, navstate_j_current, aff_g2l_current);
        }

		float lambda = 0.01;

		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			Mat3232 Hl = H;
			for(int i=0;i<H.rows() ;i++) Hl(i,i) *= (1+lambda);
			Vec32 inc = Vec32::Zero();

			//std::cout << "lambda: " << lambda << std::endl;

//			if (!fullSystem->addimu)
//			{
//				// remove the bias blocks
//				inc.head<8>() = Hl.topLeftCorner<8, 8>().ldlt().solve(-b.head<8>());
////                inc.head<17>() = Hl.topLeftCorner<17, 17>().ldlt().solve(-b.head<17>());
//			}

            if(isOptimizeSingle)
            {
				// remove the bias blocks
				inc.head<11>() = Hl.topLeftCorner<11, 11>().ldlt().solve(-b.head<11>());
                // inc.head<17>() = Hl.topLeftCorner<17, 17>().ldlt().solve(-b.head<17>());
            }
            else
            {
//				// remove the bias blocks
//				Eigen::Matrix<double, 20, 20> H_no_bias;
//				// diagonals
//				H_no_bias.topLeftCorner<11, 11>() = Hl.topLeftCorner<11, 11>();
//				H_no_bias.bottomRightCorner<9, 9>() = Hl.block<9, 9>(17, 17);
//				// off-diagonals
//				H_no_bias.topRightCorner<11, 9>() =Hl.block<11, 9>(0, 17);
//				H_no_bias.bottomLeftCorner<9, 11>() = Hl.block<9, 11>(17, 0);
//
//				Eigen::Matrix<double, 20, 1> b_no_bias;
//				b_no_bias.head(11) = b.head(11);
//				b_no_bias.tail(9) = b.segment<9>(17);
//
//				Eigen::Matrix<double, 20, 1> inc_temp = H_no_bias.ldlt().solve(-b_no_bias);
//
//				inc.head(11) = inc_temp.head(11);
//				inc.segment<9>(17) = inc_temp.tail(9);

				//std::cout << "Initial increment: " << inc_temp.transpose() << std::endl;

                inc = Hl.ldlt().solve(-b);
            }

            float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec32 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;
            incScaled.segment<3>(8) *= SCALE_IMU_V;
			incScaled.segment<3>(11) *= SCALE_IMU_ACCE;
			incScaled.segment<3>(14) *= SCALE_IMU_GYRO;

			if(!isOptimizeSingle)
			{
				incScaled.segment<3>(17) *= SCALE_XI_ROT;
				incScaled.segment<3>(20) *= SCALE_XI_TRANS;
				incScaled.segment<3>(23) *= SCALE_IMU_V;
				incScaled.segment<3>(26) *= SCALE_IMU_ACCE;
				incScaled.segment<3>(29) *= SCALE_IMU_GYRO;
			}
			//std::cout<<"increment_j: \n"<<incScaled.head(11).transpose()<<std::endl;
            //std::cout<<"increment_i: \n"<<incScaled.segment<9>(17).transpose()<<std::endl;

            if(!std::isfinite(incScaled.sum())) incScaled.setZero();

			SE3 IMUTow_j_new = SE3(navstate_j_current.pose().matrix()) * SE3::exp((Vec6)(incScaled.head<6>()));
            Vec3 velocity_j_new = navstate_j_current.velocity() + incScaled.segment<3>(8);

            SE3 IMUTow_i_new = SE3(navstate_i_current.pose().matrix()) * SE3::exp((Vec6)(incScaled.segment<6>(17)));
            Vec3 velocity_i_new = navstate_i_current.velocity() + incScaled.segment<3>(23);

            gtsam::NavState navstate_j_new = gtsam::NavState(
					gtsam::Pose3(IMUTow_j_new.matrix()),
					velocity_j_new
			);

            gtsam::NavState navstate_i_new = gtsam::NavState(
                    gtsam::Pose3(IMUTow_i_new.matrix()),
                    velocity_i_new
            );

			// calculate relative pose with ref frame
			SE3 refToNew_new = SE3(dso_vi::IMUData::convertRelativeIMUFrame2RelativeCamFrame(
					( SE3(lastRef->shell->navstate.pose().inverse().matrix()) * IMUTow_j_new ).matrix()
			)).inverse();

			//std::cout <<"lastRef->shell->navstate.pose()\n"<<lastRef->shell->navstate.pose().matrix()<<std::endl;
			//std::cout << "ref2new optimized: \n" << refToNew_new.matrix() << std::endl;

			//std::cout<<"increment of biases: "<<incScaled.tail<6>().transpose()<<std::endl;
			Vec6 pbiases_new = pbiases_current + incScaled.tail<6>();
			Vec6 biases_new = pbiases_new; // = biases_current + incScaled.segment<6>(11);
			//biases_new.setZero();
			//std::cout<<"Bias increment is:"<<incScaled.segment<6>(11).transpose()<<std::endl;

			//pbiases_new.setZero();
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];


			// linearize the imu factor (for first estimate jacobian)
			newFrame->shell->linearizeImuFactorLastFrame(
					navstate_i_first_estimate,
					navstate_j_new,
					pbiases_first_estimate,
					gtsam::imuBias::ConstantBias(biases_new)
			);
			Vec6 resNew = calcResIMU(lvl, navstate_i_new, navstate_j_new, aff_g2l_new, pbiases_new, biases_new, setting_coarseCutoffTH*levelCutoffRepeat);

			bool accept = resNew[0] < resOld[0];
					//= (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

            FrameShell::GroundtruthError previous_gt_error = newFrame->shell->last_frame->getGroundtruthError(navstate_i_new, pbiases_new);
            FrameShell::GroundtruthError current_gt_error = newFrame->shell->getGroundtruthError(navstate_j_new, biases_new);

            std::cout << "Previous state error: \n" << previous_gt_error	<< std::endl;
            std::cout << "Current state error: \n"  << current_gt_error		<< std::endl;
			std::cout << "IMU Res: \n"				<< res_imu.transpose()	<< std::endl;

            if(accept)
			{
				//std::cout<<"resNew[0] : resOld[0] "<<resNew[0] <<" : " <<resOld[0]<<" ,accept this incre"<<std::endl;
//				if(!fullSystem->addimu)
//				{
//					calcGSSSESingleIMU(lvl, H17, b17, navstate_j_current, aff_g2l_current);
//					H.topLeftCorner<8, 8>() = H17.block<8,8>(0,0);
//					b.head<8>() = b17.segment<8>(0);
//				}
                resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				biases_current = biases_new;
				std::cout<<"Current bias is :"<<biases_current.transpose()<<std::endl;
				// TODO: don't update this for single pose optimization
				pbiases_current = pbiases_new;

				if (!isOptimizeSingle)
                {
                    navstate_i_current = navstate_i_new;
                }
				navstate_j_current = navstate_j_new;


				if (isOptimizeSingle)
				{
					calcGSSSESingleIMU(lvl, H17, b17, navstate_j_current, aff_g2l_current);
					H.topLeftCorner<17, 17>() = H17;
					b.head<17>() = b17;
				}
				else
				{
					calcGSSSEDoubleIMU(lvl, H, b, navstate_i_current, navstate_j_current, aff_g2l_current);
				}

				biases_current_estimate = gtsam::imuBias::ConstantBias(biases_current);
				lambda *= 0.5;


//				refToNew_current = refToNew_new;
//				ImuTow_current = IMUTow_j_new;
//				previousToNew_current = previousToNew_new;
			}
			else
			{
				std::cout<<"resNew[0] : resOld[0] "<<resNew[0] <<" : " <<resOld[0]<<" ,reject this incre"<<std::endl;
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;

				// rollback to last good estimate if not accepted
				newFrame->shell->linearizeImuFactorLastFrame(
						navstate_i_first_estimate,
						navstate_j_current,
						pbiases_first_estimate,
						biases_current_estimate
				);
			}

			// linearize the imu factor (for first estimate jacobian)
			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}

		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0]));
		lastFlowIndicators = resOld.segment<3>(2);
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;


		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}

		// res at groundtruth
//		std::cout << "Res at groundtruth: " << std::endl;
//		newFrame->shell->linearizeImuFactorLastFrame(
//				navstate_i_gt,
//				navstate_j_gt,
//				newFrame->shell->last_frame->bias,
//				newFrame->shell->bias
//		);
//
//		Vec6 res_gt = calcResIMU(
//				lvl,
//				navstate_i_gt,
//				navstate_j_gt,
//				aff_g2l_current, biases_current, 1
//		);
//		std::cout << "res gt: " << res_gt.transpose() << std::endl;
	}

	// set!

//	lastToNew_out = refToNew_current;


	navstate_out = navstate_j_current;
	navstate_i_out = navstate_i_current;
	aff_g2l_out = aff_g2l_current;
	biases_out = biases_current;
	pbiases_out = pbiases_current;

//	gtsam::NavState navstate_j_current_bak = navstate_out;
//	gtsam::NavState navstate_i_current_bak = navstate_i_out;

	if((navstate_j_current_bak.v() - navstate_out.v()).norm()>0.3||(navstate_i_current_bak.v()-navstate_j_current_bak.v()).norm()>0.3)
	{
		if(isOptimizeSingle)
		{
			std::cout << "isOptimizeSingle" << std::endl;
		}
		std::cout<<"before navi:"<<navstate_i_current_bak<<std::endl;
		std::cout<<"before navj:"<<	navstate_j_current_bak<<std::endl;
		std::cout<<"after navi:"<<	navstate_i_out<<std::endl;
		std::cout<<"after navj:"<<	navstate_out<<std::endl;

		std::cout<<"velocity increment is too high!"<<std::endl;
	}
    Vec3 velocity_gt =fullSystem->T_dsoworld_eurocworld.block<3,3>(0,0) * newFrame->shell->groundtruth.velocity;
    float velocity_direction_error = acos(
            velocity_gt.dot(navstate_out.velocity()) / (velocity_gt.norm() * navstate_out.velocity().norm())
    ) * 180 / M_PI;
//	std::cout<<"Optimized velocity: "<<navstate_out.velocity().transpose()
//            <<" GT: " << velocity_gt.transpose()<<std::endl
//            << "Angle error: " << velocity_direction_error << std::endl;
//	std::cout<<"Optimized velocity norm: "<<navstate_out.velocity().norm()<<" GT norm: "<<newFrame->shell->groundtruth.velocity.norm()<<std::endl;

	// calculate error in gravity direction
	Vec3 gravity_gt = newFrame->shell->last_frame->groundtruth.pose.rotation().matrix().bottomRows(1).transpose();
	Vec3 gravity_est = navstate_i_current.pose().rotation().matrix().bottomRows(1).transpose();
	float gravity_error = acos(
		gravity_gt.dot(gravity_est) / (gravity_gt.norm() * gravity_est.norm())
	) * 180 / M_PI;

//	std::cout << "Orientation error: " << gravity_error << std::endl;
//
//	std::cout << "Previous pose error: "
//			  << gtsam::Pose3::Logmap(navstate_i_gt.pose().inverse().compose(navstate_i_current.pose())).transpose() << " "
//			  << (navstate_i_current.velocity() - navstate_i_gt.velocity()).transpose()
//			  << std::endl;
//
//	std::cout << "Current pose error: "
//			  << gtsam::Pose3::Logmap(navstate_j_gt.pose().inverse().compose(navstate_j_current.pose())).transpose() << " "
//			  << (navstate_j_current.velocity() - navstate_j_gt.velocity()).transpose()
//			  << std::endl;

//	std::cout<<" IMU version: affine a: "<< aff_g2l_out.a << "affine b_unscaled: "<< aff_g2l_out.b<<std::endl;
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
	std::cout<<"The bias after tracking"<<biases_out<<std::endl;

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

void CoarseTracker::updatePriors()
{
	Mat1515 Hbb;
	Mat1517 Hbm;
	Mat1717 Hmm;

	Vec15 bb;
	Vec17 bm;

	splitHessianForMarginalization(
			Hbb, Hbm, Hmm,
			bb, bm
	);

	fullSystem->Hprior.setZero();
	fullSystem->bprior.setZero();
	if (lastRef->shell == newFrame->shell->last_frame)
	{
		fullSystem->Hprior.setIdentity();
		fullSystem->Hprior.diagonal().head<6>() *= 1e2;
		fullSystem->Hprior.diagonal().segment<3>(6) *= 1;
		fullSystem->Hprior.diagonal().segment<3>(9) *= 1e-6;
		fullSystem->Hprior.diagonal().segment<3>(12) *= 1e1;

//		cv::Mat Hmm_cv(2, 2, CV_64F), Hmm_cv_inv(2, 2, CV_64F);
//		Mat22 Hmm_relevant = Hmm.topLeftCorner(2, 2);
//		cv::eigen2cv(Hmm_relevant, Hmm_cv);
//		// pseudo-inverse
//		cv::invert(Hmm_cv, Hmm_cv_inv, cv::DECOMP_SVD);
//
//		Mat22 Hmm_inv;
//		cv::cv2eigen(Hmm_cv_inv, Hmm_inv);
//
//		// exclude bias from computation
//		Mat99 Hbb_no_bias = Hbb.topLeftCorner<9, 9>();
//		Eigen::Matrix<double, 9, 11> Hbm_no_bias = Hbm.topLeftCorner<9, 11>();
//
//		fullSystem->Hprior.topLeftCorner(9, 9) = Hbb_no_bias - Hbm_no_bias.leftCols(2) * Hmm_inv * Hbm_no_bias.leftCols(2).transpose();
//		Mat99 Prior_cov = fullSystem->Hprior.topLeftCorner(9, 9);
//		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Prior_cov);
//		Eigen::MatrixXd Prior_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > 1e-6).select(saes.eigenvalues().array(), 0)).asDiagonal() * saes.eigenvectors().transpose();
//
//		Eigen::LLT<Eigen::MatrixXd> lltOfA(Prior_inv); // compute the Cholesky decomposition of A
//		if(lltOfA.info() == Eigen::NumericalIssue)
//		{
//			std::cout<<"Possibly non semi-positive definitie information matrix!"<<std::endl;
//		}
//
//		fullSystem->Hprior.topLeftCorner(9, 9) = Prior_inv;
//		fullSystem->bprior.head(9) = bb.head(9) - Hbm_no_bias.leftCols(2) * Hmm_inv * bm.head(2);
//
////        fullSystem->Hprior = Hbb - Hbm.leftCols(2) * Hmm_inv * Hbm.leftCols(2).transpose();
////        fullSystem->bprior = bb - Hbm.leftCols(2) * Hmm_inv * bm.head(2);
	}
	else
	{
//        fullSystem->Hprior = Hbb - Hbm * Hmm.inverse() * Hbm.transpose();
//        fullSystem->bprior = bb - Hbm * Hmm.inverse() * bm;

		// ignore the affine parameters
//		cv::Mat Hmm_cv(15, 15, CV_64F), Hmm_cv_inv(15, 15, CV_64F);
//		Mat1515 Hmm_relevant = Hmm.bottomRightCorner<15, 15>();
//		cv::eigen2cv(Hmm_relevant, Hmm_cv);
//		// pseudo-inverse
//		cv::invert(Hmm_cv, Hmm_cv_inv, cv::DECOMP_SVD);
//
//		Mat1515 Hmm_inv;
//		cv::cv2eigen(Hmm_cv_inv, Hmm_inv);

		cv::Mat Hmm_cv(17, 17, CV_64F), Hmm_cv_inv(17, 17, CV_64F);
		cv::eigen2cv(Hmm, Hmm_cv);
		// pseudo-inverse
		cv::invert(Hmm_cv, Hmm_cv_inv, cv::DECOMP_SVD);

		Eigen::Matrix<double, 17, 17> Hmm_inv;
		cv::cv2eigen(Hmm_cv_inv, Hmm_inv);

//		The method to compute the inverse of Hmm in VINS

//		Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
//		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
//
//		//ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
//
//		Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
//		//printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());
//
//		Eigen::VectorXd bmm = b.segment(0, m);
//		Eigen::MatrixXd Amr = A.block(0, m, m, n);
//		Eigen::MatrixXd Arm = A.block(m, 0, n, m);
//		Eigen::MatrixXd Arr = A.block(m, m, n, n);
//		Eigen::VectorXd brr = b.segment(m, n);
//		A = Arr - Arm * Amm_inv * Amr;
//		b = brr - Arm * Amm_inv * bmm;
//
//		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
//		Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
//		Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
//
//		Eigen::VectorXd S_sqrt = S.cwiseSqrt();
//		Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
//
//		linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
//		linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;


#if 0
		// exclude bias from computation
		Mat99 Hbb_no_bias = Hbb.topLeftCorner<9, 9>();
		Eigen::Matrix<double, 9, 11> Hbm_no_bias = Hbm.topLeftCorner<9, 11>();

		Mat99 Prior_cov = Hbb_no_bias - Hbm_no_bias * Hmm_inv.topLeftCorner<11, 11>() * Hbm_no_bias.transpose();
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Prior_cov);
		Eigen::MatrixXd Prior_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > 1e-6).select(saes.eigenvalues().array(), 0)).asDiagonal() * saes.eigenvectors().transpose();

		//std::cout << "Eigen values: " << Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Prior_inv).eigenvalues().matrix().transpose() << std::endl;
		Eigen::LLT<Eigen::MatrixXd> lltOfA(Prior_inv); // compute the Cholesky decomposition of A
		if(lltOfA.info() == Eigen::NumericalIssue)
		{
			std::cout<<"Possibly non semi-positive definitie information matrix!"<<std::endl;
		}

		fullSystem->Hprior.topLeftCorner<9, 9>() = Prior_inv;
		fullSystem->bprior.head<9>() = bb.head<9>() - Hbm_no_bias * Hmm_inv.topLeftCorner<11, 11>() * bm.tail<11>();
#else
        // include bias in the computation
		Mat1515 Prior_cov = Hbb - Hbm * Hmm_inv * Hbm.transpose();
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Prior_cov);
		Eigen::MatrixXd Prior_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > 1e-6).select(saes.eigenvalues().array(), 0)).asDiagonal() * saes.eigenvectors().transpose();

		//std::cout << "Eigen values: " << Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Prior_inv).eigenvalues().matrix().transpose() << std::endl;
		Eigen::LLT<Eigen::MatrixXd> lltOfA(Prior_inv); // compute the Cholesky decomposition of A
		if(lltOfA.info() == Eigen::NumericalIssue)
		{
			std::cout<<"Possibly non semi-positive definitie information matrix!"<<std::endl;
		}

		fullSystem->Hprior = Prior_inv;
		fullSystem->bprior = bb - Hbm * Hmm_inv * bm.tail(9);

		// debug: set the priors associated with bias to zero
//		fullSystem->Hprior.rightCols(6).setZero();
//		fullSystem->Hprior.bottomRows(6).setZero();
//		fullSystem->bprior.tail(6).setZero();
#endif


//		fullSystem->Hprior = Hbb - Hbm.rightCols(15) * Hmm_inv * Hbm.rightCols(15).transpose();
//		fullSystem->bprior = bb - Hbm.rightCols(15) * Hmm_inv * bm.tail(15);
	}

//	std::cout << "Prior H: \n" << fullSystem->Hprior.topLeftCorner(9, 9) << std::endl
//			  << "Prior b: " << fullSystem->bprior.head(9).transpose() << std::endl;
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
