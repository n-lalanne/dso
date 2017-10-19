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
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "FullSystem.h"




namespace dso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(int w, int h, FullSystem* fullsystem_);
	~CoarseTracker();

	bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out, SE3 &previousToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

	bool trackNewestCoarsewithIMU(
			FrameHessian* newFrameHessian,
			gtsam::NavState &navstate_i_out, gtsam::NavState &navstate_out,Vec6 &pbiases_out,  Vec6 &biases_out, AffLight &aff_g2l_out,
			int coarsestLvl,
			Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

	void updatePriors();

	void setUnscaledGaussNewtonSys(Mat3232 H, Vec32 b);

	void splitHessianForMarginalization(
			Mat1515 &Hbb,
			Mat1517 &Hbm,
			Mat1717 &Hmm,
			Vec15 &bb,
			Vec17 &bm
	);

	void setCoarseTrackingRef(
			std::vector<FrameHessian*> frameHessians);

	void makeK(
			CalibHessian* HCalib);

	bool debugPrint, debugPlot;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];
	FullSystem* fullSystem;

    void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);
	void setprevbias(const Vec6 bias);

	FrameHessian* lastRef;
	AffLight lastRef_aff_g2l;
	FrameHessian* newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
	Vec3 lastFlowIndicators;
	double firstCoarseRMSE;

	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);
	SE3 camtoimu() { return dso_vi::Tbc; }
	SE3 imutocam() { return dso_vi::Tcb; }

private:


	float* idepth[PYR_LEVELS];
	float* weightSums[PYR_LEVELS];
	float* weightSums_bak[PYR_LEVELS];


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	Vec6 calcRes(int lvl, const SE3 &refToNew, const SE3 &previousToNew, AffLight aff_g2l, float cutoffTH);
	Vec6 calcResIMU(int lvl, const gtsam::NavState previous_navstate, const gtsam::NavState current_navstate, AffLight aff_g2l,const Vec6 prev_biases,const Vec6 biases, float cutoffTH);
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGSSSESingle(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGSSSESingleIMU(int lvl, Mat1717 &H_out, Vec17 &b_out, const gtsam::NavState navState_, AffLight aff_g2l);
    // order of states: t_j, R_j, a_j, b_j, v_j, ba_j, bg_j, t_i, R_i, v_i, ba_i, bg_i
    void calcGSSSEDoubleIMU(int lvl, Mat3232 &H_out, Vec32 &b_out, const gtsam::NavState navState_previous, const gtsam::NavState navState_current, AffLight aff_g2l);
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGSSSEst(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

    // for imu errors
    Vec15 calcIMURes(gtsam::NavState previous_navstate, gtsam::NavState current_navstate, Vec6 prev_biases, Vec6 bias);
	Vec15 calcPriorRes(gtsam::NavState previous_navstate, gtsam::NavState current_navstate);
    gtsam::Matrix  J_imu_Rt, J_imu_bias, J_imu_v;
    gtsam::Matrix  J_imu_Rt_previous, J_imu_bias_previous, J_imu_v_previous;
	Mat1515 J_prior;

    Vec15 res_imu;
	Vec15 res_prior;
    Mat1515 information_imu;
	Mat1515 information_prior;

	Mat3232 H_unscaled;
	Vec32 b_unscaled;

	// pc buffers
	float* pc_u[PYR_LEVELS];
	float* pc_v[PYR_LEVELS];
	float* pc_idepth[PYR_LEVELS];
	float* pc_color[PYR_LEVELS];
	int pc_n[PYR_LEVELS];

	// warped buffers
	float* buf_warped_cx;
	float* buf_warped_cy;
	float* buf_warped_cz;
    float* buf_warped_rx;
    float* buf_warped_ry;
    float* buf_warped_rz;
	float* buf_warped_lpc_idepth;
	float* buf_warped_idepth;
	float* buf_warped_u;
	float* buf_warped_v;
	float* buf_warped_dx;
	float* buf_warped_dy;
	float* buf_warped_residual;
	float* buf_warped_weight;
	float* buf_warped_refColor;
	int buf_warped_n;


    std::vector<float*> ptrToDelete;


	Accumulator9 acc;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame);

	void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

	void makeK( CalibHessian* HCalib);


	float* fwdWarpedIDDistFinal;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	void addIntoDistFinal(int u, int v);


private:

	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

}

