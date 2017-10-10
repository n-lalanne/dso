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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;


void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

    if(adHost != 0) delete[] adHost;
    if(adTarget != 0) delete[] adTarget;
    adHost = new Mat88[nFrames*nFrames];
    adTarget = new Mat88[nFrames*nFrames];

    for(int h=0;h<nFrames;h++)
        for(int t=0;t<nFrames;t++)
        {
            FrameHessian* host = frames[h]->data;
            FrameHessian* target = frames[t]->data;

            SE3 hostToTarget = target->get_worldToImu_evalPT() * host->get_worldToImu_evalPT().inverse();

            Mat88 AH = Mat88::Identity();
            Mat88 AT = Mat88::Identity();

            AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
            AT.topLeftCorner<6,6>() = Mat66::Identity();


            Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
            AT(6,6) = -affLL[0];
            AH(6,6) = affLL[0];
            AT(7,7) = -1;
            AH(7,7) = affLL[0];

            AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
            AH.block<3,8>(3,0) *= SCALE_XI_ROT;
            AH.block<1,8>(6,0) *= SCALE_A;
            AH.block<1,8>(7,0) *= SCALE_B;
            AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
            AT.block<3,8>(3,0) *= SCALE_XI_ROT;
            AT.block<1,8>(6,0) *= SCALE_A;
            AT.block<1,8>(7,0) *= SCALE_B;

            adHost[h+t*nFrames] = AH;
            adTarget[h+t*nFrames] = AT;
        }
    cPrior = VecC::Constant(setting_initialCalibHessian);


    if(adHostF != 0) delete[] adHostF;
    if(adTargetF != 0) delete[] adTargetF;
    adHostF = new Mat88f[nFrames*nFrames];
    adTargetF = new Mat88f[nFrames*nFrames];

    for(int h=0;h<nFrames;h++)
        for(int t=0;t<nFrames;t++)
        {
            adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
            adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
        }

    cPriorF = cPrior.cast<float>();


    EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;


	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);


	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames)
	{
		for(EFPoint* p : f->points)
		{
			for(EFResidual* r : p->residualsAll)
			{
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;



	delete accSSE_top_L;
	delete accSSE_top_A;
	delete accSSE_bot;
}




void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++)
		{
			int idx = h+t*nFrames;
			adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx]
					        +frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();
	for(EFFrame* f : frames)
	{
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

		for(EFPoint* p : f->points)
			p->deltaF = p->data->idepth-p->data->idepth_zero;
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
				accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
		resInA = accSSE_top_A->nres[0];
	}
	else
	{
		accSSE_top_A->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_A->addPoint<0>(p,this);
		accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false);
		resInA = accSSE_top_A->nres[0];
	}
}

// accumulates & shifts L.
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero, accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
				accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
		resInL = accSSE_top_L->nres[0];
	}
	else
	{
		accSSE_top_L->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_top_L->addPoint<1>(p,this);
		accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
		resInL = accSSE_top_L->nres[0];
	}
}





void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
	if(MT)
	{
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
		red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
				accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
		accSSE_bot->stitchDoubleMT(red,H,b,this,true);
	}
	else
	{
		accSSE_bot->setZero(nFrames);
		for(EFFrame* f : frames)
			for(EFPoint* p : f->points)
				accSSE_bot->addPoint(p, true);
		accSSE_bot->stitchDoubleMT(red, H, b,this,false);
	}
}

void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();
	for(EFFrame* h : frames)
	{
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx);
		h->data->step.tail<2>().setZero();

		for(EFFrame* t : frames)
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose() *   adHostF[h->idx+nFrames*t->idx]
			            + xF.segment<8>(CPARS+8*t->idx).transpose() * adTargetF[h->idx+nFrames*t->idx];
	}

	if(MT)
		red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
						this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
	else
		resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0)
		{
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
		b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
		}

		p->data->step = - b*p->HdiF;
		assert(std::isfinite(p->data->step));
	}
}


double EnergyFunctional::calcMEnergyF()
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}


void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid)
{

	Accumulator11 E;
	E.initialize();
	VecCf dc = cDeltaF;

	for(int i=min;i<max;i++)
	{
		EFPoint* p = allPoints[i];
		float dd = p->deltaF;

		for(EFResidual* r : p->residualsAll)
		{
			if(!r->isLinearized || !r->isActive()) continue;

			Mat18f dp = adHTdeltaF[r->hostIDX+nFrames*r->targetIDX];
			RawResidualJacobian* rJ = r->J;



			// compute Jp*delta
			float Jp_delta_x_1 =  rJ->Jpdxi[0].dot(dp.head<6>())
						   +rJ->Jpdc[0].dot(dc)
						   +rJ->Jpdd[0]*dd;

			float Jp_delta_y_1 =  rJ->Jpdxi[1].dot(dp.head<6>())
						   +rJ->Jpdc[1].dot(dc)
						   +rJ->Jpdd[1]*dd;

			__m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
			__m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
			__m128 delta_a = _mm_set1_ps((float)(dp[6]));
			__m128 delta_b = _mm_set1_ps((float)(dp[7]));

			for(int i=0;i+3<patternNum;i+=4)
			{
				// PATTERN: E = (2*res_toZeroF + J*delta) * J*delta.
				__m128 Jdelta =            _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i),Jp_delta_x);
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i),Jp_delta_y));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i),delta_a));
				Jdelta = _mm_add_ps(Jdelta,_mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i),delta_b));

				__m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
				r0 = _mm_add_ps(r0,r0);
				r0 = _mm_add_ps(r0,Jdelta);
				Jdelta = _mm_mul_ps(Jdelta,r0);
				E.updateSSENoShift(Jdelta);
			}
			for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			{
				float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 + rJ->JIdx[1][i]*Jp_delta_y_1 +
								rJ->JabF[0][i]*dp[6] + rJ->JabF[1][i]*dp[7];
				E.updateSingleNoShift((float)(Jdelta * (Jdelta + 2*r->res_toZeroF[i])));
			}
		}
		E.updateSingle(p->deltaF*p->deltaF*p->priorF);
	}
	E.finish();
	(*stats)[0] += E.A;
}




double EnergyFunctional::calcLEnergyF_MT()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	double E = 0;
	for(EFFrame* f : frames)
        E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

	E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

	red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt,
			this, _1, _2, _3, _4), 0, allPoints.size(), 50);

	return E+red->stats[0];
}



EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);

    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;

	nResiduals++;
	r->efResidual = efr;
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;

	//bM.conservativeResize(11 *nFrames+CPARS -11);
	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();


	for(EFFrame* fh2 : frames)
	{
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}


void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();


	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;


    connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}

// from 8 to 8/17(biases are nor used for now)
void EnergyFunctional::stateexpand(MatXX &H, VecX &b)
{
	int nframes = frames.size();
	if(nframes <=2 ){
		std::cout<<"Do not call this function in vo model!"<<std::endl;
		exit(0);
	}
	std::vector<int> sizearr;
	sizearr.resize(nframes,8);
	for(int i=1;i<nframes;i++){
		if(frames[i]->data->imufactorvalid)
		{
			std::cout<< "frame "<<i <<" is valid"<<std::endl;
			sizearr[i] = 17;
			sizearr[i-1] = 17;
		}
	}
//	std::cout<<"H:\n "<<H.diagonal()<<std::endl;
//	std::cout<<"b:\n "<<b<<std::endl;
	std::vector<int> framepos;
	framepos.resize(nframes);
	framepos[0] = CPARS;
	for(int i = 1 ; i < nframes ; i++){
		framepos[i] = framepos[i-1] + sizearr[i-1];
	}


	int totalsize = std::accumulate(sizearr.begin(),sizearr.end(),CPARS);
	MatXX H_tmp = H;
	VecX b_tmp = b;
	H.conservativeResize(totalsize,totalsize);
	b.conservativeResize(totalsize);
	H.setZero();
	b.setZero();

	b.head(CPARS) = b_tmp.head(CPARS);
	H.topLeftCorner(CPARS,CPARS)=H_tmp.topLeftCorner(CPARS,CPARS);

	for(int indexi=0;indexi < nframes; indexi++)
	{
		b.segment<8>(framepos[indexi]) = b_tmp.segment<8>(CPARS+indexi*8);
		H.block<CPARS,8>(0,framepos[indexi]) = H_tmp.block<CPARS,8>(0,CPARS+indexi*8);
		for(int indexj=0;indexj<nFrames;indexj++)
		{
			H.block<8,8>(framepos[indexi],framepos[indexj])=H_tmp.block<8,8>(CPARS+indexi*8,CPARS+indexj*8);
		}
	}
	H.leftCols(CPARS) = H.topRows(CPARS).transpose();
//	std::cout<<"(after extend)H:\n "<<H.diagonal()<<std::endl;
//	std::cout<<"(after extend)b:\n "<<b<<std::endl;
}

// from 11/8 to 8(only for vector)
VecX EnergyFunctional::solutionreduce(VecX x)
{
	if(nFrames <=2 ){
		std::cout<<"Do not call this function in vo model!"<<std::endl;
		exit(0);
	}
	std::cout<<"x: "<<x.transpose()<<std::endl;
	VecX x_tmp = x;
	x.conservativeResize(CPARS+nFrames*8);
	x.setZero();
	x.head(CPARS) = x_tmp.head(CPARS);
	for(int i=0;i < nFrames; i++)
	{
		x.segment<8>(CPARS+i*8) = x_tmp.segment<8>(frames[i]->reducedframepos);
	}
	return x;
	std::cout<<"after x: "<<x.transpose()<<std::endl;
}

// from 17 to 11/8(biases are nor used for now)
void EnergyFunctional::statereduce(MatXX &H, VecX &b)
{
	if(nFrames <=2 ){
		std::cout<<"Do not call this function in vo model!"<<std::endl;
		exit(0);
	}

	std::vector<int> sizearr;
	sizearr.resize(nFrames,8);
	std::vector<int> reducedsizearr;
	reducedsizearr.resize(nFrames,8);
	for(int i=1;i<nFrames;i++){
		if(frames[i]->data->imufactorvalid)
		{
			sizearr[i] = 17;
			sizearr[i-1] = 17;
			reducedsizearr[i] = 11;
			reducedsizearr[i-1] = 11;
		}
		frames[i-1]->reducedstatesize = reducedsizearr[i-1];
		frames[i]->reducedstatesize = reducedsizearr[i];
	}
	std::vector<int> framepos;
	std::vector<int> reducedframepos;
	framepos.resize(nFrames);
	reducedframepos.resize(nFrames);
	framepos[0] = CPARS;
	reducedframepos[0] = CPARS;
	frames[0]->framepos = framepos[0];
	frames[0]->reducedframepos = reducedframepos[0];
	for(int i = 1 ; i < nFrames ; i++){
		framepos[i] = framepos[i-1] + sizearr[i-1];
		reducedframepos[i] = reducedframepos[i-1] + reducedsizearr[i-1];
		frames[i]->framepos = framepos[i];
		frames[i]->reducedframepos = reducedframepos[i];
	}
	reducedtotalsize = std::accumulate(reducedsizearr.begin(),reducedsizearr.end(),CPARS);

	std::cout<<"H: "<<H.diagonal().transpose()<<std::endl;
	std::cout<<"b: "<<b.transpose()<<std::endl;
	MatXX H_tmp = H;
	VecX b_tmp = b;

	H.conservativeResize(reducedtotalsize,reducedtotalsize);
	b.conservativeResize(reducedtotalsize);
	H.setZero();
	b.setZero();

	b.head(CPARS) = b_tmp.head(CPARS);
	H.topLeftCorner(CPARS,CPARS)=H_tmp.topLeftCorner(CPARS,CPARS);

	int posi,posj,rposi,rposj,blocksize;
	for(int i=0;i<nFrames;i++)
	{

		posi = framepos[i];
		rposi = reducedframepos[i];
		if(reducedsizearr[i] == 8) {
			b.segment<8>(rposi) = b_tmp.segment<8>(posi);
			H.block<CPARS, 8>(0, rposi) = H_tmp.block<CPARS, 8>(0, posi);
			for (int j = 0; j < nFrames; j++) {
				posj = framepos[j];
				rposj = reducedframepos[j];
				H.block<8, 8>(rposi, rposj)
						= H_tmp.block<8, 8>(posi, posj);
			}
		}
		else if(reducedsizearr[i] == 11)
		{
			b.segment<11>(rposi) = b_tmp.segment<11>(posi);
			H.block<CPARS, 11>(0, rposi) = H_tmp.block<CPARS, 11>(0, posi);
			for (int j = 0; j < nFrames; j++)
			{
				posj = framepos[j];
				rposj = reducedframepos[j];
				H.block<11, 11>(rposi, rposj)
						= H_tmp.block<11, 11>(posi, posj);
			}
		}
		else
		{
			assert(reducedsizearr[i] == 11|| reducedsizearr[i] == 8);
		}
	}


	H.leftCols(CPARS) = H.topRows(CPARS).transpose();
	std::cout<<"after H: "<<H.diagonal().transpose()<<std::endl;
	std::cout<<"after b: "<<b.transpose()<<std::endl;
}

//// brief: add those Kinematics constarints
void EnergyFunctional::accumulateIMU_ST(MatXX &H, VecX &b)
{
    if(nFrames <=2 ){
        std::cout<<"Do not call this function in vo model!"<<std::endl;
        exit(0);
    }
    std::vector<int> sizearr;
    sizearr.resize(nFrames,8);
    for(int i=1;i<nFrames;i++){
		if(frames[i]->data->imufactorvalid)
            {
                sizearr[i] = 17;
                sizearr[i-1] = 17;
            }
		frames[i]->statesize = sizearr[i];
		frames[i-1]->statesize = sizearr[i-1];
    }
    totalsize = std::accumulate(sizearr.begin(),sizearr.end(),CPARS);

    H = MatXX::Zero(totalsize, totalsize);
    b = VecX::Zero(totalsize);
    int currentpos = CPARS-1;
    for(int indexi=1;indexi<frames.size();indexi++)
    {
        if(!frames[indexi]->data->imufactorvalid)
        {
            currentpos += frames[indexi-1]->statesize;
            continue;
        }
        Mat3434 H_temp;
        Vec34 b_temp;
        Mat1515 information_imu;
        Vec15 res_imu;

        H_temp.setZero();
        b_temp.setZero();

        information_imu = frames[indexi]->data->shell->getIMUcovarianceBA().inverse();
        res_imu = frames[indexi]->data->kfimures;

        Mat1517 J_imu_travb_previous;
        J_imu_travb_previous.setZero();
        J_imu_travb_previous.block<15, 3>(0, 0) = frames[indexi]->data->J_imu_Rt_j.block<15, 3>(0, 3);//J_imu_Rt_previous.block<15, 3>(0, 3);
        J_imu_travb_previous.block<15, 3>(0, 3) = frames[indexi]->data->J_imu_Rt_j.block<15, 3>(0, 0);//J_imu_Rt_previous.block<15, 3>(0, 0);
        J_imu_travb_previous.block<15, 3>(0, 8) = frames[indexi]->data->J_imu_v_j.block<15, 3>(0, 0);//J_imu_v_previous.block<15, 3>(0, 0);

        // ------------------ don't ignore the cross terms in hessian between i and jth poses ------------------
        Mat1517 J_imu_travb_current;
        J_imu_travb_current.setZero();
        J_imu_travb_current.block<15, 3>(0, 0) = frames[indexi]->data->J_imu_Rt_i.block<15, 3>(0, 3);//J_imu_Rt.block<15, 3>(0, 3);
        J_imu_travb_current.block<15, 3>(0, 3) = frames[indexi]->data->J_imu_Rt_i.block<15, 3>(0, 0);//J_imu_Rt.block<15, 3>(0, 0);
        J_imu_travb_current.block<15, 3>(0, 8) = frames[indexi]->data->J_imu_v_i.block<15, 3>(0, 0);//J_imu_v.block<15, 3>(0, 0);

        Mat1534 J_imu_complete;
        J_imu_complete.leftCols(17) = J_imu_travb_previous;
        J_imu_complete.rightCols(17) = J_imu_travb_current;

        H_temp.noalias() = J_imu_complete.transpose() * information_imu * J_imu_complete;
        b_temp.noalias() = J_imu_complete.transpose() * information_imu * res_imu;

        //// TODO: set the coressponding blocks in H_imu
        H.block<34,34>(currentpos,currentpos) += H_temp;
        b.segment<34>(currentpos) += b_temp;
        currentpos += frames[indexi-1]->statesize;
    }
}


void EnergyFunctional::solveVISystemF(int iteration, double lambda, CalibHessian* HCalib){
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	std::cout<<"sloving VI ba"<<std::endl;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc , H_imu;
	VecX  bL_top, bA_top, bM_top, b_sc, b_imu;

	accumulateAF_MT(HA_top, bA_top,multiThreading);


	accumulateLF_MT(HL_top, bL_top,multiThreading);


	accumulateSCF_MT(H_sc, b_sc,multiThreading);

	bM_top = (bM+ HM * getStitchedDeltaF());

	std::cout<<"vi model: bL_top:\n"<<bL_top<<std::endl;
	std::cout<<"vi model: bM_top:\n"<<bM_top<<std::endl;
	std::cout<<"vi model: bA_top:\n"<<bA_top<<std::endl;
	std::cout<<"vi model: b_sc:\n"<<b_sc<<std::endl;

	accumulateIMU_ST(H_imu, b_imu);
	std::cout<<"vi model: b_imu:\n"<<b_imu<<std::endl;
	MatXX HFinal_top;
	VecX bFinal_top;

	stateexpand(HA_top, bA_top);
	stateexpand(HL_top, bL_top);
	stateexpand(H_sc, b_sc);


	HFinal_top = HL_top  + HA_top + H_imu;
	//std::cout<<bL_top.rows()<<" "<<bM_top.rows()<<" "<<bA_top.rows()<<" "<<b_sc.rows() <<std::endl;
	bFinal_top = bL_top  + bA_top - b_sc + b_imu;

	lastHS = HFinal_top - H_sc;
	lastbS = bFinal_top;





	for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
	HFinal_top -= H_sc * (1.0f/(1+lambda));


	VecX x;
	if(setting_solverMode & SOLVER_SVD)
	{
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	}
	else
	{
		statereduce(HFinal_top,bFinal_top);
		std::cout<<"bFinal_top:\n"<<bFinal_top<<std::endl;
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

    std::cout<<"The vi incremnt is :"<<x.transpose()<<std::endl;


	//// Todo: reduce the state to 8 for orthogonalization and after this operation, change it back
//	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
//	{
//
//		VecX xOld = x;
//		orthogonalize(&x, 0);
//	}


	lastX = x;

	//resubstituteF(x, HCalib);
	currentLambda= lambda;
	VIresubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;

}

void EnergyFunctional::VIresubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*17);
	// calculate pose, a, b steps
	resubstituteF_MT(solutionreduce(x), HCalib, MT);

	// calculate velocity step
	for(EFFrame* h : frames)
	{
		if(h->statesize != 8) {
			h->data->vstep = -x.segment<3>(h->reducedframepos + 8);
			// TODO: update this to optimize the bias
			h->data->biasstep.setZero();
			std::cout << "the velocity step of the frame" << h->frameID << " after:\n" << h->data->vstep << std::endl;
		}
	}

	// calculate bias step
}

void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);
	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension


//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//



	if((int)fh->idx != (int)frames.size()-1)
	{
		int io = fh->idx*8+CPARS;	// index of frame to move to end
		int ntail = 8*(nFrames-fh->idx-1);
		assert((io+8+ntail) == nFrames*8+CPARS);

		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);



//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";


	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();


//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);

	// schur-complement!
    MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

    std::cout << "HMScaled: " << HMScaled.diagonal().transpose() << std::endl;
    std::cout << "bMScaled: " << bMScaled.transpose() << std::endl;

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

    std::cout << "unscale HMScaled: " << HMScaled.diagonal().transpose() << std::endl;
    std::cout << "unscale bMScaled: " << bMScaled.transpose() << std::endl;

    std::cout << "before H: " << HM.diagonal().transpose() << std::endl;
    std::cout << "before b: " << bM.transpose() << std::endl;

	// set.
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

    std::cout << "after H: " << HM.diagonal().transpose() << std::endl;
    std::cout << "after b: " << bM.transpose() << std::endl;

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++)
	{
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);




//	VecX eigenvalues

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}




void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);


	allPointsToMarg.clear();
	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE)
			{
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
                        connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);
	for(EFPoint* p : allPointsToMarg)
	{
		accSSE_top_A->addPoint<2>(p,this);
		accSSE_bot->addPoint(p,false);
		removePoint(p);
	}
	MatXX M, Msc;
	VecX Mb, Mbsc;
	accSSE_top_A->stitchDouble(M,Mb,this,false,false);
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames) if(f->frameID==0) haveFirstFrame=true;

		if(!haveFirstFrame)
			orthogonalize(&b, &H);

	}

	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
		orthogonalize(&bM, &HM);

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{


	for(EFFrame* f : frames)
	{
		for(int i=0;i<(int)f->points.size();i++)
		{
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP)
			{
				removePoint(p);
				i--;
			}
		}
	}

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";


	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());





	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();



	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++)
	{
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;


//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";

}


void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{
	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX HL_top, HA_top, H_sc , H_imu;
	VecX  bL_top, bA_top, bM_top, b_sc, b_imu;

	accumulateAF_MT(HA_top, bA_top,multiThreading);


	accumulateLF_MT(HL_top, bL_top,multiThreading);



	accumulateSCF_MT(H_sc, b_sc,multiThreading);



	bM_top = (bM+ HM * getStitchedDeltaF());

    std::cout << "HM: " << HM.diagonal().transpose() << std::endl;
    std::cout << "bM_top" << bM_top.transpose() << std::endl;


	MatXX HFinal_top;
	VecX bFinal_top;

	if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM)
	{
		// have a look if prior is there.
		bool haveFirstFrame = false;
		for(EFFrame* f : frames)
			if(f->frameID==0) haveFirstFrame=true;




		MatXX HT_act =  HL_top + HA_top - H_sc;
		VecX bT_act =   bL_top + bA_top - b_sc;


		if(!haveFirstFrame)
			orthogonalize(&bT_act, &HT_act);
		HFinal_top = HT_act + HM;
		bFinal_top = bT_act + bM_top;





		lastHS = HFinal_top;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

	}
	else
	{


		HFinal_top = HL_top + HM + HA_top;
		bFinal_top = bL_top + bM_top + bA_top - b_sc;

		std::cout<<"vo model: bL_top:\n"<<bL_top<<std::endl;
		std::cout<<"vo model: bM_top:\n"<<bM_top<<std::endl;
		std::cout<<"vo model: bA_top:\n"<<bA_top<<std::endl;
		std::cout<<"vo model: b_sc:\n"<<b_sc<<std::endl;

		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);
		HFinal_top -= H_sc * (1.0f/(1+lambda));
	}






	VecX x;
	if(setting_solverMode & SOLVER_SVD)
	{
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++)
		{
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++)
		{
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }

			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;

	}
	else
	{
		std::cout<<"vo model: bFinal_top:\n"<<bFinal_top<<std::endl;
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

    std::cout << "x before: " << x.transpose() << std::endl;

	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) || (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER)))
	{
		VecX xOld = x;
		orthogonalize(&x, 0);
	}

    std::cout << "x after: " << x.transpose() << std::endl;

	lastX = x;


	//resubstituteF(x, HCalib);
	currentLambda= lambda;
	resubstituteF_MT(x, HCalib,multiThreading);
	currentLambda=0;


}
void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points)
		{
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll)
			{
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
			}
		}


	EFIndicesValid=true;
}


VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8); d.head<CPARS>() = cDeltaF.cast<double>();
	for(int h=0;h<nFrames;h++) d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}



}
