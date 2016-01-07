#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "random.h"
#include "utility.h"
#include "bvm.h"


#define MAX_REFINE_ITER  20		
#define COMPRESS_THRES   400


double Solver_BVM::_update (double maxDistance2, int maxDistance2Idx)
{
    double beta = sqrt(r2/(maxDistance2 + c*c));
	double rate = 1.0 - beta;

    // update center
    int i;
	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] *= beta;					

	// update constant and center norm
	c        *= beta;					
	coreNorm2 = coreNorm2*(beta) + kappa*(rate) - maxDistance2*(beta*rate);

	// update gradient
	Qfloat* kernelColumn = kernelQ->get_Q(maxDistance2Idx, coreNum, coreIdx);
	for (i = 0; i < coreNum; i++)				
	{
		int coreIdx_i = coreIdx[i];
		if (coreGrad[coreIdx_i] != 0.0)
			coreGrad[coreIdx_i] = (Qfloat)(coreGrad[coreIdx_i]*beta + rate*kernelColumn[i]);
	}

    return rate;
}

double Solver_BVM::ComputeSolution(double *_alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i++)
	{
        int ii = coreIdx[i];
		if (alpha[ii] > Threshold)
		{	
            sumAlpha  += alpha[ii];
            _alpha[ii] = alpha[ii]*y[ii];
			bias      += _alpha[ii];
		}
	}
	bias /= sumAlpha;

	for(i = 0; i < coreNum; i++)
		_alpha[coreIdx[i]] /= sumAlpha;
	return bias;
}

void Solver_BVM::Init(const svm_problem *prob, const svm_parameter *_param, double *_alpha)
{
    // init		
    param   = _param;
    alpha   = _alpha;
    numData = prob->l;
    posIdx  = new int[numData];	
	negIdx  = new int[numData];
	y       = new schar[numData];
    posNum  = 0;
	negNum  = 0;
	
    int i;
	for(i = 0; i < numData; i++)
	{
		if (prob->y[i] > 0)
		{
			y[i]             = 1;
			posIdx[posNum++] = i;
		}
		else
		{
			y[i]             = -1;
			negIdx[negNum++] = i;
		}
	}

    // initialize the kernel 
 	kernelQ   = new BVM_Q(prob, param, y);
	kappa     = kernelQ->getKappa();		// square radius of kernel feature space
	r2        = kappa;					// square radius of EB
	c         = sqrt(r2);				// augmented center coeff.	
	coreNorm2 = kappa;					// square normal of the center

    // initialize the coreset
	coreIdx	 = new int[numData];	
	coreNum	 = 0;	
	coreGrad = new Qfloat[numData];
	chklist  = new char[numData];
	for (i = 0; i < numData; i++)
	{
		coreGrad[i] = 0.0;
		chklist[i]  = 0;
	}

	coreIdx[coreNum++]   = 0;
	alpha   [coreIdx[0]] = 1.0;
	coreGrad[coreIdx[0]] = (Qfloat)kappa;
	chklist [coreIdx[0]] = 1;	
}

int Solver_BVM::Solve(int num_basis, double bvm_eps)
{
    // iterate on epsilons
    maxNumBasis          = num_basis;
	int updateNum        = 0;	
	double epsilonFactor = EPS_SCALING;
    for(double currentEpsilon = INITIAL_EPS; currentEpsilon/epsilonFactor > bvm_eps; currentEpsilon *= epsilonFactor)
	{
		// check epsilon
		currentEpsilon = (currentEpsilon < bvm_eps ? bvm_eps : currentEpsilon);
		double sepThres = kappa * (1.0 - (currentEpsilon + currentEpsilon * currentEpsilon * 0.5));

		// solve problem with current epsilon (warm start from the previous solution)
		double maxDistance2 = 0.0;
		int maxDistance2Idx = 0;
		while (maxDistance2Idx != -1)
		{
            // increase the usage of internal cache by constraining the search points when #BV is more than COMPRESS_THRES
			int refineIter = 0;
			while (coreNum > COMPRESS_THRES  && refineIter < MAX_REFINE_ITER)
			{			
				// compute (1+eps)^2*radius^2 - c^2
				double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

				// get a probabilistic sample
				maxDistance2    = radius_eps;
				maxDistance2Idx = -1;
				double tmpdist  = coreNorm2 + kappa;
 				for(int sampleIter = 0; (sampleIter < 10) && (maxDistance2Idx == -1); sampleIter++)
				{
					for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
					{
						// balanced and random sample						
						int rand32bit = random();
						int idx       = coreIdx[rand32bit % coreNum];

						// compute distance
						if (coreGrad[idx] != 0.0)
						{
							_maxDistInCache(idx, tmpdist, maxDistance2, maxDistance2Idx);							
						}
						else
						{
							bool depend;
							double dot_c  = kernelQ->dot_c_wc(idx,coreNum,coreIdx,alpha,depend);							
							coreGrad[idx] = (Qfloat) dot_c;
							_maxDistCompute(idx, dot_c, tmpdist, maxDistance2, maxDistance2Idx);
						}
					}
				}

				// check maximal distance
				if (maxDistance2Idx != -1)
				{					
					double rate = _update (maxDistance2, maxDistance2Idx);
                    alpha[maxDistance2Idx] += (rate);

					// info
					updateNum++;
//					if (updateNum%20 < 1) info(".");
#ifndef RELEASE_VER
					printf("#%d, #cv: %d, R: %.10f, |c-x|: %.10f, r: %g\n",updateNum, coreNum, r2-c*c, maxDistance2, rate);
#endif
				}
				refineIter ++;
			}

            // search any point
			{			
				// compute (1+eps)^2*radius^2 - c^2
				double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

				// get a probabilistic sample
				maxDistance2    = radius_eps;
				maxDistance2Idx = -1;
				double tmpdist  = coreNorm2 + kappa;
 				for(int sampleIter = 0; (sampleIter < 10) && (maxDistance2Idx == -1); sampleIter++)
				{
					for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
					{
						// balanced and random sample						
						int rand32bit = random();
						int idx       = (sampleNum+sampleNum < param->sample_size ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum]);

						// compute distance
						if (coreGrad[idx] != 0.0)
						{
                            _maxDistInCache(idx, tmpdist, maxDistance2, maxDistance2Idx);							
						}
						else
						{
							bool depend = true;
							double dot_c = kernelQ->dot_c_wc(idx, coreNum, coreIdx,alpha,depend,sepThres);
							if (depend == true)
								continue;
							if (chklist[idx] == 1)
								coreGrad[idx] = (Qfloat) dot_c;
                            _maxDistCompute(idx, dot_c, tmpdist, maxDistance2, maxDistance2Idx);
						}
					}
				}

				// check maximal distance
				if (maxDistance2Idx != -1)
				{					
					double rate = _update (maxDistance2, maxDistance2Idx);
                    if (alpha[maxDistance2Idx] == 0.0)
					{
						coreIdx[coreNum++]       = maxDistance2Idx;
						chklist[maxDistance2Idx] = 1;					
					}
					alpha[maxDistance2Idx] += rate;

					// info
					updateNum++;
//					if (updateNum%20 < 1) info(".");
#ifndef RELEASE_VER
					printf("#%d, #cv: %d, R: %.8f, |c-x|: %.8f, r: %g\n",updateNum, coreNum, r2-c*c, maxDistance2, rate);
#endif
				}
			}		
			if (IsExitOnMaxIter())
			{
				currentEpsilon = bvm_eps;
				break;
			}
		}
	}
	info("\n");

    return coreNum;
}


void solve_bvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{	
    // info
	info("num pattern = %ld\n", prob->l);

	// init		
	srandom(0);
    for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

    // solve BVM
    Solver_BVM solver;
	solver.Init(prob,param,alpha);  	
	double bvm_eps = param->eps;
	if (bvm_eps <= 0.0)
        bvm_eps = (4e-6/(solver.GetKappa()-1.0/param->C)+2.0/param->C/E_NUM_CV)/2.0/solver.GetKappa();
	printf("epsilon = %.9g\n",bvm_eps);
    int coreNum = solver.Solve(param->num_basis,bvm_eps);

    // compute solution vector		
	double THRESHOLD = 1e-5/coreNum;
	double bias      = solver.ComputeSolution(alpha, THRESHOLD);	
	double coreNorm2 = solver.GetCoreNorm2();

	// info in BVM		
	si->obj    = 0.5*coreNorm2;
	si->rho    = -bias;
	si->margin = coreNorm2;
}

