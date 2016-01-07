#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#include "cvm.h"


//
// Solver for Lagrangian SVM classification and regression
//
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) - 0.5(\alpha^T D)
//
//		e^T \alpha = 1
//		\alpha >= 0
//
// Given:
//
//	Q, D and an initial feasible point \alpha
//	num is the size of vectors and matrices
//
// solution will be put in \alpha
//

Solver_Lag::Solver_Lag(int *CoreIdx, int numCore, const QMatrix& Q, const double* D, double *inAlpha, double Eps, double MinEps, int initSize)
{
	this->l				= numCore;
	this->Q				= &Q;	
	this->allocatedSize = initSize;
	this->eps		    = Eps;
	this->minEps	    = MinEps;

	// The following members (inheritanced from Solver) will be used in update_alpha_status
	this->Cp = INF;							
	this->Cn = INF;	

	// all the following members (inheritanced from Solver) will not be used
	this->b     = NULL;
	this->G_bar = NULL;
	this->y     = NULL;
	active_set  = NULL;
	unshrinked  = true;
			
	{
		alpha		 = (double *)calloc(allocatedSize,sizeof(double));		// initialized as zero		
		alpha_status = (char *)calloc(allocatedSize,sizeof(char));			// initialize alpha_status
		for (int i=0;i<numCore;i++)
		{
			alpha[i] = inAlpha[i];
			update_alpha_status2(i);
		}
	}

	// initialize gradient
	{
		G	  = (double*) calloc(allocatedSize,sizeof(double));		
		vec_d = (double *)calloc(allocatedSize,sizeof(double));	
		int i;
		for(i=0;i<numCore;i++)
		{
			if(!is_lower_bound(i))						// non-zero alpha
			{
				const Qfloat *Q_i = Q.get_Q(CoreIdx[i],numCore,CoreIdx);
				double alpha_i = alpha[i];
				for(int j=0;j<numCore;j++)				
					G[j] += alpha_i*Q_i[j];
			}
		}

		for(i=0; i<numCore; i++)
		{
			vec_d[i] = D[i]*0.5;			
			G[i] -= vec_d[i];
		}
	}
	CacheQ_i = (Qfloat *)calloc(allocatedSize,sizeof(Qfloat));
}

Solver_Lag::~Solver_Lag()
{
	free(CacheQ_i);
	free(vec_d);	
	free(G);
	free(alpha);
	free(alpha_status);
}

int Solver_Lag::Solve(int *CoreIdx, int numCore, const double* newD)
{
	assert( numCore >= this->l );
	int prevSize   = this->l;
	this->l        = numCore;										// update the patterns	
	this->_CoreIdx = CoreIdx;

	// The problem is now larger (allocate space if necessary)
	if ( numCore >= allocatedSize )
	{			
		allocatedSize = (int)(1.5*allocatedSize);
		CacheQ_i      = (Qfloat*)realloc(CacheQ_i,     allocatedSize*sizeof(Qfloat));
		alpha         = (double*)realloc(alpha,		   allocatedSize*sizeof(double));
		vec_d         = (double*)realloc(vec_d,		   allocatedSize*sizeof(double));
		alpha_status  = (char*)  realloc(alpha_status, allocatedSize*sizeof(char));
		G             = (double*)realloc(G,            allocatedSize*sizeof(double));
	}

	// Initialize the newly-added part:	
	for (int i=prevSize; i<numCore; i++)
	{
		alpha[i] = 0.0;									// adding zero at the end			
		update_alpha_status2(i);			
		vec_d[i] = newD[i-prevSize]*0.5;

		// gradient
		G[i] = -vec_d[i];								// linear -> constant
		const Qfloat *Q_i = Q->get_Q(CoreIdx[i],numCore,CoreIdx);
		for (int j=0; j<prevSize; j++)
			G[i] += Q_i[j] * alpha[j];					// quadratic -> linear
	}
	QD = Q->get_QD();
	
	////////////////////////////////////////////////////////////////////////////////////
	// optimization step
	int iter = 0;	
	while(1)
	{
		int i,j, notSelected;
		if( (notSelected = select_working_set(i,j)) != 0 )
		{			
			if ( iter == 0 )
			{				
				// internal epsilon is too large
				while ( notSelected && (eps > minEps) )
				{
					eps *= EPS_SCALING;
					eps  = max(eps,minEps);
#ifndef RELEASE_VER
					printf("Dynamic Scheme: EPS now is %.10g\n", eps);
#else
					printf("o");
#endif
					notSelected = select_working_set(i,j);
				} 
				if (notSelected) 
				{
					printf("not selected\n");
					break;
				}
			}
			else break;
		}
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully		
		const double old_alpha_i = alpha[i];
		const double old_alpha_j = alpha[j];
		const double sum   = alpha[i] + alpha[j];		// original sum (must maintain during update)
		const Qfloat *Q_i  = CacheQ_i;		
		const Qfloat *Q_j  = Q->get_Q(CoreIdx[j],numCore, CoreIdx);
		const double P_val = max((double)(Q_i[i]+Q_j[j]-2.0*Q_i[j]), 0.0);	// it should be non-negative
		const double Q_val = (G[i]-G[j]) - old_alpha_i * P_val;		
		
		if ( P_val < EPSILON )											// non quadratic problem
		{
			if ( Q_val >= 0.0 )
				alpha[i] = 0.0;
			else alpha[i] = sum;										// unbounded linear problem
		}
		else
		{
			alpha[i] = (-Q_val/P_val);
			if ( alpha[i] < 0.0 )
				alpha[i] = 0.0;
			else if ( alpha[i] > sum )
				alpha[i] = sum;
		}
		alpha[j] = (sum - alpha[i]);

		// update the alpha's status
		update_alpha_status2(i);
		update_alpha_status2(j);

		// update G		
		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		for(int k=0;k<numCore;k++)
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;

		// stop if little improvement (eps^2 is a very very small number)
		if ( fabs(delta_alpha_i) + fabs(delta_alpha_j) < SQUARE(eps) )
			break;
	}
	
#ifndef RELEASE_VER
	info("#SMO=%d ",iter);
#endif
	return iter;
}

int Solver_Lag::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes - grad(f)_i, i 
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -grad(f)_j < -grad(f)_i, j for \alpha_j>0	
	double Gmax  = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<this->l;t++)
		if(-G[t] >= Gmax)
		{
			Gmax = -G[t];
			Gmax_idx = t;	
		}
		
	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(_CoreIdx[i],this->l,_CoreIdx);
	else
		return 1;

	// using approximate second order information to maximize the functional gain
	int j;
	for(j=0;j<this->l;j++)
	{
		if (!is_lower_bound(j))
		{
			double grad_diff=Gmax+G[j];
			if (grad_diff >= eps)
			{
				double obj_diff; 
				double quad_coef=Q_i[i]+QD[_CoreIdx[j]]-2.0*Q_i[j];
				if (quad_coef > 0.0)
					obj_diff = -(grad_diff*grad_diff)/quad_coef;
				else
					obj_diff = -(grad_diff*grad_diff)/TAU;
				if (obj_diff <= obj_diff_min)
				{
					Gmin_idx=j;
					obj_diff_min = obj_diff;
				}
			}
		}
	}

	if(Gmin_idx == -1)
 		return 1;

	for(j=0;j<this->l;j++)
		CacheQ_i[j] = Q_i[j];
	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

double Solver_Lag::computeObj() const
{
	double obj = 0.0;
	for (int i=0; i<this->l; i++)
		obj += alpha[i] * (G[i] - vec_d[i]);
	obj /= 2;
    
	return obj;
}

double Solver_Lag::computeCNorm() const
{
	double obj = 0.0;
	for (int i=0; i<this->l; i++)
		obj += alpha[i] * (G[i] + vec_d[i]);	
    
	return obj;
}

//---------------------------------------------------------------------------------------------------------------------

void Solver_Core::Init(const svm_problem *_prob, const svm_parameter* _param)
{
	prob  = _prob;
	param = _param;
	_Init();
}

bool Solver_Core::Create(double CVM_eps)
{
	bool flag = _Create(CVM_eps);
	if (flag) 
	{
		// use a small subset for initial MEB
		solver->Solve(coreIdx,coreNum,tempD);
		outAlpha = solver->getAlpha();
		ComputeRadius2();
	}
	return flag;
}


int Solver_Core::Solve(int num_basis, double cvm_eps)
{
	this->maxNumBasis = num_basis;

    // The convergence of CVM does not require the exact MEB on the core-set. 
    // With the recent advance of core-set approximation, the (1+epsilon/2)-MEB approximation on the core-set 
    // can guarantee the quality and the convergence of the (1+epsilon)-MEB approximation for the whole data set.
    // See [Kumar, Mitchell, and Yildirim, 2003]
    //
	// iterate on epsilons
	double epsilonFactor = EPS_SCALING;
	for(double currentEpsilon = INITIAL_EPS; currentEpsilon/epsilonFactor > cvm_eps; currentEpsilon *= epsilonFactor)
	{
		// check epsilon
		currentEpsilon = (currentEpsilon < cvm_eps ? cvm_eps : currentEpsilon);		

		// solve problem with current epsilon (warm start from the previous solution)
		double maxDistance2 = 0.0;
		int maxDistance2Idx = 0;
		double factor       = 1.0 + currentEpsilon;
		factor             *= factor;

        // The convergence of CVM does not require the exact most violating point. 
        // A violating point is enough for the convergence of CVM. 
        // Probabilistic speedup lead to a good tradeoff between convergence and complexity at each iteration
        //
		while (maxDistance2Idx != -1)
		{
			// get a probabilistic sample
			maxDistance2    = r2 * factor;
			maxDistance2Idx = -1;			
			for(int sampleIter = 0; (sampleIter < 7) && (maxDistance2Idx == -1); sampleIter++)			
				maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);

			// check maximal distance
			if (maxDistance2Idx != -1)
			{	
				_UpdateCoreSet(maxDistance2Idx);	
#ifndef RELEASE_VER
				printf("#%d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);
#endif				
				solver->Solve(coreIdx,coreNum,tempD);
				outAlpha = solver->getAlpha();				
				ComputeRadius2();
//				if (coreNum%20 < 1) info(".");				
			}
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
		}
	}	
	info("\n");

	return coreNum;
}
