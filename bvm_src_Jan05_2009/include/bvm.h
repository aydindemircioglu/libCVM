#ifndef _BVM_H_
#define _BVM_H_

#include "cvm.h"

// API for BVM

void solve_bvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);


//------------------------------------------------------------------------------------------------------------------

//
// Gram matrix of BVM
//
class BVM_Q: public Kernel
{
public:
	BVM_Q(const svm_problem* prob_, const svm_parameter* param_, schar *y_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init		
		prob  = prob_;
		param = param_;
		y     = y_;
		kappa = (Qfloat)((this->*kernel_function)(0,0) + 1.0 + (1.0/(param->C)));

		if (!Kernel::IsSelfConst(*param))
		{
			printf("kernel: %d, BVM can work for isotropic kernels only!\n",param->kernel_type);
			exit(-1);
		}
		
		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}
	~BVM_Q() { delete kernelCache; }

	Qfloat *get_QD() const { return NULL; }
	Qfloat *get_Q(int idx, int basisNum, int* basisIdx) const
	{
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	
#ifdef COMP_STAT			
			kernelEval += (basisNum - numRet);
#endif

			// fill remaining		 
			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				if (idx != idx2)		
					Q[i] = y[idx]*y[idx2]*(Qfloat)((this->*kernel_function)(idx, idx2) + 1.0);			
				else				
					Q[i] = kappa;
			}						
		}
		return Q;
	}	
	double dot_c_wc(int idx, int basisNum, int* basisIdx, double *coeff, bool &depend, double thres = INF)
	{
		double dist = 0.0;		
		depend      = false;
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{			
			for (int j=0; j<basisNum; j++)
				if (idx != basisIdx[j] && Q_i[j] >= thres)
				{
					depend = true;
					return INF;
				}
				else
					dist += Q_i[j]*coeff[basisIdx[j]];							
		}		
		return dist;
	}

	Qfloat getKappa() const { return kappa; }		
	void swap_index(int i, int j) const { printf("CVC_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	schar* y;	
	Qfloat kappa;

	sCache *kernelCache;
	mutable int kernelEval;	
};


//------------------------------------------------------------------------------------------------------------------

//
// Solver for BVM
//
class Solver_BVM 
{
public:
	Solver_BVM() {}
	~Solver_BVM()
	{
		// free memory			
		delete [] y;
		delete [] chklist;
		delete [] coreIdx;
        delete [] coreGrad;
        delete [] posIdx;
		delete [] negIdx;

		delete kernelQ;
	}

   	void   Init(const svm_problem* prob, const svm_parameter* param, double *_alpha);
	int    Solve(int num_basis, double bvm_eps);		
	double ComputeSolution(double *alpha, double Threshold);

    bool   IsExitOnMaxIter() const { return (coreNum >= min(maxNumBasis,numData)); }
    double GetCoreNorm2 () const { return coreNorm2; } 
	double ComputeRadius2() const { return r2; }
	double GetKappa() const { return kappa; }
	
protected:	
	inline void _maxDistInCache(int idx, double tmpdist, double &maxDistance2, int &maxDistance2Idx)
    {
        double dist2 = tmpdist - 2.0*coreGrad[idx];
		if (dist2 > maxDistance2)
		{
			maxDistance2    = dist2;
			maxDistance2Idx = idx;
		}        
    }
	inline void _maxDistCompute(int idx, double dot_c, double tmpdist, double &maxDistance2, int &maxDistance2Idx)
    {
        double dist2 = tmpdist - 2.0*dot_c;
		if (dist2 > maxDistance2)
		{
			maxDistance2    = dist2;
			maxDistance2Idx = idx;
		}        
    }
    double _update (double maxDistance2, int maxDistance2Idx);    

private:
	int posNum;
	int negNum;	
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;	
    int numData;

    double *alpha;
	schar  *y;
	BVM_Q  *kernelQ;
   	double kappa;		// square radius of kernel feature space
	double r2;			// square radius of EB
	double c;			// augmented center coeff.	
	double coreNorm2;	// square normal of the center

    int     maxNumBasis;
    int    *coreIdx;
	int     coreNum;
	Qfloat *coreGrad;
	char   *chklist;

    const svm_parameter *param;
};


#endif //_BVM_H_
