#ifndef _CVM_H_
#define _CVM_H_

#include "random.h"
#include "svm.h"
#include "utility.h"

#define RELEASE_VER 1
//#define COMP_STAT 1

#define SMO_EPS			1e-3
#define PREDICT_BND		4e-6
#define E_NUM_CV		15000
#define INITIAL_CS		20
#define INITIAL_EPS		0.1
#define EPS_SCALING		0.5

// API for CVDD (MEB)
void solve_cvdd(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si);

// API for CVM (CCMEB)
void solve_cvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);

// API for CVR (CCMEB)
void solve_cvr(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si);


//------------------------------------------------------------------------------------------------------------------

//
// Sparse caching for kernel evaluations
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class sCache
{
public:
	sCache(const svm_parameter* param_, int num);
	virtual ~sCache();

	Qfloat* get_data(int idx, int basisNum, int& numRet);
	bool has_data(int idx) { return (head[idx].len > 0); } 

protected:		
	struct shead_t
	{
		shead_t *prev, *next;	// a cicular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
		int max_len;
		int refcount;
	};

	shead_t *head;
	shead_t lru_head;
	void lru_delete(shead_t *h);
	void lru_insert(shead_t *h);

	int numData;
	int maxItem;	
};

//------------------------------------------------------------------------------------------------------------------

//
// Gram matrix of CVDD
//
class CVDD_Q : public Kernel
{
public:
	CVDD_Q(const svm_problem* prob_, const svm_parameter* param_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		prob  = prob_;
		param = param_;
		QD    = new Qfloat[prob->l];
		C_inv = (Qfloat)(1.0/(param->C));
		
		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{	
			Eta = (Qfloat)((this->*kernel_function)(0,0) + C_inv);
			for (i=0; i< prob->l; i++)
				QD[i] = Eta;
		}
		else
		{	
			Eta = 0.0;
			for (i=0; i< prob->l; i++)	
			{
				QD[i] = (Qfloat)((this->*kernel_function)(i, i) + C_inv);
				if (QD[i] > Eta)
					Eta = QD[i];
			}	
		}		

		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~CVDD_Q()
	{
		delete kernelCache;
		delete [] QD;		
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
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
					Q[i] = (Qfloat)((this->*kernel_function)(idx, idx2));
				else				
					Q[i] = QD[idx];
			}						
		}
		return Q;
	}
	
	double dist2c_wc(int idx, int basisNum, int* basisIdx, double *coeff, double cNorm)
	{
		double dist = 0.0;		
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				dist += Q_i[j]*coeff[j];		
			dist  = cNorm - 2.0 * dist;				
		}		
		return dist;
	}
	
	Qfloat* get_QD() const { return QD; }	
	Qfloat get_Eta() const { return Eta; }
	Qfloat get_Kappa() const { return Eta-C_inv; }
	void swap_index(int i, int j) const { printf("CVDD_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	Qfloat C_inv;

	Qfloat Eta;	
	Qfloat *QD;

	sCache *kernelCache;
	mutable int kernelEval;	
};

//------------------------------------------------------------------------------------------------------------------


//
// Gram matrix of CVM
//
class CVC_Q : public Kernel
{
public:
	CVC_Q(const svm_problem* prob_, const svm_parameter* param_, schar *y_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		prob  = prob_;
		param = param_;
		y     = y_;
		QD    = new Qfloat[prob->l];		
		C_inv = (Qfloat)(1.0/(param->C));

		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{	
			Eta = (Qfloat)((this->*kernel_function)(0,0) + 1.0 + C_inv);
			for (i=0; i< prob->l; i++)
				QD[i] = Eta;
		}
		else
		{
			double tmp = 1.0 + C_inv;
			Eta        = 0.0;
			for (i=0; i< prob->l; i++)	
			{
				QD[i] = (Qfloat)((this->*kernel_function)(i, i) + tmp);
				if (QD[i] > Eta)
					Eta = QD[i];
			}	
		}		

		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~CVC_Q()
	{
		delete kernelCache;
		delete [] QD;		
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
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
					Q[i] = QD[idx];
			}						
		}
		return Q;
	}
	
	double dist2c_wc(int idx, int basisNum, int* basisIdx, double *coeff, double cNorm)
	{
		double dist = 0.0;		
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				dist += Q_i[j]*coeff[j];		
			dist  = cNorm - 2.0 * dist;				
		}		
		return dist;
	}
	
	Qfloat* get_QD() const { return QD; }	
	Qfloat get_Eta() const { return Eta; }
	Qfloat get_Kappa() const { return Eta-C_inv; }
	void swap_index(int i, int j) const { printf("CVC_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	schar* y;
	Qfloat C_inv;

	Qfloat Eta;	
	Qfloat *QD;

	sCache *kernelCache;
	mutable int kernelEval;	
};

//------------------------------------------------------------------------------------------------------------------

//
// Gram matrix of CVM
//
class CVR_Q : public Kernel
{
public:
	CVR_Q(const svm_problem* prob_, const svm_parameter* param_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		int numVar = 2*prob_->l;
		prob	   = prob_;
		param      = param_;
		QD         = new Qfloat[numVar];
		LinCoef    = new double[numVar];
		C_MU_inv   = (Qfloat)(param->mu*prob->l/param->C);	
		double tmp = 2.0/param->C;
		
		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{
			Kappa         = (Qfloat)((this->*kernel_function)(0,0) + 1.0);		
			double Kappa2 = Kappa + C_MU_inv;
			Eta           = 0.0;
			for (i=0; i < prob->l; i++)
			{
				QD[i]              =  (Qfloat)Kappa2;
				QD[i+prob->l]      =  (Qfloat)Kappa2;
				double tmpLinCoef  =  prob->y[i]*tmp;				
				LinCoef[i]		   =  tmpLinCoef;
				LinCoef[i+prob->l] = -tmpLinCoef;
				Eta                =  max(Eta, Kappa2+tmpLinCoef);
				Eta                =  max(Eta, Kappa2-tmpLinCoef);
			}
		}
		else
		{
			double tmp = 1.0 + C_MU_inv;
			Eta        = 0.0;
			Kappa      = 0.0;
			for (i=0; i < prob->l; i++)	
			{
				double Kappa2	   =  (this->*kernel_function)(i, i) + tmp;
				QD[i]			   =  (Qfloat)Kappa2;
				QD[i+prob->l]	   =  (Qfloat)Kappa2;
				double tmpLinCoef  =  prob->y[i]*tmp;			
				LinCoef[i]		   =  tmpLinCoef;
				LinCoef[i+prob->l] = -tmpLinCoef;
				Eta                =  max(Eta, Kappa2+tmpLinCoef);
				Eta                =  max(Eta, Kappa2-tmpLinCoef);
				if (Kappa2 > Kappa)
					Kappa = (Qfloat)Kappa2;
			}	
			Kappa -= C_MU_inv;
		}		
		for (i=0; i < numVar; i++)	
			LinCoef[i] += Eta;
				
		buffer[0]   = new Qfloat[numVar];
		buffer[1]   = new Qfloat[numVar];
		next_buffer = 0;
		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~CVR_Q()
	{
		delete kernelCache;		
		delete [] QD;		
		delete [] LinCoef;
		delete buffer[0];
		delete buffer[1];
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
	{
		assert(basisNum % 2 == 0);
		int numCache = basisNum/2;
		int si       = 1;
		int real_idx = idx;		
		if (real_idx >= prob->l)
		{
			real_idx -= prob->l;
			si        = -1;
		}
		
		int numRet;
		Qfloat *data = kernelCache->get_data(real_idx, numCache, numRet);
		if (data != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (numCache - numRet);
#endif
			// fill remaining		 
			int i;
			for(i = numRet; i < numCache; i++)
				data[i] = (Qfloat)((this->*kernel_function)(real_idx, basisIdx[2*i]) + 1.0);

			// reorder and copy
			Qfloat *buf = buffer[next_buffer];
			next_buffer = 1 - next_buffer;

			for(i=0; i < numCache; i++)
			{
				int bufIdx1  =  i*2;
				int bufIdx2  =  i*2+1;
				buf[bufIdx1] =  si*data[i];
				buf[bufIdx2] = -buf[bufIdx1];
				if (basisIdx[bufIdx1] == idx)
					buf[bufIdx1] = QD[idx];
				if (basisIdx[bufIdx2] == idx)
					buf[bufIdx2] = QD[idx];
			}					
			return buf;
		}
		else 
			return NULL;
	}
	
	double dot_c_wc(int idx, int basisNum, int* basisIdx, double *coeff)
	{
		assert(basisNum % 2 == 0);
		double dot   = 0.0;
		int numCache = basisNum/2;
		int si       = 1;
		int real_idx = idx;		
		if (real_idx >= prob->l)
		{
			real_idx -= prob->l;
			si        = -1;
		}		
		int numRet;
		Qfloat *data = kernelCache->get_data(real_idx, numCache, numRet);
		if (data != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (numCache - numRet);
#endif
			// fill remaining		 
			int i;
			for(i = numRet; i < numCache; i++)
				data[i] = (Qfloat)((this->*kernel_function)(real_idx, basisIdx[2*i]) + 1.0);

			for(i=0; i < numCache; i++)
				dot += data[i] * (coeff[i*2]-coeff[i*2+1]);			
			dot *= si;
		}		
		return dot;
	}
	
	double *get_LinCoef() const { return LinCoef; }
	Qfloat *get_QD() const { return QD; }	
	Qfloat get_Eta() const { return (Qfloat)Eta; }
	Qfloat get_Kappa() const { return Kappa; }
	void swap_index(int i, int j) const { printf("CVR_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	Qfloat C_MU_inv;

	double Eta;	
	Qfloat Kappa;
	Qfloat *QD;
	double *LinCoef;

	sCache *kernelCache;
	mutable int kernelEval;		
	mutable int next_buffer;		 // which buffer to fill	
	Qfloat *buffer[2];		 // sometimes, the outside program needs 2 columns (at the same time)	
};


//------------------------------------------------------------------------------------------------------------------

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
class Solver_Lag : public Solver
{
public:
	// constructor
	//
	// The index of coreset, core set size, gram matrix, linear coefficients, initial alpha
	Solver_Lag(int *CoreIdx, int numCore, const QMatrix& Q, const double* D, double *inAlpha, double Eps, double MinEps, int initSize = INITIAL_ALLOCATION_SIZE);
	~Solver_Lag();
		
	// The index of coreset, core set size, linear coefficients
	// return how many iteration used in SMO
	int Solve(int *CoreIdx, int numCore, const double* newD);
	double computeObj() const;
	double computeCNorm() const;
	const double* getGradient() const { return G; }	
	double *getAlpha() const { return alpha; }

private:
	double* vec_d;								// vector d, linear objective part		
	int allocatedSize;							// allocated space for storage	
	int *_CoreIdx;
	Qfloat *CacheQ_i;
	double minEps;

	int select_working_set(int &i, int &j);
	void update_alpha_status2(int i)
	{
		if(alpha[i] <= 0.0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	double get_C(int i) { return Cp; }
	double calculate_rho() {printf("not necessary to compute rho\n"); exit(-1);}
	void do_shrinking() {printf("not implemented\n"); exit(-1);}	
};

//------------------------------------------------------------------------------------------------------------------

//
// Base solver for core set approximation algorithm
//
class Solver_Core
{
public:
	Solver_Core() {}
	virtual ~Solver_Core() {}

	void Init(const svm_problem* prob, const svm_parameter* param);
	bool Create(double cvm_eps);
	int Solve(int num_basis, double cvm_eps);
	double GetCoreNorm2 () const { return coreNorm2; } 
	virtual bool IsExitOnMaxIter() const = 0;
	virtual double ComputeRadius2() = 0;
	virtual double ComputeSolution(double *alpha, double Threshold) = 0;

protected:
	virtual void _Init() = 0;
	virtual bool _Create(double cvm_eps) = 0;
	virtual double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx) = 0;
	virtual void _UpdateCoreSet(int maxDistance2Idx) = 0;	

	Solver_Lag          *solver;
	const svm_parameter *param;
	const svm_problem   *prob;

	int    maxNumBasis;
	int   *coreIdx;
	int    coreNum;
	double coreNorm2;
	double r2;	

	double *outAlpha;
	double  tmpAlpha[2*INITIAL_CS];
	double  tempD   [2*INITIAL_CS];
};

//------------------------------------------------------------------------------------------------------------------

//
// Solver for CVDD
//
class Solver_CVDD : public Solver_Core
{
public:
	Solver_CVDD() {}
	~Solver_CVDD()
	{
		// free memory			
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum >= min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = -2.0*solver->computeObj();
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:	
	CVDD_Q *kernelQ;
	Qfloat *QD;
	char  *chklist;	
};

//------------------------------------------------------------------------------------------------------------------

//
// Solver for CVM
//
class Solver_CVM : public Solver_Core
{
public:
	Solver_CVM() {}
	~Solver_CVM()
	{
		// free memory	
		delete [] posIdx;
		delete [] negIdx;
		delete [] y;
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum >= min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = Eta - coreNorm2;
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:
	int posNum;
	int negNum;	
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;
	bool posTurn;	
	schar *y;
	CVC_Q *kernelQ;
	char  *chklist;
	double Eta;
};

//------------------------------------------------------------------------------------------------------------------

//
// Solver for CVR
//
class Solver_CVR : public Solver_Core
{
public:
	Solver_CVR() {}
	~Solver_CVR()
	{
		// free memory	
		delete [] posIdx;
		delete [] negIdx;
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum/2 >= min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = -2.0*solver->computeObj();
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:
	int posNum;
	int negNum;	
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;
	int numVar;
	bool posTurn;
	CVR_Q *kernelQ;
	Qfloat *QD;
	char  *chklist;
	double *LinCoef;
};



#endif //_CVM_H_