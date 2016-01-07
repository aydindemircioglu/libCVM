#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <assert.h>

#include "cvm.h"

#define CACHE_DELTA 10


//
// Sparse caching for kernel evaluations
//
// l is the number of total data items
// size is the cache size limit in bytes
//
sCache::sCache(const svm_parameter* param_, int num)
{
	// init cache and usage
	numData  = num;
	head     = (shead_t *) calloc(numData,sizeof(shead_t));	// initialized to 0	
	lru_head.next = lru_head.prev = &lru_head;	

	// compute maximal space
	maxItem = max((int)((max(param_->cache_size,4.0)*(1<<20))/sizeof(Qfloat)),numData*2); // cache must be large enough for two columns		
}

sCache::~sCache() 
{ 		
	for(shead_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);		
	free(head);	
}

void sCache::lru_delete(shead_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void sCache::lru_insert(shead_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

Qfloat* sCache::get_data(int idx,  int len, int& numRet)
{	
	shead_t *h = &head[idx];
	h->refcount ++;

	if(h->len > 0) lru_delete(h);
	if(len > h->max_len)
	{
		int more   = len + CACHE_DELTA - h->max_len;
		h->max_len = h->max_len + more;
		
		// free old space
		while(maxItem < more)
		{
			shead_t *old = lru_head.next;
			lru_delete(old);

			if (old->refcount <=0)
			{
				free(old->data);
				old->data    = NULL;
				maxItem     += old->len;				
				old->len     = 0;	
				old->max_len = 0;
			}
			else
			{
				old->refcount --;
				lru_insert(old);
			}
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*h->max_len);
		if (h->data == NULL)
		{	
			while(h->data == NULL && lru_head.next != &lru_head)
			{
				shead_t *old = lru_head.next;
				lru_delete(old);

				if (old->refcount <= 0)
				{
					free(old->data);
					old->data    = NULL;
					maxItem     += old->len;				
					old->len     = 0;	
					old->max_len = 0;
					h->data = (Qfloat *)calloc(h->max_len,sizeof(Qfloat));
				}
				else
				{
					old->refcount --;
					lru_insert(old);
				}
			}
			h->len  = 0;
			if (h->data == NULL)
			{
				printf ("sCache cannot allocate memory!\n");
				return NULL;
			}
		}
		maxItem -= more;		
	}

	lru_insert(h);
	numRet = h->len;
	h->len = len;
	return h->data;
}

//---------------------------------------------------------------------------------------------------------------------

void Solver_CVDD::_Init()
{
	// count the distribution of data	
	chklist = new char[prob->l];
	coreIdx = new int[prob->l];
	for(int i = 0; i < prob->l; i++)
	{	
		chklist[i] = 0;
		coreIdx[i] = -1;
	}

	// initialized the kernel
	outAlpha = &tmpAlpha[0];
	kernelQ  = new CVDD_Q(prob, param);		
	QD       = kernelQ->get_QD();
	coreNum  = 0;	

	// choose a small subset as initialization	
	for(int sampleNum = 0; sampleNum < INITIAL_CS; sampleNum++)
	{	
		int idx;
		do
		{	
			int rand32bit = random();
			idx			  = rand32bit % prob->l;
		} while (chklist[idx] > 0);
		chklist[idx]        = 1;		
		coreIdx[coreNum++]  = idx;
		outAlpha[sampleNum] = 1.0/INITIAL_CS;
		tempD[sampleNum]    = QD[idx];
	}	
}

bool Solver_CVDD::_Create(double cvm_eps)
{
	solver = new Solver_Lag(coreIdx,coreNum,*kernelQ,tempD,outAlpha,SMO_EPS,param->eps);
	return (solver != NULL);
}

double Solver_CVDD::_maxDistFromSampling(double maxDistance2, int &maxDistance2Idx)
{	
	for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
	{	
		int idx;
		do
		{	
			int rand32bit = random();
			idx			  = rand32bit % prob->l;
		} while (chklist[idx] > 0);
		double dist2 = kernelQ->dist2c_wc(idx, coreNum, coreIdx, outAlpha, coreNorm2 + QD[idx]);
		if (dist2 > maxDistance2)
		{						
			maxDistance2 = dist2;
			maxDistance2Idx = idx;
		}					
	}
	return maxDistance2;
}

inline void Solver_CVDD::_UpdateCoreSet(int maxDistance2Idx)
{	
	// update center
	chklist[maxDistance2Idx] = 1;
	coreIdx[coreNum++]       = maxDistance2Idx;	
	tempD[0]                 = QD[maxDistance2Idx];
}	

double Solver_CVDD::ComputeSolution(double *alpha, double Threshold)
{	
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i++)
	{
		if (outAlpha[i] > Threshold)
		{
			int ii    = coreIdx[i];
			alpha[ii] = outAlpha[i];	
			sumAlpha += outAlpha[i];
		}
	}	
	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] /= sumAlpha;
	return (r2/sumAlpha);
}



//---------------------------------------------------------------------------------------------------------------------

inline bool PosNeg(bool PosTurn, int pNum, int nNum)
{
	if (pNum<=0)
		PosTurn = false;
	else if (nNum<=0)
		PosTurn = true;
	else
		PosTurn = !PosTurn;
	return PosTurn;
}

void Solver_CVM::_Init()
{
	// count the distribution of data
	posIdx  = new int[prob->l];
	posNum	= 0;
	negIdx  = new int[prob->l];
	negNum	= 0;
	y       = new schar[prob->l];
	chklist = new char[prob->l];
	coreIdx = new int[prob->l];
	for(int i = 0; i < prob->l; i++)
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
		chklist[i] = 0;
		coreIdx[i] = -1;
	}

	// initialized the kernel
	outAlpha = &tmpAlpha[0];
	kernelQ  = new CVC_Q(prob, param, y);		
	Eta		 = kernelQ->get_Eta();	
	pNum     = posNum;
	nNum     = negNum;
	coreNum  = 0;

	// choose a small subset as initialization
	posTurn  = true;		
	for(int sampleNum = 0; sampleNum < INITIAL_CS; sampleNum++)
	{					
		posTurn = PosNeg(posTurn,pNum,nNum);
		int idx;
		do
		{
			// balanced and random sample
			int rand32bit = random();
			idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
		} while (chklist[idx] > 0);
		chklist[idx] = 1;
		if (y[idx] > 0)
			pNum--;
		else 
			nNum--;
		coreIdx[coreNum++]  = idx;
		outAlpha[sampleNum] = 1.0/INITIAL_CS;
		tempD[sampleNum]    = 0.0;
	}	
}

bool Solver_CVM::_Create(double cvm_eps)
{
	solver = new Solver_Lag(coreIdx,coreNum,*kernelQ,tempD,outAlpha,SMO_EPS,param->eps);
	return (solver != NULL);
}

double Solver_CVM::_maxDistFromSampling(double maxDistance2, int &maxDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	posTurn        = true;
	for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
	{					
		posTurn = PosNeg(posTurn,pNum,nNum);
		int idx;
		do
		{
			// balanced and random sample
			int rand32bit = random();
			idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
		} while (chklist[idx] > 0);
		double dist2 = kernelQ->dist2c_wc(idx, coreNum, coreIdx, outAlpha, tmpdist);
		if (dist2 > maxDistance2)
		{						
			maxDistance2 = dist2;
			maxDistance2Idx = idx;
		}					
	}
	return maxDistance2;
}

inline void Solver_CVM::_UpdateCoreSet(int maxDistance2Idx)
{
	if (y[maxDistance2Idx] > 0) 
		pNum--;
	else
		nNum--;

	// update center
	chklist[maxDistance2Idx] = 1;
	coreIdx[coreNum++]       = maxDistance2Idx;	
}	

double Solver_CVM::ComputeSolution(double *alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i++)
	{
		if (outAlpha[i] > Threshold)
		{
			int ii    = coreIdx[i];
			alpha[ii] = outAlpha[i]*y[ii];
			bias     += alpha[ii];
			sumAlpha += outAlpha[i];
		}
	}
	bias /= sumAlpha;

	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] /= sumAlpha;
	return bias;
}

//---------------------------------------------------------------------------------------------------------------------

void Solver_CVR::_Init()
{
	// count the distribution of data		
	posIdx  = new int[prob->l];
	posNum  = 0;
	negIdx  = new int[prob->l];
	negNum  = 0;
	chklist = new char[prob->l];
	int i;
	for(i = 0; i < prob->l; i++)
	{	
		if (prob->y[i] > 0.0)
			posIdx[posNum++] = i;
		else
			negIdx[negNum++] = i;
		chklist[i] = 0;		
	}
	pNum    = posNum;
	nNum    = negNum;	
	numVar  = prob->l*2;
	coreIdx = new int[numVar];
	for(i = 0; i < numVar; i++)
		coreIdx[i] = -1;
	printf("begin to construct kernel!\n");
	
	// initialized the kernel
	outAlpha = &tmpAlpha[0];
	kernelQ  = new CVR_Q(prob, param);	
	QD       = kernelQ->get_QD();
	LinCoef  = kernelQ->get_LinCoef();	
	coreNum  = 0;

	// choose a small subset as initialization
	posTurn = true;		
	for(int sampleNum = 0; sampleNum < INITIAL_CS; sampleNum++)
	{					
		posTurn = PosNeg(posTurn,pNum,nNum);
		int idx;
		do
		{
			// balanced and random sample
			int rand32bit = random();
			idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
		} while (chklist[idx] > 0);
		chklist[idx] = 1;
		if (prob->y[idx] > 0.0)
			pNum--;
		else 
			nNum--;

		coreIdx[coreNum]    = idx;
		outAlpha[coreNum]   = 1.0/INITIAL_CS;
		tempD[coreNum]      = LinCoef[idx];
		coreIdx[coreNum+1]  = idx + prob->l;
		outAlpha[coreNum+1] = 0.0;
		tempD[coreNum+1]    = LinCoef[idx + prob->l];
		coreNum            += 2;
	}		
}

bool Solver_CVR::_Create(double cvm_eps)
{
	solver = new Solver_Lag(coreIdx,coreNum,*kernelQ,tempD,outAlpha,SMO_EPS,param->eps);
	return (solver != NULL);
}

double Solver_CVR::_maxDistFromSampling(double maxDistance2, int &maxDistance2Idx)
{	
	posTurn = true;
	for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
	{					
		posTurn = PosNeg(posTurn,pNum,nNum);
		int idx;
		do
		{
			// balanced and random sample
			int rand32bit = random();
			idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
		} while (chklist[idx] > 0);		
		double tmp   = 2.0 * kernelQ->dot_c_wc(idx, coreNum, coreIdx, outAlpha);
		double dist2 = max(coreNorm2-tmp+LinCoef[idx], coreNorm2+tmp+LinCoef[idx+prob->l]);
		if (dist2 > maxDistance2)
		{						
			maxDistance2    = dist2;
			maxDistance2Idx = idx;
		}		
	}
	return maxDistance2;
}

inline void Solver_CVR::_UpdateCoreSet(int maxDistance2Idx)
{	
	if (prob->y[maxDistance2Idx] > 0.0) 
		pNum--;
	else
		nNum--;

	// update center
	int maxDistance2Idx2     = maxDistance2Idx + prob->l;
	chklist[maxDistance2Idx] = 1;
	coreIdx[coreNum]         = maxDistance2Idx;	
	coreIdx[coreNum+1]       = maxDistance2Idx2;	
	coreNum                 += 2;
	tempD[0]                 = LinCoef[maxDistance2Idx];
	tempD[1]                 = LinCoef[maxDistance2Idx2];
}	

double Solver_CVR::ComputeSolution(double *alpha, double Threshold)
{	
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i+=2)
	{
		int ii = coreIdx[i];
		if (outAlpha[i] > Threshold)		
			sumAlpha += outAlpha[i];
		else
			outAlpha[i] = 0.0;
		if (outAlpha[i+1] > Threshold)		
			sumAlpha += outAlpha[i+1];
		else
			outAlpha[i+1] = 0.0;
		double tmpAlpha = outAlpha[i] - outAlpha[i+1];
		alpha[ii] = tmpAlpha * param->C;
		bias += alpha[ii];
	}	

	bias /= sumAlpha;
	for(i = 0; i < coreNum; i+=2)
		alpha[coreIdx[i]] /= sumAlpha;

	return bias;
}


//---------------------------------------------------------------------------------------------------------------------

void solve_cvdd(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si)
{
	// info
	info("num pattern = %ld\n", prob->l);
	
	// init		
	srandom(0);	
	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

	// solve CVDD
	Solver_CVDD solver;
	solver.Init(prob,param);
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = (PREDICT_BND/(solver.GetKappa())+2.0/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	printf("epsilon = %.8g\n",cvm_eps);
	bool flag = solver.Create(cvm_eps);	
	if (flag)
	{
		int coreNum = solver.Solve(param->num_basis,cvm_eps);

		// compute solution vector		
		double THRESHOLD = 1e-5/coreNum;
		double bias      = solver.ComputeSolution(alpha, THRESHOLD);	

		// info in CVDD
		si->obj    = solver.ComputeRadius2();
		si->rho    = si->obj;
		si->margin = solver.GetCoreNorm2();
	}
}

//---------------------------------------------------------------------------------------------------------------------

void solve_cvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	// info
	info("num pattern = %ld\n", prob->l);
	
	// init		
	srandom(0);	
	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

	// solve CVM
	Solver_CVM solver;
	solver.Init(prob,param);
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = (PREDICT_BND/(solver.GetKappa())+2.0/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	printf("epsilon = %.8g\n",cvm_eps);
	bool flag = solver.Create(cvm_eps);	
	if (flag)
	{
		int coreNum = solver.Solve(param->num_basis,cvm_eps);

		// compute solution vector		
		double THRESHOLD = 1e-5/coreNum;
		double bias      = solver.ComputeSolution(alpha, THRESHOLD);	
		double coreNorm2 = solver.GetCoreNorm2();

		// info in CVM		
		si->obj    = 0.5*coreNorm2;
		si->rho    = -bias;
		si->margin = coreNorm2;
	}
}

//---------------------------------------------------------------------------------------------------------------------

void solve_cvr(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si)
{
	// info
	info("num pattern = %ld\n", prob->l);
	
	// init		
	srandom(0);	
	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

	// solve CVR
	Solver_CVR solver;
	solver.Init(prob,param);
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = (PREDICT_BND/(solver.GetKappa())+2.0*param->mu*prob->l/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	printf("epsilon = %.8g\n",cvm_eps);
	bool flag = solver.Create(cvm_eps);	
	if (flag)
	{
		int coreNum = solver.Solve(param->num_basis,cvm_eps);
		printf("finish the solver\n");

		// compute solution vector				
		double THRESHOLD = min(solver.GetEta()/param->C, 1e-5/coreNum);
		double bias      = solver.ComputeSolution(alpha, THRESHOLD);	

		// info in CVR
		si->obj    =  solver.ComputeRadius2() - solver.GetEta();
		si->rho    = -bias;
		si->margin =  solver.GetCoreNorm2();		
	}
}
