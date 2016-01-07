#ifndef UTILITY_H_kldfljkflajkfldjksljkfsdaklfdsakl
#define UTILITY_H_kldfljkflajkfldjksljkfsdaklfdsakl

#ifdef WIN32
	#include <crtdbg.h>
#endif
#include <math.h>
#include <assert.h>


//-------------------- constant -----------------------------------------------------
#define INF						HUGE_VAL									// a big number
#define EPSILON					1e-8										// a small number
#define INITIAL_ALLOCATION_SIZE 3000										// how many space allocated initially
#define NEAR_ZERO(x)			( fabs(x) <= EPSILON )
#define TAU						1e-12

//-------------------- helper function -----------------------------------------------------
// generate a non-negative 30-bit random number

int delta(int x, int y);													// delta function
double dotProduct( const double* x1, const double* x2, int length );		// dot product between x1 and x2

template<class T>
T sum( const T* x, int length)												// sum of components of the vector x
{
	T sum = 0;
	for (int i=0; i<length; i++)
		sum += x[i];
	return sum;
}

#define SQUARE(x) ( (x) * (x) )												// square function

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))						// allocating memory


//#if defined(linux) | defined(WIN32)
template <class T> inline void swap_2(T& x, T& y) { T t=x; x=y; y=t; }		// swap function
//#endif


//------------------------------------------ Debug -------------------------------------------------------------------
// The following macros set and clear, respectively, given bits
// of the C runtime library debug flag, as specified by a bitmask.
#ifdef WIN32
	#ifdef   _DEBUG
		#define  SET_CRT_DEBUG_FIELD(a) _CrtSetDbgFlag((a) | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))
		#define  CLEAR_CRT_DEBUG_FIELD(a) _CrtSetDbgFlag(~(a) & _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG))
	#else
		#define  SET_CRT_DEBUG_FIELD(a)   ((void) 0)
		#define  CLEAR_CRT_DEBUG_FIELD(a) ((void) 0)
	#endif
#endif



double getRunTime();



#endif //UTILITY_H_kldfljkflajkfldjksljkfsdaklfdsakl


