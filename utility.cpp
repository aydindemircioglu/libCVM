#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef _MSC_VER
#include <sys/times.h>
#include <unistd.h>
#else
#include <time.h>
#endif



//----------------------------------------------------------------------------------
// Global helper functions

int delta(int x, int y)
{
	return ( (x==y)? (1) : (0) );
}

double dotProduct( const double* x1, const double* x2, int length )
{
	double sum = 0;
	for (int i=0; i<length; i++)
		sum += ( x1[i] * x2[i] );

	return sum;
}




double getRunTime()
{
#ifdef _MSC_VER
  clock_t current = clock();
  return (double)(current) / CLOCKS_PER_SEC;
#else
  struct tms current;
  times(&current);
  
  double norm = (double)sysconf(_SC_CLK_TCK);
  return(((double)current.tms_utime)/norm);
#endif
}


