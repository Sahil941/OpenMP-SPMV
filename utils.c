#include "utils.h"
#include <stdio.h>
#include <math.h>
void getmul(const float* val, const float* vec, const int* rIndex, const int*cIndex, int nz, float* res)
{
	int i; 
	for (i = 0; i < nz; i++)
	{
		int rInd = rIndex[i];
		int cInd = cIndex[i];
		res[rInd] += val[i] * vec[cInd];
	}
}

int checkerror(const float* resp, const float* ress, int dim)
{
	int i;
	for (i = 0; i < dim; i++)
	{
		if (fabs(( ((double)resp[i]) - ((double)ress[i]) )/((double)resp[i]) ) == 0){
			return 1;
		}
		if (fabs(( ((double)resp[i]) - ((double)ress[i]) )/((double)resp[i]) ) > 1E-6){
			return 0;
		}
	}

	return 1;

}