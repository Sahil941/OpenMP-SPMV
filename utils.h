#ifndef UTILS_H
#define UTILS_H
#include <time.h>
typedef struct{
	time_t tv_sec; /* seconds */
	long tv_nsec; /* nanoseconds */
}timespect;

int checkerror(const float *resp, const float *ress, int dim);
void getmul(const float *val, const float *vec, const int *rIndex,
	const int *cIndex, int nz, float *res);

#endif
