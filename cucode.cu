/*
*********************************************
*  314 Principles of Programming Languages  *
*  Fall 2016                                *
*********************************************
*
*   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file
*   and a vector from a txt file, perform matrix multiplication and store the
*   result to output.txt. This is the parallel and static version of sparse matrix vector
*   multiplication.
*
*

*
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h> 
#include <sys/time.h>


extern "C" {
	#include "mmio.h"
	#include "utils.h"
}

//sorting according to the index
void quicksort(float* a, float* vindex, int* rindex, int* cindex, int n)
{
	int i, j, m;
	float p, t, s;
	if (n < 2)
		return;
	p = vindex[n / 2];

	for (i = 0, j = n - 1;; i++, j--) {
		while (vindex[i]<p)
			i++;
		while (p<vindex[j])
			j--;
		if (i >= j)
			break;
		t = a[i];
		a[i] = a[j];
		a[j] = t;

		s = vindex[i];
		vindex[i] = vindex[j];
		vindex[j] = s;

		m = rindex[i];
		rindex[i] = rindex[j];
		rindex[j] = m;

		m = cindex[i];
		cindex[i] = cindex[j];
		cindex[j] = m;
	}
	quicksort(a, vindex, rindex, cindex, i);
	quicksort(a + i, vindex + i, rindex + i, cindex + i, n - i);
}

__global__ void spmv_atomic_kernel(int nnz, int *coord_row, int *coord_col, float *A, float *x, float *y){
	int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
	int thread_num = blockDim.x * gridDim.x;
	int iter = 1;
	if (nnz % thread_num){
		iter = (nnz % thread_num) + 1;
	}else{
		iter = (nnz % thread_num);
	}

	for (int i = 0; i < iter; i++){
		int data_id = thread_id + i * thread_num;
		if (data_id < nnz){
			float data = A[data_id];
			int row = coord_row[data_id];
			int col = coord_col[data_id];
			float temp = data * x[col];
			atomicAdd(&y[row], temp);
		}
	}
}

int main(int argc, char *argv[])
{
	int ret_code;
	MM_typecode matcode;
	FILE *f;
	int M, N, nz;   //M is row number, N is column number and nz is the number of entry
	int i, vecdim, *rIndex, *cIndex, *rsIndex, *reIndex;
	float *val, *res, *vec, *vIndex;
	if (argc < 7)
	{
		fprintf(stderr, "Usage: %s -mat [martix-market-filename] -ivec [input-vector-filename] -alg [alg-option] -blocksize [thread-blocksize]\n", argv[0]);
		exit(1);
	}

	printf("\nOpening input matrix file: %s\n", argv[2]);
	if ((f = fopen(argv[2], "r")) == NULL)
	{
		printf("Fail to open the input matrix file!\n");
		exit(1);
	}
	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
		mm_is_sparse(matcode))
	{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		exit(1);
	}

	/* find out size of sparse matrix .... */
	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
		exit(1);

	/* reseve memory for matrices */
	rIndex = (int *)malloc(nz * sizeof(int));
	cIndex = (int *)malloc(nz * sizeof(int));
	val = (float *)malloc(nz * sizeof(float));

	/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
	for (i = 0; i<nz; i++)
	{
		fscanf(f, "%d %d %g\n", &rIndex[i], &cIndex[i], &val[i]);
		rIndex[i]--;  /* adjust from 1-based to 0-based */
		cIndex[i]--;
	}

	if (f != stdin) fclose(f);

	
	printf("Opening input vector file: %s\n", argv[4]);
	//open and load the vector input  
	if ((f = fopen(argv[4], "r")) == NULL)
	{
		printf("Fail to open the input vector file!\n");
		exit(1);
	}
	fscanf(f, "%d\n", &vecdim);
	if (vecdim != M)
	{
		printf("dimension mismatch!\n");
		exit(1);
	}
	vec = (float*)malloc(vecdim * sizeof(float));
	for (i = 0; i<vecdim; i++)
	{
		fscanf(f, "%g\n", &vec[i]);
	}
	if (f != stdin) fclose(f);

	//the original calculation result
	float* res_seq = (float*)malloc(M*sizeof(float));
	memset(res_seq, 0, M*sizeof(float));

	getmul(val, vec, rIndex, cIndex, nz, res_seq);

	vIndex = (float*)malloc(nz*sizeof(float));
	memset(vIndex, 0, nz*sizeof(float));
	for (i = 0; i < nz; i++)
	{
		vIndex[i] = (float)rIndex[i] * N + cIndex[i];
		if (vIndex[i] < 0)
		{
		    printf("Error!\n");
		    exit(1);
		}
	}

	quicksort(val, vIndex, rIndex, cIndex, nz);
	
	printf("%f\n", *res_seq);

	//We use rsIndex/reIndex to keep the start/end position of each row. The intial values are 
	//-1 or -2 for all entries.  rsIndex[i] indicates the start poistion of the i-th row. Hence 
	//the position index of the i-th row is from rsIndex[i] to reIndex[i]
	rsIndex = (int*)malloc(M*sizeof(int)); //start/end position of each row
	memset(rsIndex, -1, M*sizeof(int));

	reIndex = (int*)malloc(M*sizeof(int));
	memset(reIndex, -2, M*sizeof(int));
       
	for (i = 0; i<nz; i++)
	{
		int tmp = (int)(vIndex[i] / N);
		if (rsIndex[tmp] == -1)
		{
			rsIndex[tmp] = i;
			reIndex[tmp] = i;
		}
		else
			reIndex[tmp] = i;
	}
	//int thread_num = atoi(argv[3]);
	//omp_set_num_threads(thread_num); 

	
  printf("\n Start computation ... \n");
	struct timeval start, end;
	


	gettimeofday(&start, NULL);
  /************************/
	/* now calculate the multiplication */
	/************************/
	res = (float *)malloc(M*sizeof(float));
	memset(res, 0, M*sizeof(float));

	// Your OpenMP pragma should be inserted for one or both loops below.
	// You need to determine which loop is safe to be parallelized.
	// You will also need to use correct parallelization parameters. Please use a
	// static schedule for this parallelization stratey.
	
	int blockSize = 1024;
	int *d_rIndex;
	int *d_cIndex;
	float *d_val;
	float *d_vec;
	float *d_res;

	if (argc == 9){
		if (strcmp(argv[7], "-blocksize") == 0){
			blockSize = atoi(argv[8]);
		}
	}

	printf("Before cudaMalloc \n");
	cudaMalloc((void **)&d_rIndex, (sizeof(int) * blockSize));
	cudaMalloc((void **)&d_cIndex, (sizeof(int) * blockSize));
	cudaMalloc((void **)&d_val, (sizeof(float) * blockSize));
	cudaMalloc((void **)&d_vec, (sizeof(float) * blockSize));
	cudaMalloc((void **)&d_res, (sizeof(float) * blockSize));
	printf("After cudaMalloc \n");

	cudaMemcpy(d_rIndex, rIndex, (sizeof(int) * blockSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cIndex, cIndex, (sizeof(int) * blockSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, (sizeof(float) * blockSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec, (sizeof(float) * blockSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res, (sizeof(float) * blockSize), cudaMemcpyHostToDevice);
	printf("After cudaMemcpy \n");

	spmv_atomic_kernel<<<1, blockSize>>>(nz, d_rIndex, d_cIndex, d_val, d_vec, d_res);
	
	cudaMemcpy(res, d_res, (sizeof(float) * blockSize), cudaMemcpyDeviceToHost);

	cudaFree(d_rIndex);
	cudaFree(d_cIndex);
	cudaFree(d_val);
	cudaFree(d_vec);
	cudaFree(d_res);
	
	

	gettimeofday(&end, NULL);

	
  printf(" End of computation ... \n\n");
  printf("%f\n", *res);
  
  long elapsed_time = ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec));

		
	if (!checkerror(res, res_seq, M))
	{
		printf("Calculation Error!\n");
		exit(1);
	}
	else {
		printf(" Test Result Passed ... \n");
	}


	printf(" Static Parallelization Total time: %ld micro-seconds\n\n",  elapsed_time);

	if (!checkerror(res, res_seq, M))
	{
		printf("Calculation Error!\n");
		exit(1);
	}

	// save the result
	if ((f = fopen("output.txt", "w")) == NULL)
	{
		printf("Fail to open the output file!\n");
		exit(1);
	}
	for (i = 0; i<vecdim; i++)
	{
		fprintf(f, "%lg\n", res[i]);
	}
	fclose(f);

	free(res_seq);
	free(vIndex);
	free(res);
	free(vec);
	free(rIndex);
	free(cIndex);
	free(val);
	free(rsIndex);
	free(reIndex);
	return 0;
}