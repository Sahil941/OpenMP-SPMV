CC = /usr/local/cuda-7.5/bin/nvcc
objects = mmio.o spmv_N_thread_static.c mmio.c utils.c spmv_atomic_kernel.cu

all: $(objects)
	$(CC) -arch=sm_20 $(objects) -o spmv

%.o: %.c
	$(CC) -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o spmv