OPT= 
DEVICE_EMU= #-deviceemu
CFLAGS=$(OPT) -fopenmp
NVFLAGS=$(OPT) -arch=sm_13 $(DEVICE_EMU)
LDFLAGS=$(OPT)  
CC=g++
NVCC=nvcc 
//NVCC = /usr/local/cuda/bin/nvcc
//NVCC = /usr/local/cuda/cuda/bin/nvcc
LD=nvcc $(DEVICE_EMU)
OBJS=main.o

.PHONY: main 

main: $(OBJS)
	$(NVCC) $(LDFLAGS) -o main $(OBJS)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $*.o
.cpp.o :
	$(NVCC) $(NVFLAGS) -c $< -o $*.o
%.o : %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $*.o 

clean:
	-rm main $(OBJS)
