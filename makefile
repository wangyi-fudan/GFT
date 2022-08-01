all:	train gft 
train:	*.cu makefile model.config
	/usr/local/cuda-11.5/bin/nvcc train.cu -o train -arch=sm_80 -lcublas -Xptxas -O3
gft:	*.cpp *.hpp makefile model.config
	g++ gft.cpp -o gft -march=native -Ofast -Wall -s -static

