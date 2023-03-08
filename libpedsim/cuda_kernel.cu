#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

/* ---------------------------
	SET HEATMAP FUNCTIONS
-----------------------------*/

void Ped::Model::setupHeatmapCuda()
{
	// Allocate memory on CPU
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.

	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));
	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	// Initialize values, point to right memory
	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE*i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	desiredX = (float*)malloc(agents.size()*sizeof(float));
	desiredY = (float*)malloc(agents.size()*sizeof(float));


	cudaStatus = cudaMalloc(&d_desiredX, agents.size()*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "malloc d_desiredX fail\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {printf("success malloc X\n");}

	cudaStatus = cudaMalloc(&d_desiredY, agents.size()*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "malloc d_desiredX fail\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {printf("success malloc Y\n");}

	// Allocate memory on GPU
	cudaStatus = cudaMalloc(&d_heatmap, SIZE*sizeof(int));
	cudaStatus = cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*sizeof(int));
	cudaStatus = cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*sizeof(int));

	// Copy memory from host to device
	cudaStatus = cudaMemcpy(d_heatmap, heatmap, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "copy scaled fail\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {printf("success malloc Y\n");}
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap, SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap, SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
Error:
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda heatmap setup fail.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda heatmap setup succeeded.\n"); // This is a good thing
	}
}

/* ---------------------------
	UPPDATE HEATMAP FUNCTIONS
  --------------------------*/ 

__global__ void kernel_fade(int *dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x < SIZE && y < SIZE)
	{
		int id = y*SIZE + x;
		dev_heatmap[id] = (int)round(dev_heatmap[id] * 0.80);
	}
}

__global__ void kernel_agents(int *dev_heatmap, int size_agents, float *desiredX, float *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size_agents){
		int x = desiredX[i];
		int y = desiredY[i];

		if(x>=0 && x<SIZE && y>=0 && y<SIZE)
			atomicAdd(&dev_heatmap[y*SIZE + x], 40);
	}
}

__global__ void kernel_clip(int *dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x < SIZE && y < SIZE){
		int id = y*SIZE + x;
		dev_heatmap[id] = dev_heatmap[id] < 255 ? dev_heatmap[id] : 255;
	}
}

__global__ void kernel_scale(int *dev_heatmap, int *dev_scaled_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < SCALED_SIZE && y < SCALED_SIZE)
	{
		int id = y*SIZE + x;
		int value = dev_heatmap[id];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
				int s_y = y * CELLSIZE + cellY;
                int s_x = x * CELLSIZE + cellX;
				dev_scaled_heatmap[s_y*SCALED_SIZE + s_x] = value;
			}
		}

	}
}

__global__ void kernel_blur(int *dev_heatmap, int *dev_blurred_heatmap, int *dev_scaled_heatmap)
{
	//weights for blur
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x >= 2 && x < SCALED_SIZE && y >= 2 && y < SCALED_SIZE)
	{
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				sum += w[2 + k][2 + l] * dev_scaled_heatmap[(y + k) * SCALED_SIZE + (x + l)];
			}
		}
		int value = sum / WEIGHTSUM;
		dev_blurred_heatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
	}
}


void Ped::Model::updateHeatmapCuda() 
{
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_desiredX, desiredX, agents.size()*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "fail copy desiredx\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {
		fprintf(stderr, "success copy desiredx\n");
	}

	cudaStatus = cudaMemcpy(d_desiredY, desiredY, agents.size()*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "fail copy desiredy\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {
		fprintf(stderr, "success copy desiredy\n");
	}

	// Fade heatmap
	kernel_fade<<<CELLSIZE, SIZE>>>(d_heatmap);
	kernel_agents<<<1, agents.size()>>>(d_heatmap, agents.size(), d_desiredX, d_desiredY);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "fail copy desiredy\n");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	} else {
		fprintf(stderr, "x\n");
		fprintf(stderr, "%f.\n", desiredX[0]);
		fprintf(stderr, "y\n");
		fprintf(stderr, "%f.\n", desiredY[0]);

	}

	//Clip heatmap
	kernel_clip<<<1, SIZE>>>(d_heatmap);

	//Scale heatmap
	kernel_scale<<<1, SIZE>>>(d_heatmap, d_scaled_heatmap);

	// // Blur heatmap
	kernel_blur<<<1, SIZE>>>(d_heatmap, d_blurred_heatmap, d_scaled_heatmap);

	//cudaStreamSynchronize();
	// Copy memory from host to device
	cudaStatus = cudaMemcpy(heatmap, d_heatmap, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(scaled_heatmap, d_scaled_heatmap, SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(blurred_heatmap, d_blurred_heatmap, SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);



Error:
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda update succeeded.\n"); // This is a good thing
	}

}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}

