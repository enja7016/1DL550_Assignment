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

__global__ void setupHeatmap(int* hm, int **heatmap)
{
	int i = blockIdx.x * blockDim.x + threadId.x;
	heatmap[i] = hm + SIZE * i;
}

__global__ void setupScaledHeatmap(int *shm, int** scaled_heatmap)
{
	int i = blockIdx.x * blockDim.x + threadId.x;
	scaled_heatmap[i] = shm + SCALED_SIZE*I;
}

void Ped::Model::setupHeatmapCuda()
{
	
	int *hm, *shm, *bhm;
	cudaMalloc(&hm, SIZE * SIZE * sizeof(int));
    cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMalloc(&heatmap, SIZE * sizeof(int*));
    cudaMalloc(&scaled_heatmap, SCALED_SIZE * sizeof(int*));
	cudaMallocHost(&blurred_heatmap, SCALED_SIZE * sizeof(int*));
	cudaMallocHost(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));

	cudaMalloc(&desiredX, agents.size() * sizeof(int));
    cudaMalloc(&desiredY, agents.size() * sizeof(int));


	cudaMemset(hm, 0, SIZE * SIZE);
    cudaMemset(shm, 0, SCALED_SIZE * SCALED_SIZE);
    cudaMemset(bhm, 1, SCALED_SIZE * SCALED_SIZE);

	setupHeatmap<<<1,SIZE>>>(hm,heatmap);
	cudaDeviceSynchronize();

	setupScaledHeatmap<<<CELLSIZE,SIZE>>>(shm,scaled_heatmap);
	cudaDeviceSynchronize();

	for (int i = 0; i < SCALED_SIZE; i++)
	{
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;

}

/* ---------------------------
	UPPDATE HEATMAP FUNCTIONS
  --------------------------*/ 

__global__ void kernel_fade(int **heatmap)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;	
	// fade one row at a time
	for (int row = 0; row < SIZE; row++) {
      heatmap[row][i] = (int)round(heatmap[row][tid] * 0.80);
    }
}

__global__ void kernel_agents(int *dev_heatmap, int size_agents, int *desiredX, int *desiredY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size_agents){
		int x = (int) desiredX[i];
		int y = (int) desiredY[i];
		atomicAdd(&dev_heatmap[y*SIZE + x], 40);

			// intensify heat for better color results
			//&dev_heatmap[y][x] += 40;
	}
}

__global__ void kernel_clip(int *dev_heatmap, int size_agents, int *desiredX, int *desiredY)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size_agents) {
		int x = (int) desiredX[tid];
		int y = (int) desiredY[tid];

		atomicMin(&dev_heatmap[y*SIZE + x], 255);
	}
}


__global__ void kernel_scale(int *dev_heatmap, int *dev_scaled_heatmap)
{
	int ytid = blockIdx.y * blockDim.y + threadIdx.y;	
	int xtid = blockIdx.x * blockDim.x + threadIdx.x;	
	int value = dev_heatmap[ytid*SIZE + xtid];
	for (int cellY = 0; cellY < CELLSIZE; cellY++)
	{
		for (int cellX = 0; cellX < CELLSIZE; cellX++)
		{
			dev_scaled_heatmap[(ytid*CELLSIZE+cellY) * SIZE * CELLSIZE + xtid*CELLSIZE*cellX] = value;

		}

	}
}

__global__ void kernel_blur(int *dev_heatmap, int *dev_blurred_heatmap, int *dev_scaled_heatmap)
{
	const int w[5][5] = 
	{{1, 4, 7, 4, 1},
	{4, 16, 26, 16, 4},
	{7, 26, 41, 26, 7},
	{4, 16, 26, 16, 4},
	{1, 4, 7, 4, 1}};


	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if (x >= 2 && x < SCALED_SIZE && y >= 2 && y < SCALED_SIZE)
	{
		int sum = 0;
		for (int k = -2; k < 3; k++)
		{
			for (int l = -2; l < 3; l++)
			{
				sum += w[2 + k][2 + l] * dev_scaled_heatmap[(y + k) * SCALED_SIZE + x + l];
			}
		}
		int value = sum / WEIGHTSUM;
		#if __CUDA_ARCH__ >= 200
			printf("SCALING ERROR: %i", value);
		#endif
		dev_blurred_heatmap[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
	}
}


void Ped::Model::updateHeatmapCuda() 
{

	cudaMemcpy(d_desiredX, desiredX, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_desiredY, desiredY, agents.size() * sizeof(int), cudaMemcpyHostToDevice);
	//cout << d_desiredX[0]	<< "\n";

	dim3 threads_per_block(32, 32);
    dim3 num_blocks(SIZE / threads_per_block.x, SIZE / threads_per_block.y);
	// Fade heatmap
	kernel_fade<<<num_blocks, threads_per_block>>>(d_heatmap);

    int threads_per_blocki = 1024;
    int num_blocksi = (agents.size() + threads_per_blocki - 1) / threads_per_blocki;
	kernel_agents<<<num_blocksi, threads_per_blocki>>>(d_heatmap, agents.size(), d_desiredX, d_desiredY);


	//Clip heatmap
	kernel_clip<<<num_blocks, threads_per_block>>>(d_heatmap, agents.size(), d_desiredX, d_desiredY);

	//Scale heatmap
	kernel_scale<<<num_blocks, threads_per_block>>>(d_heatmap, d_scaled_heatmap);

	// Blur heatmap
	dim3 num_blocks_SCALED(SCALED_SIZE / threads_per_block.x, SCALED_SIZE / threads_per_block.y);
	kernel_blur<<<num_blocks_SCALED,threads_per_block >>>(d_heatmap, d_blurred_heatmap, d_scaled_heatmap);
	//cout << d_blurred_heatmap[0] << "\n";

	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}

