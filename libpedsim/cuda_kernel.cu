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


	int *desiredX = (int*)malloc(agents.size()*sizeof(int));
	int *desiredY = (int*)malloc(agents.size()*sizeof(int));

	for (int i = 0; i < agents.size(); i++)
	{
		Ped::Tagent* agent = agents[i];
		desiredX[i] = agent->getDesiredX();
		desiredY[i] = agent->getDesiredY();
	}




	cudaMalloc(&d_desiredX, agents.size()*sizeof(int));
	cudaMalloc(&d_desiredY, agents.size()*sizeof(int));
	// Allocate memory on GPU
	cudaMalloc(&d_heatmap, SIZE*sizeof(int*));
	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*sizeof(int*));
	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*sizeof(int*));

	// Copy memory from host to device
	cudaMemcpy(d_heatmap, heatmap, SIZE*sizeof(int*), cudaMemcpyHostToDevice);
	//cudaMemset(d_blurred_heatmap, 0, SCALED_SIZE*SCALED_SIZE*sizeof(int*));
	//cudaMemset(d_scaled_heatmap, 0, SCALED_SIZE*SCALED_SIZE*sizeof(int*));
	cudaMemcpy(d_scaled_heatmap, scaled_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap, SCALED_SIZE*sizeof(int*), cudaMemcpyHostToDevice);
}

/* ---------------------------
	UPPDATE HEATMAP FUNCTIONS
  --------------------------*/ 

__global__ void kernel_fade(int **dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x < SIZE && y < SIZE)
	{
		dev_heatmap[y][x] = (int)round(dev_heatmap[y][x] * 0.80);
	}
}

__global__ void kernel_agents(int **dev_heatmap, int size_agents, int *desiredX, int *desiredY)
{
	int xtid = blockIdx.x * blockDim.x + threadIdx.x;
	int ytid = blockIdx.y * blockDim.y + threadIdx.y;	
	if(xtid < size_agents && ytid < size_agents){
		int x = desiredX[xtid];
		int y = desiredY[ytid];

		if(x>=0 && x<SIZE && y>=0 && y<SIZE)
			// intensify heat for better color results
			//&dev_heatmap[y][x] += 40;
			atomicAdd(&dev_heatmap[ytid][xtid], 40);
	}
}

__global__ void kernel_clip(int **dev_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	if (x < SIZE && y < SIZE){
		dev_heatmap[y][x] = dev_heatmap[y][x] < 255 ? dev_heatmap[y][x] : 255;
	}
}

__global__ void kernel_scale(int **dev_heatmap, int **dev_scaled_heatmap)
{
	int xtid = blockIdx.x * blockDim.x + threadIdx.x;	
	int ytid = blockIdx.y * blockDim.y + threadIdx.y;	
	if (xtid < SIZE && ytid < SIZE)
	{
		int value = dev_heatmap[ytid][xtid];
		for (int cellY = 0; cellY < CELLSIZE; cellY++)
		{
			for (int cellX = 0; cellX < CELLSIZE; cellX++)
			{
				dev_scaled_heatmap[ytid * CELLSIZE + cellY][xtid * CELLSIZE + cellX] = value;

			}
		}
	}
}


    // __global__ void kernel_blur(int* d_scaled_heatmap,
    //                             int* d_blurred_heatmap) {
    //     /*
    //         Apply gaussian blur filter
    //         Parallize: parallelize the outer 2 for-loops
    //             dim3 filter_bSize(32, 32);
    //             dim3 filter_blocks(SCALED_SIZE / filter_bSize.x, SCALED_SIZE / filter_bSize.y);
    //     */
    //     __shared__ int shared_shm[32][32];

    //     int y = blockIdx.y * blockDim.y + threadIdx.y;
    //     int x = blockIdx.x * blockDim.x + threadIdx.x;
        
    //     //shared_shm[threadIdx.y][threadIdx.x] = d_scaled_heatmap[y * SCALED_SIZE + x];
        
    //     __syncthreads();

    //     // register w
    //     const int r_w[5][5] = {
    //         { 1, 4, 7, 4, 1 },
    //         { 4, 16, 26, 16, 4 },
    //         { 7, 26, 41, 26, 7 },
    //         { 4, 16, 26, 16, 4 },
    //         { 1, 4, 7, 4, 1 }
    //     };

    //     if(2 <= x && x < SCALED_SIZE - 2 && 2 <= y && y < SCALED_SIZE - 2) {
    //         int sum = 0 ;
    //         for (int k = -2; k < 3; k++) {
    //             for (int l = -2; l < 3; l++) {
    //                 // int shm_y = threadIdx.y + k;
    //                 // int shm_x = threadIdx.x + l;
    //                 int v;
	// 				v = d_scaled_heatmap[(y + k) * SCALED_SIZE + x + l];
    //                 // sum += d_w[(2 + k) * 5 + (2 + l)] * v;
    //                 sum += r_w[2 + k][2 + l] * v;
    //             }
    //         }
    //         int val = sum / 273;
    //         d_blurred_heatmap[y * SCALED_SIZE + x] = 0x00FF0000 | val << 24;
    //     }   
    // }

// __global__ void kernel_blur(int *dev_heatmap, int *dev_blurred_heatmap, int *dev_scaled_heatmap)
// {
// 	//weights for blur
// 	const int w[5][5] = {
// 		{ 1, 4, 7, 4, 1 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 7, 26, 41, 26, 7 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 1, 4, 7, 4, 1 }
// 	};
// 	int x = blockIdx.x * blockDim.x + threadIdx.x;	
// 	int y = blockIdx.y * blockDim.y + threadIdx.y;	
// 	if (x >= 2 && x < SCALED_SIZE && y >= 2 && y < SCALED_SIZE)
// 	{
// 		int sum = 0;
// 		for (int k = -2; k < 3; k++)
// 		{
// 			for (int l = -2; l < 3; l++)
// 			{
// 				sum += w[2 + k][2 + l] * dev_scaled_heatmap[y + k][x + l];
// 			}
// 		}
// 		int value = sum / WEIGHTSUM;
// 		dev_blurred_heatmap[y][x] = 0x00FF0000 | value << 24;
// 	}
// }


__global__ void kernel_blur(int **dev_blurred_heatmap, int **dev_scaled_heatmap) {
	// Weights for blur filter
	const int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	// Apply gaussian blurfilter		       
		if (x < SCALED_SIZE - 2 && y < SCALED_SIZE - 2 )
		{
			int sum = 0;
			for (int k = -2; k < 3; k++)
			{
				for (int l = -2; l < 3; l++)
				{
					sum += w[2 + k][2 + l] * dev_scaled_heatmap[y+k][x+l];
				}
			}
			int value = sum / WEIGHTSUM;
			dev_blurred_heatmap[y][x] = 0x00FF0000 | value << 24;
		}

}


void Ped::Model::updateHeatmapCuda() 
{
	// Create streams
	// cudaStream_t stream1, stream2;
	// cudaStreamCreate(&stream1);
	// cudaStreamCreate(&stream2);

	// cudaEvent_t ev1;
	// cudaEventCreate(&ev1);
	// Create events

	// Fade heatmap
	kernel_fade<<<1, SIZE>>>(d_heatmap);
	// cudaEventRecord(ev1, stream1);

	cudaMemcpy(d_desiredX, desiredX, agents.size()*sizeof(Ped::Tagent), cudaMemcpyHostToDevice);
	cudaMemcpy(d_desiredY, desiredY, agents.size()*sizeof(Ped::Tagent), cudaMemcpyHostToDevice);

	//cudaStreamWaitEvent(stream1, ev1);

	cudaDeviceSynchronize();

	kernel_agents<<<1, agents.size()>>>(d_heatmap, agents.size(), desiredX, desiredY);
	//Clip heatmap
	kernel_clip<<<1, SIZE>>>(d_heatmap);
	//Scale heatmap
	kernel_scale<<<1, SIZE>>>(d_heatmap, d_scaled_heatmap);

	// // Blur heatmap

	// dim3 filter_bSize(32, 32);
	// dim3 filter_blocks(SCALED_SIZE*SCALED_SIZE + yED_SIZE / filter_bSize.x, SCALED_SIZE / filter_bSize.y);
		
	kernel_blur<<<1,SIZE>>>(d_scaled_heatmap, d_blurred_heatmap);

	cudaMemcpy(blurred_heatmap[0], d_blurred_heatmap, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

	// cudaStreamDestroy(stream1);
	// cudaStreamDestroy(stream2);


}

int Ped::Model::getHeatmapSize() const {
	return SCALED_SIZE;
}

