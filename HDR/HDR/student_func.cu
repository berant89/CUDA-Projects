/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#define BLKW 32 //blockDim.x
#define BLKH 16 //blockDim.y
#define BLKR 32 //Block dimension for reduction.

__global__
void findMin(const float* const d_in, float* d_out, int size)
{
	extern __shared__ float sdata[]; //Shared data

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = 99999;
	for(;i < size; i += blockDim.x)
		sdata[tid] = min(sdata[tid], d_in[i]);
	__syncthreads();

	i = blockIdx.x*blockDim.x + tid;
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s && i < size)
			sdata[tid] = min(sdata[tid], sdata[tid + s]);
		__syncthreads();
	}

	if(tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

__device__
float atmoicMaxf(float* adr, float val)
{
	int* adr_as_int = (int*) adr;
	int old = *adr_as_int, assumed;
	while(val > __int_as_float(old))
	{
		assumed = old;
		old = atomicCAS(adr_as_int, assumed, __float_as_int(val));
	}

	return __int_as_float(old);
}

__global__
void findMax(const float* const d_in, float* d_out, int size)
{
	extern __shared__ float sdata[]; //Shared data

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	sdata[tid] = 0;
	for(;i < size; i += blockDim.x)
		sdata[tid] = max(sdata[tid], d_in[i]);
	__syncthreads();

	i = blockIdx.x*blockDim.x + tid;
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if(tid < s && i < size)
			sdata[tid] = max(sdata[tid], sdata[tid + s]);
		__syncthreads();
	}

	if(tid == 0)
		atmoicMaxf(d_out, sdata[0]);
		//d_out[blockIdx.x] = sdata[0];
}

__global__
void generate_histo(const float* d_logLuminance, unsigned int* d_histo , const float* d_min, const float* d_max, const int size, const int numBins)
{
	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x*blockDim.x + tid;

	if(tid < size)
	{
		float lumRange = *d_max - *d_min;
		unsigned int myBin = 0;

		myBin = min(static_cast<unsigned int>(numBins-1), static_cast<unsigned int>((d_logLuminance[id] - *d_min)/lumRange * numBins));
		atomicAdd(&(d_histo[myBin]), 1);
	}
}

inline __device__
unsigned int scanInclusive(unsigned int idata, volatile unsigned int* sdata, unsigned int numBins)
{
	unsigned int i = 2 * threadIdx.x - (threadIdx.x & (numBins - 1));
	sdata[i] = 0;
	i += numBins;
	sdata[i] = idata;

	for(unsigned int offset = 1; offset < numBins; offset <<=1)
	{
		__syncthreads();
		unsigned int t = sdata[i] + sdata[i - offset];
		__syncthreads();
		sdata[i] = t;
	}

	return sdata[i];
}

inline __device__
unsigned int scanExclusive(unsigned int idata, volatile unsigned int* sdata, unsigned int numBins)
{
	return scanInclusive(idata, sdata, numBins) - idata;
}

__global__
void calc_cdf(unsigned int* const d_cdf, unsigned int* d_histo, int numBins)
{
	__shared__ unsigned int sdata[BLKR * BLKR]; //Shared data

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + tid;

	unsigned int idata = 0;

	if(i < numBins)
		idata = 
		d_cdf[(BLKR * BLKR) - 1 + (BLKR * BLKR) * i] +
		d_histo[(BLKR * BLKR) - 1 + (BLKR * BLKR) * i];

	unsigned int odata = scanExclusive(idata, sdata, numBins);
}

__global__
void uniformUpdate(unsigned int* const d_cdf, unsigned int* d_buf)
{
	__shared__ unsigned int buf;
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(threadIdx.x == 0)
		buf = d_buf[blockIdx.x];

	__syncthreads();

	d_cdf[id] = buf;
}

void cudaInit(float **d_min, float **d_max, float **d_intermmediateA, float **d_intermmediateB, unsigned int** d_histo, int numBins, int size)
{
	int array_bytes = sizeof(float) * size;
	checkCudaErrors(cudaMalloc((void**) d_min, sizeof(float)));
	checkCudaErrors(cudaMemset(*d_min, 0, 1));

	checkCudaErrors(cudaMalloc((void**) d_max, sizeof(float)));
	checkCudaErrors(cudaMemset(*d_max, 0, 1));

	checkCudaErrors(cudaMalloc((void**) d_intermmediateA, array_bytes));
	checkCudaErrors(cudaMemset(*d_intermmediateA, 0, size));

	checkCudaErrors(cudaMalloc((void**) d_intermmediateB, array_bytes));
	checkCudaErrors(cudaMemset(*d_intermmediateB, 0, size));

	checkCudaErrors(cudaMalloc((void**) d_histo, sizeof(unsigned int)*numBins));
	checkCudaErrors(cudaMemset(*d_histo, 0, numBins));
}

void cudaCleanUp(float *d_min, float *d_max, float *d_intermmediateA, float *d_intermmediateB, unsigned int* d_histo)
{
	checkCudaErrors(cudaFree(d_min));
	checkCudaErrors(cudaFree(d_max));
	checkCudaErrors(cudaFree(d_intermmediateA));
	checkCudaErrors(cudaFree(d_intermmediateB));
	checkCudaErrors(cudaFree(d_histo));
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	float *d_min, *d_max, *d_intermmediateA, *d_intermmediateB;
	unsigned int* d_histo;
	int size = numCols*numRows;
	const int maxthreads = BLKR * BLKR;
	int threads = maxthreads;

	const int maxblocks = size/maxthreads;
	int blocks = maxblocks;

	const dim3 blockSize(BLKR, BLKR, 1);
	const dim3 gridSize((numCols + BLKR - 1)/BLKR, (numRows + BLKR - 1)/BLKR);

	cudaInit(&d_min, &d_max, &d_intermmediateA, &d_intermmediateB, &d_histo, numBins, size);

	//Find Min value in logLuminance.
	findMin<<<blocks, threads, sizeof(float)*threads>>>(d_logLuminance, d_intermmediateA, size);
	//Sync the device and check for errors before continuing.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//Find Max value in logLuminance.
	findMax<<<blocks, threads, sizeof(float)*threads>>>(d_logLuminance, d_max, size);
	//Sync the device and check for errors before continuing.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	threads = blocks; blocks = 1;

	findMin<<<blocks, threads, sizeof(float)*threads>>>(d_intermmediateA, d_min, size);
	//Sync the device and check for errors before continuing.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//findMax<<<blocks, threads, sizeof(float)*threads>>>(d_intermmediateB, d_max, size);
	//Sync the device and check for errors before continuing.
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	generate_histo<<<maxblocks, maxthreads>>>(d_logLuminance, d_histo, d_min, d_max, size, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	calc_cdf<<<maxblocks, maxthreads>>>(d_cdf, d_histo, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	cudaCleanUp(d_min, d_max, d_intermmediateA, d_intermmediateB, d_histo);
}
