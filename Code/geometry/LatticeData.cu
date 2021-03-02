
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "geometry/LatticeData.h"

#include "lb/lattices/D3Q15.cuh"
#include "lb/lattices/D3Q19.cuh"
#include "lb/lattices/D3Q27.cuh"


using namespace hemelb;
using namespace hemelb::geometry;
using namespace hemelb::lb;



#define DmQn lattices::GPU:: HEMELB_LATTICE



__global__
void TransposeStreamingIndicesKernel(
  site_t* streamingIndices,
  site_t* neighbourIndices,
  site_t nSites,
  site_t nDirections
)
{
  // get input index
  int inIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if ( inIndex >= nSites * nDirections )
  {
    return;
  }

  // inIndex = inDirection * nSites + inSite
  Direction inDirection = inIndex / nSites;
  site_t inSite = inIndex % nSites;

  // decode output index from AoS layout
  site_t outIndex = neighbourIndices[inSite * nDirections + inDirection];
  site_t outSite;
  site_t outDirection;

  // check for rubbish site
  if ( outIndex == nSites * nDirections )
  {
    outSite = 0;
    outDirection = nDirections;
  }
  else
  {
    outSite = outIndex / nDirections;
    outDirection = outIndex % nDirections;
  }

  // encode output index to SoA layout
  streamingIndices[inIndex] = outDirection * nSites + outSite;
}



__global__
void TransposeStreamingIndicesSharedKernel(
  site_t* streamingIndicesForReceivedDistributions,
  site_t totalSharedFs,
  site_t nSites,
  site_t nDirections
)
{
  // get input index
  int inIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if ( inIndex >= totalSharedFs )
  {
    return;
  }

  // decode output index from AoS layout
  site_t outIndex = streamingIndicesForReceivedDistributions[inIndex];
  site_t outSite = outIndex / nDirections;
  site_t outDirection = outIndex % nDirections;

  // encode output index to SoA layout
  streamingIndicesForReceivedDistributions[inIndex] = outDirection * nSites + outSite;
}



__global__
void PrepareStreamingIndicesKernel(
  site_t* streamingIndices,
  const geometry::SiteData* siteData,
  site_t nSites,
  site_t nDirections
)
{
  int inIndex = blockIdx.x * blockDim.x + threadIdx.x;

  if ( inIndex >= nSites * nDirections )
  {
    return;
  }

  Direction inDirection = inIndex / nSites;
  site_t inSite = inIndex % nSites;
  auto& site = siteData[inSite];

  // NashZerothOrderPressureDelegate::StreamLink()
  // SimpleBounceBackDelegate::StreamLink()
  // SimpleCollideAndStreamDelegate::StreamLink()
  if ( site.HasIolet(inDirection) || site.HasWall(inDirection) )
  {
    streamingIndices[inIndex] = DmQn::INVERSEDIRECTIONS[inDirection] * nSites + inSite;
  }
}



void LatticeData::PrepareStreamingIndicesGPU()
{
  site_t nSites = GetLocalFluidSiteCount();
  site_t nDirections = latticeInfo.GetNumVectors();
  int blockSize = 32;
  int gridSize = (nSites * nDirections + blockSize - 1) / blockSize;

  // transpose streaming indices to match SoA memory layout
  site_t* tempIndices_dev;
  CUDA_SAFE_CALL(cudaMalloc(&tempIndices_dev, latticeInfo.GetNumVectors() * localFluidSites * sizeof(site_t)));

  TransposeStreamingIndicesKernel<<<gridSize, blockSize>>>(
    tempIndices_dev,
    streamingIndices_dev,
    nSites,
    nDirections
  );
  CUDA_SAFE_CALL(cudaGetLastError());

  std::swap(tempIndices_dev, streamingIndices_dev);
  CUDA_SAFE_CALL(cudaFree(tempIndices_dev));

  // transpose shared streaming indices to match SoA memory layout
  TransposeStreamingIndicesSharedKernel<<<gridSize, blockSize>>>(
    streamingIndicesForReceivedDistributions_dev,
    totalSharedFs,
    nSites,
    nDirections
  );
  CUDA_SAFE_CALL(cudaGetLastError());

  // prepare streaming indices with boundary conditions
  PrepareStreamingIndicesKernel<<<gridSize, blockSize>>>(
    streamingIndices_dev,
    siteData_dev,
    nSites,
    nDirections
  );
  CUDA_SAFE_CALL(cudaGetLastError());
}



__global__
void CopyReceivedKernel(
  const site_t* streamingIndicesForReceivedDistributions,
  const distribn_t* fOldShared,
  distribn_t* fNew,
  site_t totalSharedFs
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= totalSharedFs )
  {
    return;
  }

  fNew[streamingIndicesForReceivedDistributions[i]] = fOldShared[i];
}



void LatticeData::CopyReceivedGPU(int blockSize)
{
  if ( totalSharedFs == 0 )
  {
    return;
  }

  int gridSize = (totalSharedFs + blockSize - 1) / blockSize;

  CopyReceivedKernel<<<gridSize, blockSize>>>(
    streamingIndicesForReceivedDistributions_dev,
    GetFOldGPU(neighbouringProcs[0].FirstSharedDistribution),
    GetFNewGPU(0),
    totalSharedFs
  );
  CUDA_SAFE_CALL(cudaGetLastError());
}
