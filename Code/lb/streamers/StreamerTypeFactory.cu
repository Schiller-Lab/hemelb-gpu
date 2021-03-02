
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/collisions/Collisions.h"
#include "lb/iolets/InOutLetCosine.cuh"
#include "lb/kernels/Kernels.h"
#include "lb/lattices/Lattices.h"
#include "lb/streamers/Streamers.h"

#include "lb/lattices/D3Q15.cuh"
#include "lb/lattices/D3Q19.cuh"
#include "lb/lattices/D3Q27.cuh"



using namespace hemelb;
using namespace hemelb::lb;



#define DmQn lattices::GPU:: HEMELB_LATTICE



class Normal_LBGK_SBB_Nash
{
public:
  typedef lattices:: HEMELB_LATTICE LatticeType;
  typedef typename collisions::Normal<kernels::LBGK<LatticeType>> CollisionType;
  typedef typename streamers::SimpleBounceBackDelegate<CollisionType> WallLinkType;
  typedef typename streamers::NashZerothOrderPressureDelegate<CollisionType> IoletLinkType;

  typedef typename streamers::StreamerTypeFactory<
    CollisionType,
    WallLinkType,
    IoletLinkType
  > Type;
};



__global__
void Normal_LBGK_SBB_Nash_StreamAndCollide(
  site_t firstIndex,
  site_t siteCount,
  distribn_t lbmParams_tau,
  distribn_t lbmParams_omega,
  const iolets::InOutLetCosineGPU* inlets,
  const iolets::InOutLetCosineGPU* outlets,
  const site_t* streamingIndices,
  const geometry::SiteData* siteData,
  const distribn_t* fOld,
  distribn_t* fNew,
  site_t totalSiteCount,
  unsigned long timeStep
)
{
  site_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= siteCount )
  {
    return;
  }

  site_t siteIndex = firstIndex + i;
  auto& site = siteData[siteIndex];

  // initialize hydroVars
  distribn_t f_old_j;
  distribn_t f_new_j;
  distribn_t density = 0.0;
  double3 momentum = make_double3(0.0, 0.0, 0.0);

  for ( Direction j = 0; j < DmQn::NUMVECTORS; ++j )
  {
    // copy fOld[i, j] to local memory
    f_old_j = fOld[j * totalSiteCount + siteIndex];

    // Normal::DoCalculatePreCollision()
    // LBGK::DoCalculateDensityMomentumFeq()
    // Lattice::CalculateDensityAndMomentum()
    density += f_old_j;
    momentum.x += DmQn::CXD[j] * f_old_j;
    momentum.y += DmQn::CYD[j] * f_old_j;
    momentum.z += DmQn::CZD[j] * f_old_j;
  }

  // Lattice::CalculateFeq()
  const distribn_t density_1 = 1. / density;
  const distribn_t momentumMagnitudeSquared =
      momentum.x * momentum.x
      + momentum.y * momentum.y
      + momentum.z * momentum.z;

  for ( Direction j = 0; j < DmQn::NUMVECTORS; ++j )
  {
    if ( site.HasIolet(j) )
    {
      // NashZerothOrderPressureDelegate::StreamLink()
      // get iolet
      auto& iolet = (site.GetSiteType() == geometry::INLET_TYPE)
        ? inlets[site.GetIoletId()]
        : outlets[site.GetIoletId()];

      // get density at the iolet
      distribn_t ioletDensity = iolet.GetDensity(timeStep);

      // compute momentum at the iolet
      distribn_t component =
          momentum.x * iolet.normal.x
          + momentum.y * iolet.normal.y
          + momentum.z * iolet.normal.z;

      component *= ioletDensity / density;

      double3 ioletMomentum;
      ioletMomentum.x = iolet.normal.x * component;
      ioletMomentum.y = iolet.normal.y * component;
      ioletMomentum.z = iolet.normal.z * component;

      // compute f_eq at the iolet
      // Lattice::CalculateFeq()
      const distribn_t ioletDensity_1 = 1. / ioletDensity;
      const distribn_t momentumMagnitudeSquared =
          ioletMomentum.x * ioletMomentum.x
          + ioletMomentum.y * ioletMomentum.y
          + ioletMomentum.z * ioletMomentum.z;

      Direction jj = DmQn::INVERSEDIRECTIONS[j];
      const distribn_t mom_dot_ei =
          DmQn::CXD[jj] * ioletMomentum.x
          + DmQn::CYD[jj] * ioletMomentum.y
          + DmQn::CZD[jj] * ioletMomentum.z;

      f_new_j = DmQn::EQMWEIGHTS[jj]
          * (ioletDensity
              - (3. / 2.) * ioletDensity_1 * momentumMagnitudeSquared
              + (9. / 2.) * ioletDensity_1 * mom_dot_ei * mom_dot_ei
              + 3. * mom_dot_ei);
    }
    else
    {
      // copy fOld[i, j] to local memory
      f_old_j = fOld[j * totalSiteCount + siteIndex];

      // Lattice::CalculateFeq()
      const distribn_t mom_dot_ei =
          DmQn::CXD[j] * momentum.x
          + DmQn::CYD[j] * momentum.y
          + DmQn::CZD[j] * momentum.z;

      f_new_j = DmQn::EQMWEIGHTS[j]
          * (density
              - (3. / 2.) * density_1 * momentumMagnitudeSquared
              + (9. / 2.) * density_1 * mom_dot_ei * mom_dot_ei
              + 3. * mom_dot_ei);

      // Normal::DoCollide()
      // LBGK::DoCollide()
      f_new_j = f_old_j + lbmParams_omega * (f_old_j - f_new_j);
    }

    // stream f_new_j to pre-determined output location
    site_t outputIndex = streamingIndices[j * totalSiteCount + siteIndex];

    fNew[outputIndex] = f_new_j;
  }
}



template<>
void Normal_LBGK_SBB_Nash::Type::StreamAndCollideGPU(
  const site_t firstIndex,
  const site_t siteCount,
  const lb::LbmParameters* lbmParams,
  geometry::LatticeData* latDat,
  lb::SimulationState* simState,
  const iolets::InOutLetCosineGPU* inlets,
  const iolets::InOutLetCosineGPU* outlets,
  int blockSize
)
{
  if ( siteCount == 0 )
  {
    return;
  }

  int gridSize = (siteCount + blockSize - 1) / blockSize;

  Normal_LBGK_SBB_Nash_StreamAndCollide<<<gridSize, blockSize>>>(
    firstIndex,
    siteCount,
    lbmParams->GetTau(),
    lbmParams->GetOmega(),
    inlets,
    outlets,
    latDat->GetStreamingIndicesGPU(),
    latDat->GetSiteDataGPU(),
    latDat->GetFOldGPU(0),
    latDat->GetFNewGPU(0),
    latDat->GetLocalFluidSiteCount(),
    simState->Get0IndexedTimeStep()
  );
  CUDA_SAFE_CALL(cudaGetLastError());
}
