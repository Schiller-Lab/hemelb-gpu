#ifndef HEMELB_LB_STREAMERS_SIMPLEBOUNCEBACK_H
#define HEMELB_LB_STREAMERS_SIMPLEBOUNCEBACK_H

#include "lb/kernels/BaseKernel.h"
#include "lb/streamers/BaseStreamer.h"

namespace hemelb
{
  namespace lb
  {
    namespace streamers
    {
      template<typename CollisionImpl>
      class SimpleBounceBack : public BaseStreamer<SimpleBounceBack<CollisionImpl> >
      {
        public:
          typedef CollisionImpl CollisionType;

        private:
          CollisionType collider;

        public:
          SimpleBounceBack(kernels::InitParams& initParams) :
              collider(initParams)
          {

          }

          template<bool tDoRayTracing>
          inline void DoStreamAndCollide(const site_t firstIndex,
                                         const site_t siteCount,
                                         const LbmParameters* lbmParams,
                                         geometry::LatticeData* latDat,
                                         lb::MacroscopicPropertyCache& propertyCache)
          {
            for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
            {
              const geometry::Site site = latDat->GetSite(siteIndex);

              distribn_t *fOld = site.GetFOld();

              kernels::HydroVars<typename CollisionType::CKernel> hydroVars(fOld);

              collider.CalculatePreCollision(hydroVars, site);

              collider.Collide(lbmParams, hydroVars);

              for (unsigned int ii = 0; ii < CollisionType::CKernel::LatticeType::NUMVECTORS; ii++)
              {
                // The actual bounce-back lines, including streaming and collision. Basically swap
                // the non-equilibrium components of f in each of the opposing pairs of directions.
                site_t streamingDestination =
                    site.HasBoundary(ii) ?
                      (siteIndex * CollisionType::CKernel::LatticeType::NUMVECTORS)
                          + CollisionType::CKernel::LatticeType::INVERSEDIRECTIONS[ii] :
                      site.GetStreamedIndex(ii);

                // Remember, oFNeq currently hold the equilibrium distribution. We
                // simultaneously use this and correct it, here.
                * (latDat->GetFNew(streamingDestination)) = hydroVars.GetFPostCollision()[ii];
              }

              //TODO: Necessary to specify sub-class?
              BaseStreamer<SimpleBounceBack>::template UpdateMinsAndMaxes<tDoRayTracing>(hydroVars.v_x,
                                                                                         hydroVars.v_y,
                                                                                         hydroVars.v_z,
                                                                                         site,
                                                                                         hydroVars.GetFNeq().f,
                                                                                         hydroVars.density,
                                                                                         lbmParams,
                                                                                         propertyCache);
            }
          }

          template<bool tDoRayTracing>
          inline void DoPostStep(const site_t iFirstIndex,
                                 const site_t iSiteCount,
                                 const LbmParameters* iLbmParams,
                                 geometry::LatticeData* bLatDat,
                                 lb::MacroscopicPropertyCache& propertyCache)
          {

          }

          inline void DoReset(kernels::InitParams* init)
          {
            collider.Reset(init);
          }

      };
    }
  }
}

#endif /* HEMELB_LB_STREAMERS_SIMPLEBOUNCEBACK_H */
