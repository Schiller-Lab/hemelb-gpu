
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_KERNELS_ENTROPICANSUMALI_H
#define HEMELB_LB_KERNELS_ENTROPICANSUMALI_H

#include <cstdlib>

#include "lb/kernels/Entropic.h"
#include "lb/kernels/BaseKernel.h"
#include "lb/HFunction.h"
#include "util/UtilityFunctions.h"

namespace hemelb
{
  namespace lb
  {
    namespace kernels
    {
      // We have to declare this up here in order for it to be used as a template parameter in the
      // following declaration. Moving the template specialisation to the bottom of the file would
      // prevent it from being used as the HydroVars for this kernel.
      template<class LatticeType> class EntropicAnsumali;

      template<class LatticeType>
      struct HydroVars<EntropicAnsumali<LatticeType> > : public HydroVarsBase<LatticeType>
      {
        public:
          HydroVars(const distribn_t* const f) :
              HydroVarsBase<LatticeType>(f)
          {

          }

          site_t index;
      };

      /**
       * Entropic: This class implements the entropic kernel as according to Anusmali et al, a modification to the standard
       * LBGK kernel which ensures the increase of entropy.
       */
      template<class LatticeType>
      class EntropicAnsumali : public BaseKernel<EntropicAnsumali<LatticeType>, LatticeType>
                               , public Entropic<LatticeType>
      {
        public:
          /**
           * Constructor, passes parameters onto the base class.
           * @param initParams
           */
          EntropicAnsumali(InitParams& initParams) :
              Entropic<LatticeType>(&initParams)
          {
          }

          /**
           * Calculates the density and momentum for the given f. Then calculates the
           * equilibrium distribution as described by Ansumali.
           * @param hydroVars
           * @param index The current lattice site index.
           */
          inline void DoCalculateDensityMomentumFeq(HydroVars<EntropicAnsumali<LatticeType> >& hydroVars, site_t index)
          {
            hydroVars.index = index;
            LatticeType::CalculateDensityAndMomentum(hydroVars.f,
                                                     hydroVars.density,
                                                     hydroVars.momentum.x,
                                                     hydroVars.momentum.y,
                                                     hydroVars.momentum.z);
            LatticeType::CalculateEntropicFeqAnsumali(hydroVars.density,
                                                      hydroVars.momentum.x,
                                                      hydroVars.momentum.y,
                                                      hydroVars.momentum.z,
                                                      hydroVars.f_eq.f);

            for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
            {
              hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
            }
          }

          /**
           * Calculates the equilibrium f distribution for the given density and momentum, as
           * described by Ansumali.
           * @param hydroVars
           * @param index The current lattice site index.
           */
          inline void DoCalculateFeq(HydroVars<EntropicAnsumali<LatticeType> >& hydroVars, site_t index)
          {
            hydroVars.index = index;
            LatticeType::CalculateEntropicFeqAnsumali(hydroVars.density,
                                                      hydroVars.momentum.x,
                                                      hydroVars.momentum.y,
                                                      hydroVars.momentum.z,
                                                      hydroVars.f_eq.f);

            for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
            {
              hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
            }
          }
      };
    }
  }
}

#endif /* HEMELB_LB_KERNELS_ENTROPICANSUMALI_H */
