
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_BUILDSYSTEMINTERFACE_H
#define HEMELB_LB_BUILDSYSTEMINTERFACE_H

#include "lb/kernels/Kernels.h"
#include "lb/kernels/momentBasis/MomentBases.h"
#include "lb/kernels/rheologyModels/RheologyModels.h"
#include "lb/collisions/Collisions.h"
#include "lb/streamers/Streamers.h"

namespace hemelb
{
  namespace lb
  {
    /**
     * The names of the following classes must correspond to options given for the CMake
     * HEMELB_KERNEL parameter.
     */
    template<class Lattice>
    class LBGK
    {
      public:
        typedef kernels::LBGK<Lattice> Type;
    };

    /**
     * The entropic implementation by Ansumali et al.
     */
    template<class Lattice>
    class EntropicAnsumali
    {
      public:
        typedef kernels::EntropicAnsumali<Lattice> Type;
    };

    /**
     * The entropic implementation by Chikatamarla et al.
     */
    template<class Lattice>
    class EntropicChik
    {
      public:
        typedef kernels::EntropicChik<Lattice> Type;
    };

    /**
     * MRT currently we only have DHumieres implementation, on D3Q15 and D3Q19 lattices.
     */
    template<class Lattice>
    class MRT
    {
    };

    template<>
    class MRT<lattices::D3Q15>
    {
      public:
        typedef kernels::MRT<kernels::momentBasis::DHumieresD3Q15MRTBasis> Type;
    };

    template<>
    class MRT<lattices::D3Q19>
    {
      public:
        typedef kernels::MRT<kernels::momentBasis::DHumieresD3Q19MRTBasis> Type;
    };

    template<class Lattice>
    class TRT
    {
      public:
        typedef kernels::TRT<Lattice> Type;
    };

    /**
     * Non-Newtonian kernel with Carreau-Yasuda rheology model.
     */
    template<class Lattice>
    class NNCY
    {
      public:
        typedef kernels::LBGKNN<kernels::rheologyModels::CarreauYasudaRheologyModelHumanFit, Lattice> Type;
    };

    /**
     * Non-Newtonian kernel with Carreau-Yasuda rheology model fitted to experimental data on murine blood viscosity.
     */
    template<class Lattice>
    class NNCYMOUSE
    {
      public:
        typedef kernels::LBGKNN<kernels::rheologyModels::CarreauYasudaRheologyModelMouseFit, Lattice> Type;
    };

    /**
     * Non-Newtonian kernel with Casson rheology model.
     */
    template<class Lattice>
    class NNC
    {
      public:
        typedef kernels::LBGKNN<kernels::rheologyModels::CassonRheologyModel, Lattice> Type;
    };

    /**
     * Non-Newtonian kernel with truncated power law rheology model.
     */
    template<class Lattice>
    class NNTPL
    {
      public:
        typedef kernels::LBGKNN<kernels::rheologyModels::TruncatedPowerLawRheologyModel, Lattice> Type;
    };

    /**
     * The following classes have names corresponding to the options given in the build system for
     * HEMELB_WALL_INLET_BOUNDARY / HEMELB_WALL_OUTLET_BOUNDARY
     */
    /**
     * Nash in/outlet + SBB
     */
    template<class Collision>
    class NASHZEROTHORDERPRESSURESBB
    {
      public:
        typedef typename streamers::NashZerothOrderPressureIoletSBB<Collision>::Type Type;
    };

    /**
     * Ladd in/outlet + SBB
     */
    template<class Collision>
    struct LADDIOLETSBB
    {
        typedef typename streamers::LaddIoletSBB<Collision>::Type Type;
    };

    /**
     * Nash in/outlet + BFL
     */
    template<class Collision>
    class NASHZEROTHORDERPRESSUREBFL
    {
      public:
        typedef typename streamers::NashZerothOrderPressureIoletBFL<Collision>::Type Type;
    };

    /**
     * Ladd in/outlet + BFL
     */
    template<class Collision>
    struct LADDIOLETBFL
    {
        typedef typename streamers::LaddIoletBFL<Collision>::Type Type;
    };
    /**
     * Nash in/outlet + GZS
     */
    template<class Collision>
    class NASHZEROTHORDERPRESSUREGZS
    {
      public:
        typedef typename streamers::NashZerothOrderPressureIoletGZS<Collision>::Type Type;
    };

    /**
     * Ladd in/outlet + GZS
     */
    template<class Collision>
    struct LADDIOLETGZS
    {
        typedef typename streamers::LaddIoletGZS<Collision>::Type Type;
    };

    /**
     * Nash/Krueger in/outlet + SBB
     */
    template<class Collision>
    struct VIRTUALSITEIOLETSBB
    {
        typedef typename streamers::VirtualSiteIolet<Collision> Type;
    };
  }
}

#endif /* HEMELB_LB_BUILDSYSTEMINTERFACE_H */
