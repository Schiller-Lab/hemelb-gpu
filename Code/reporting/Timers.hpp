// 
// Copyright (C) University College London, 2007-2012, all rights reserved.
// 
// This file is part of HemeLB and is CONFIDENTIAL. You may not work 
// with, install, use, duplicate, modify, redistribute or share this
// file, or any part thereof, other than as allowed by any agreement
// specifically made by you with University College London.
// 

#ifndef HEMELB_REPORTING_TIMERS_HPP
#define HEMELB_REPORTING_TIMERS_HPP

#include "reporting/Timers.h"

namespace hemelb
{
  namespace reporting
  {
    template<class ClockPolicy, class CommsPolicy> void TimersBase<ClockPolicy, CommsPolicy>::Reduce()
    {
      double timings[numberOfTimers];
      for (unsigned int ii = 0; ii < numberOfTimers; ii++)
      {
        timings[ii] = timers[ii].Get();
      }

      // TODO: Should this be World() or NetworkTopology::GetComms()?
      const net::MpiCommunicator& comms = net::MpiCommunicator::World();
      CommsPolicy::Reduce(timings,
                          &maxes[0],
                          numberOfTimers,
                          net::MpiDataType<double>(),
                          MPI_MAX,
                          0,
                          comms);
      CommsPolicy::Reduce(timings,
                          &means[0],
                          numberOfTimers,
                          net::MpiDataType<double>(),
                          MPI_SUM,
                          0,
                          comms);
      CommsPolicy::Reduce(timings, &mins[0], numberOfTimers, net::MpiDataType<double>(), MPI_MIN, 0, comms);
      for (unsigned int ii = 0; ii < numberOfTimers; ii++)
      {
        means[ii] /= (double) (net::IOCommunicator::Instance()->Size());
      }
    }

    template<class ClockPolicy, class CommsPolicy> void TimersBase<ClockPolicy, CommsPolicy>::Report(ctemplate::TemplateDictionary& dictionary)
    {
      dictionary.SetIntValue("THREADS", CommsPolicy::GetProcessorCount());

      for (unsigned int ii = 0; ii < numberOfTimers; ii++)
      {
        ctemplate::TemplateDictionary *timer = dictionary.AddSectionDictionary("TIMER");
        timer->SetValue("NAME", timerNames[ii]);
        timer->SetFormattedValue("LOCAL", "%.3g", timers[ii].Get());
        timer->SetFormattedValue("MIN", "%.3g", Mins()[ii]);
        timer->SetFormattedValue("MEAN", "%.3g", Means()[ii]);
        timer->SetFormattedValue("MAX", "%.3g", Maxes()[ii]);
      }
    }

  }
}

#endif // HEMELB_REPORTING_TIMERS_HPP
