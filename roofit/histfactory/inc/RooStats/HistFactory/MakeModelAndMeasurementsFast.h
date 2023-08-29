#ifndef HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H
#define HISTFACTORY_MAKEMODELANDMEASUREMENTSFAST_H

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"

#include "RooWorkspace.h"
#include "RooPlot.h"

#include <iostream>
#include <string>
#include <vector>

class TFile;

namespace RooStats{
  namespace HistFactory{

      RooFit::OwningPtr<RooWorkspace> MakeModelAndMeasurementFast(
            RooStats::HistFactory::Measurement& measurement,
            HistoToWorkspaceFactoryFast::Configuration const& cfg={}
    );

    void FormatFrameForLikelihood(RooPlot* frame, std::string xTitle=std::string("#sigma / #sigma_{SM}"), std::string yTitle=std::string("-log likelihood"));
    void FitModel(RooWorkspace *, std::string data_name="obsData");
    void FitModelAndPlot(const std::string& measurementName, const std::string& fileNamePrefix, RooWorkspace &, std::string, std::string, TFile&, std::ostream&);
  }
}


#endif
