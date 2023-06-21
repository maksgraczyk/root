// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
// Author: Giovanni Petrucciani (UCSD) (log-interpolation)
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::FlexibleInterpVar
 *  \ingroup HistFactory
 */

#include "Riostream.h"
#include <math.h>
#include "TMath.h"

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooTrace.h"

#include "RooFit/Detail/EvaluateFuncs.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"

using namespace std;

ClassImp(RooStats::HistFactory::FlexibleInterpVar);

using namespace RooStats;
using namespace HistFactory;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

FlexibleInterpVar::FlexibleInterpVar()
{
  _nominal = 0;
  _interpBoundary=1.;
  TRACE_CREATE
}


////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title,
             const RooArgList& paramList,
             double argNominal, std::vector<double> lowVec, std::vector<double> highVec) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(argNominal), _low(lowVec), _high(highVec), _interpBoundary(1.)
{
  for (auto param : paramList) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName()
             << " is not of type RooAbsReal" << endl ;
      R__ASSERT(0) ;
    }
    _paramList.add(*param) ;
    _interpCode.push_back(0); // default code
  }
  if (int(_low.size() ) != _paramList.getSize() || _low.size() != _high.size()) {
     coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") invalid input low/high vectors " << endl;
     R__ASSERT(int(_low.size() ) == _paramList.getSize());
     R__ASSERT(_low.size() == _high.size());
  }

  TRACE_CREATE
}

////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title,
             const RooArgList& paramList,
             double argNominal, const RooArgList& lowList, const RooArgList& highList) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(argNominal), _interpBoundary(1.)
{
  for (auto const *val : static_range_cast<RooAbsReal *>(lowList)){
    _low.push_back(val->getVal()) ;
  }

  for (auto const *val : static_range_cast<RooAbsReal *>(highList)) {
    _high.push_back(val->getVal()) ;
  }

  for (auto param : paramList) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName()
             << " is not of type RooAbsReal" << endl ;
      R__ASSERT(0) ;
    }
    _paramList.add(*param) ;
    _interpCode.push_back(0); // default code
  }
  if (int(_low.size() ) != _paramList.getSize() || _low.size() != _high.size()) {
     coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") invalid input low/high lists " << endl;
     R__ASSERT(int(_low.size() ) == _paramList.getSize());
     R__ASSERT(_low.size() == _high.size());
  }

  TRACE_CREATE
}




////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title,
                 const RooArgList& paramList,
                 double argNominal, vector<double> lowVec, vector<double> highVec,
                 vector<int> code) :
  RooAbsReal(name, title),
  _paramList("paramList","List of paramficients",this),
  _nominal(argNominal), _low(lowVec), _high(highVec), _interpCode(code), _interpBoundary(1.)
{
  for (auto param : paramList) {
    if (!dynamic_cast<RooAbsReal*>(param)) {
      coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") ERROR: paramficient " << param->GetName()
             << " is not of type RooAbsReal" << endl ;
      // use R__ASSERT which remains also in release mode
      R__ASSERT(0) ;
    }
    _paramList.add(*param) ;
  }

  if (int(_low.size() ) != _paramList.getSize() || _low.size() != _high.size() || _low.size() != _interpCode.size()) {
     coutE(InputArguments) << "FlexibleInterpVar::ctor(" << GetName() << ") invalid input vectors " << endl;
     R__ASSERT(int(_low.size() ) == _paramList.getSize());
     R__ASSERT(_low.size() == _high.size());
     R__ASSERT(_low.size() == _interpCode.size());
  }

  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor of flat polynomial function

FlexibleInterpVar::FlexibleInterpVar(const char* name, const char* title) :
  RooAbsReal(name, title),
  _paramList("paramList","List of coefficients",this),
  _nominal(0), _interpBoundary(1.)
{
  TRACE_CREATE;
}

////////////////////////////////////////////////////////////////////////////////

FlexibleInterpVar::FlexibleInterpVar(const FlexibleInterpVar& other, const char* name) :
  RooAbsReal(other, name),
  _paramList("paramList",this,other._paramList),
  _nominal(other._nominal), _low(other._low), _high(other._high), _interpCode(other._interpCode), _interpBoundary(other._interpBoundary)

{
  TRACE_CREATE;
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

FlexibleInterpVar::~FlexibleInterpVar()
{
  TRACE_DESTROY;
}


////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setInterpCode(RooAbsReal& param, int code){
  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setInterpCode ERROR:  " << param.GetName()
                            << " is not in list" << std::endl;
  } else if(_interpCode.at(index) != code){
      coutI(InputArguments) << "FlexibleInterpVar::setInterpCode :  " << param.GetName()
                            << " is now " << code << std::endl;
      _interpCode.at(index) = code;
      // GHL: Adding suggestion by Swagato:
      setValueDirty();
  }
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setAllInterpCodes(int code){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    _interpCode.at(i) = code;
  }
  // GHL: Adding suggestion by Swagato:
  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setNominal(double newNominal){
  coutW(InputArguments) << "FlexibleInterpVar::setNominal : nominal is now " << newNominal << endl ;
  _nominal = newNominal;

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setLow(RooAbsReal& param, double newLow){
  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setLow ERROR:  " << param.GetName()
             << " is not in list" << endl ;
  } else {
      coutW(InputArguments) << "FlexibleInterpVar::setLow :  " << param.GetName()
             << " is now " << newLow << endl ;
    _low.at(index) = newLow;
  }

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::setHigh(RooAbsReal& param, double newHigh){
  int index = _paramList.index(&param);
  if(index<0){
      coutE(InputArguments) << "FlexibleInterpVar::setHigh ERROR:  " << param.GetName()
             << " is not in list" << endl ;
  } else {
      coutW(InputArguments) << "FlexibleInterpVar::setHigh :  " << param.GetName()
             << " is now " << newHigh << endl ;
    _high.at(index) = newHigh;
  }

  setValueDirty();
}

////////////////////////////////////////////////////////////////////////////////

void FlexibleInterpVar::printAllInterpCodes(){
  for(unsigned int i=0; i<_interpCode.size(); ++i){
    coutI(InputArguments) <<"interp code for " << _paramList.at(i)->GetName() << " = " << _interpCode.at(i) <<endl;
    // GHL: Adding suggestion by Swagato:
    if( _low.at(i) <= 0.001 ) coutE(InputArguments) << GetName() << ", " << _paramList.at(i)->GetName() << ": low value = " << _low.at(i) << endl;
    if( _high.at(i) <= 0.001 ) coutE(InputArguments) << GetName() << ", " << _paramList.at(i)->GetName() << ": high value = " << _high.at(i) << endl;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Const getters

const RooListProxy& FlexibleInterpVar::variables() const { return _paramList; }
double FlexibleInterpVar::nominal() const { return _nominal; }
const std::vector<double>& FlexibleInterpVar::low() const { return _low; }
const std::vector<double>& FlexibleInterpVar::high() const { return _high; }

////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of polynomial

double FlexibleInterpVar::evaluate() const
{
   double total(_nominal);
   for (std::size_t i = 0; i < _paramList.size(); ++i) {
      if (_interpCode[i] < 0 || _interpCode[i] > 4) {
         coutE(InputArguments) << "FlexibleInterpVar::evaluate ERROR:  param " << i
                               << " with unknown interpolation code" << endl;
      }
      double paramVal = static_cast<const RooAbsReal *>(&_paramList[i])->getVal();
      total = RooFit::Detail::EvaluateFuncs::flexibleInterp(
         _interpCode[i], _low[i], _high[i], _interpBoundary, _nominal, paramVal, total);
   }

  if(total<=0) {
     total= TMath::Limits<double>::Min();
  }

  return total;
}

void FlexibleInterpVar::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   unsigned int n = _interpCode.size();

   std::string resName = "total_" + ctx.getTmpVarName();
   ctx.addToCodeBody(this, "double " + resName + " = " + std::to_string(_nominal) + ";\n");
   std::string code = "";
   for (std::size_t i = 0; i < n; ++i) {
      code += resName + " = " +
              ctx.buildCall("RooFit::Detail::EvaluateFuncs::flexibleInterp", _interpCode[i],
                            _low[i], _high[i], _interpBoundary,
                            _nominal, _paramList[i], resName) +
              ";\n";
   }
   code += resName + " = " + resName + " <= 0 ? TMath::Limits<double>::Min() : " + resName + ";\n";
   ctx.addToCodeBody(this, code);
   ctx.addResult(this, resName);
}

void FlexibleInterpVar::computeBatch(cudaStream_t* /*stream*/, double* output, size_t /*nEvents*/, RooFit::Detail::DataMap const& dataMap) const
{
  double total(_nominal) ;

  for (std::size_t i = 0; i < _paramList.size(); ++i) {
     if (_interpCode[i] < 0 || _interpCode[i] > 4) {
        coutE(InputArguments) << "FlexibleInterpVar::evaluate ERROR:  param " << i << " with unknown interpolation code"
                              << endl;
     }
     total = RooFit::Detail::EvaluateFuncs::flexibleInterp(_interpCode[i], _low[i],
                                                                          _high[i], _interpBoundary, _nominal,
                                                                          dataMap.at(&_paramList[i])[0], total);
  }

  if(total<=0) {
     total= TMath::Limits<double>::Min();
  }

  output[0] = total;
}

void FlexibleInterpVar::printMultiline(ostream& os, Int_t contents,
                   bool verbose, TString indent) const
{
  RooAbsReal::printMultiline(os,contents,verbose,indent);
  os << indent << "--- FlexibleInterpVar ---" << endl;
  printFlexibleInterpVars(os);
}

void FlexibleInterpVar::printFlexibleInterpVars(ostream& os) const
{
  for (int i=0;i<(int)_low.size();i++) {
    auto& param = static_cast<RooAbsReal&>(_paramList[i]);
    os << setw(36) << param.GetName()<<": "<<setw(7) << _low[i]<<"  "<<setw(7) << _high[i]
       <<endl;
  }
}
