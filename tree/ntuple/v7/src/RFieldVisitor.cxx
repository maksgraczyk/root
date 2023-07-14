/// \file RFieldVisitor.cxx
/// \ingroup NTuple ROOT7
/// \author Simon Leisibach <simon.leisibach@gmail.com>
/// \date 2019-06-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


//----------------------------- RPrepareVisitor --------------------------------


void ROOT::Experimental::RPrepareVisitor::VisitField(const Detail::RFieldBase &field)
{
   auto subFields = field.GetSubFields();
   for (auto f : subFields) {
      RPrepareVisitor visitor;
      f->AcceptVisitor(visitor);
      fNumFields += visitor.fNumFields;
      fDeepestLevel = std::max(fDeepestLevel, 1 + visitor.fDeepestLevel);
   }
}


void ROOT::Experimental::RPrepareVisitor::VisitFieldZero(const RFieldZero &field)
{
   VisitField(field);
   fNumFields--;
   fDeepestLevel--;
}


//---------------------------- RPrintSchemaVisitor -----------------------------


void ROOT::Experimental::RPrintSchemaVisitor::SetDeepestLevel(int d)
{
   fDeepestLevel = d;
}

void ROOT::Experimental::RPrintSchemaVisitor::SetNumFields(int n)
{
   fNumFields = n;
   SetAvailableSpaceForStrings();
}

void ROOT::Experimental::RPrintSchemaVisitor::VisitField(const Detail::RFieldBase &field)
{
   fOutput << fFrameSymbol << ' ';

   std::string key = fTreePrefix;
   key += "Field " + fFieldNoPrefix + std::to_string(fFieldNo);
   fOutput << RNTupleFormatter::FitString(key, fAvailableSpaceKeyString);
   fOutput << " : ";

   std::string value = field.GetName();
   if (!field.GetType().empty())
      value += " (" + field.GetType() + ")";
   fOutput << RNTupleFormatter::FitString(value, fAvailableSpaceValueString);
   fOutput << fFrameSymbol << std::endl;

   auto subFields = field.GetSubFields();
   auto fieldNo = 1;
   for (auto iField = subFields.begin(); iField != subFields.end(); ) {
      RPrintSchemaVisitor visitor(*this);
      visitor.fFieldNo = fieldNo++;
      visitor.fFieldNoPrefix += std::to_string(fFieldNo) + ".";

      auto f = *iField;
      ++iField;
      // TODO(jblomer): implement tree drawing
      visitor.fTreePrefix += "  ";
      f->AcceptVisitor(visitor);
   }
}


void ROOT::Experimental::RPrintSchemaVisitor::VisitFieldZero(const RFieldZero &fieldZero)
{
   auto fieldNo = 1;
   for (auto f : fieldZero.GetSubFields()) {
      RPrintSchemaVisitor visitor(*this);
      visitor.fFieldNo = fieldNo++;
      f->AcceptVisitor(visitor);
   }
}


//--------------------------- RPrintValueVisitor -------------------------------

void ROOT::Experimental::RPrintValueVisitor::PrintIndent()
{
   if (fPrintOptions.fPrintSingleLine)
      return;

   for (unsigned int i = 0; i < fLevel; ++i)
      fOutput << "  ";
}


void ROOT::Experimental::RPrintValueVisitor::PrintName(const Detail::RFieldBase &field)
{
   if (fPrintOptions.fPrintName)
      fOutput << "\"" << field.GetName() << "\": ";
}


void ROOT::Experimental::RPrintValueVisitor::PrintCollection(const Detail::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "[";
   auto elems = field.SplitValue(fValue);
   for (std::size_t i = 0; i < elems.size(); ++i) {
      RPrintOptions options;
      options.fPrintSingleLine = true;
      options.fPrintName = false;
      auto subField = elems[i].GetField();
      RPrintValueVisitor elemVisitor(std::move(elems[i]), fOutput, 0 /* level */, options);
      subField->AcceptVisitor(elemVisitor);

      if (i == (elems.size() - 1))
         break;
      else
         fOutput << ", ";
   }
   fOutput << "]";
}


void ROOT::Experimental::RPrintValueVisitor::VisitField(const Detail::RFieldBase &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "\"<unsupported type: " << field.GetType() << ">\"";
}


void ROOT::Experimental::RPrintValueVisitor::VisitBoolField(const RField<bool> &field)
{
   PrintIndent();
   PrintName(field);
   if (*fValue.Get<bool>())
      fOutput << "true";
   else
      fOutput << "false";
}


void ROOT::Experimental::RPrintValueVisitor::VisitDoubleField(const RField<double> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<double>();
}


void ROOT::Experimental::RPrintValueVisitor::VisitFloatField(const RField<float> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<float>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitCharField(const RField<char> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<char>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt8Field(const RField<std::int8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::int8_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt16Field(const RField<std::int16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::int16_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitIntField(const RField<int> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<int>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitInt64Field(const RField<std::int64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::int64_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitStringField(const RField<std::string> &field)
{
   PrintIndent();
   PrintName(field);
   // TODO(jblomer): escape double quotes
   fOutput << "\"" << *fValue.Get<std::string>() << "\"";
}


void ROOT::Experimental::RPrintValueVisitor::VisitUInt8Field(const RField<std::uint8_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << static_cast<int>(*fValue.Get<std::uint8_t>());
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt16Field(const RField<std::uint16_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::uint16_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitUInt32Field(const RField<std::uint32_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::uint32_t>();
}


void ROOT::Experimental::RPrintValueVisitor::VisitUInt64Field(const RField<std::uint64_t> &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << *fValue.Get<std::uint64_t>();
}

void ROOT::Experimental::RPrintValueVisitor::VisitCardinalityField(const RCardinalityField &field)
{
   PrintIndent();
   PrintName(field);
   if (field.As32Bit()) {
      fOutput << *fValue.Get<std::uint32_t>();
      return;
   }
   if (field.As64Bit()) {
      fOutput << *fValue.Get<std::uint64_t>();
      return;
   }
   R__ASSERT(false && "unsupported cardinality size type");
}

void ROOT::Experimental::RPrintValueVisitor::VisitBitsetField(const RBitsetField &field)
{
   constexpr auto nBitsULong = sizeof(unsigned long) * 8;
   const auto *asULongArray = fValue.Get<unsigned long>();

   PrintIndent();
   PrintName(field);
   fOutput << "\"";
   std::size_t i = 0;
   std::string str;
   for (std::size_t word = 0; word < (field.GetN() + nBitsULong - 1) / nBitsULong; ++word) {
      for (std::size_t mask = 0; (mask < nBitsULong) && (i < field.GetN()); ++mask, ++i) {
         bool isSet = (asULongArray[word] & (static_cast<unsigned long>(1) << mask)) != 0;
         str = std::to_string(isSet) + str;
      }
   }
   fOutput << str << "\"";
}

void ROOT::Experimental::RPrintValueVisitor::VisitArrayField(const RArrayField &field)
{
   PrintCollection(field);
}


void ROOT::Experimental::RPrintValueVisitor::VisitClassField(const RClassField &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "{";
   auto elems = field.SplitValue(fValue);
   for (std::size_t i = 0; i < elems.size(); ++i) {
      if (!fPrintOptions.fPrintSingleLine)
         fOutput << std::endl;

      RPrintOptions options;
      options.fPrintSingleLine = fPrintOptions.fPrintSingleLine;
      auto subField = elems[i].GetField();
      RPrintValueVisitor visitor(std::move(elems[i]), fOutput, fLevel + 1, options);
      subField->AcceptVisitor(visitor);

      if (i == (elems.size() - 1)) {
         if (!fPrintOptions.fPrintSingleLine)
            fOutput << std::endl;
         break;
      } else {
         fOutput << ",";
         if (fPrintOptions.fPrintSingleLine)
           fOutput << " ";
      }
   }
   PrintIndent();
   fOutput << "}";
}


void ROOT::Experimental::RPrintValueVisitor::VisitRecordField(const RRecordField &field)
{
   PrintIndent();
   PrintName(field);
   fOutput << "{";
   auto elems = field.SplitValue(fValue);
   for (std::size_t i = 0; i < elems.size(); ++i) {
      if (!fPrintOptions.fPrintSingleLine)
         fOutput << std::endl;

      RPrintOptions options;
      options.fPrintSingleLine = fPrintOptions.fPrintSingleLine;
      auto subField = elems[i].GetField();
      RPrintValueVisitor visitor(std::move(elems[i]), fOutput, fLevel + 1, options);
      subField->AcceptVisitor(visitor);

      if (i == (elems.size() - 1)) {
         if (!fPrintOptions.fPrintSingleLine)
            fOutput << std::endl;
         break;
      } else {
         fOutput << ",";
         if (fPrintOptions.fPrintSingleLine)
           fOutput << " ";
      }
   }
   PrintIndent();
   fOutput << "}";
}

void ROOT::Experimental::RPrintValueVisitor::VisitNullableField(const RNullableField &field)
{
   PrintIndent();
   PrintName(field);
   auto elems = field.SplitValue(fValue);
   if (elems.empty()) {
      fOutput << "null";
   } else {
      RPrintOptions options;
      options.fPrintSingleLine = true;
      options.fPrintName = false;
      auto subField = elems[0].GetField();
      RPrintValueVisitor visitor(std::move(elems[0]), fOutput, fLevel, options);
      subField->AcceptVisitor(visitor);
   }
}

void ROOT::Experimental::RPrintValueVisitor::VisitEnumField(const REnumField &field)
{
   PrintIndent();
   PrintName(field);
   auto intValue = std::move(field.SplitValue(fValue)[0]);
   RPrintOptions options;
   options.fPrintSingleLine = true;
   options.fPrintName = false;
   auto subfield = intValue.GetField();
   RPrintValueVisitor visitor(std::move(intValue), fOutput, fLevel, options);
   subfield->AcceptVisitor(visitor);
}

void ROOT::Experimental::RPrintValueVisitor::VisitCollectionClassField(const RCollectionClassField &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitVectorField(const RVectorField &field)
{
   PrintCollection(field);
}


void ROOT::Experimental::RPrintValueVisitor::VisitVectorBoolField(const RField<std::vector<bool>> &field)
{
   PrintCollection(field);
}

void ROOT::Experimental::RPrintValueVisitor::VisitRVecField(const RRVecField &field)
{
   PrintCollection(field);
}

//---------------------------- RNTupleFormatter --------------------------------


std::string ROOT::Experimental::RNTupleFormatter::FitString(const std::string &str, int availableSpace)
{
   int strSize{static_cast<int>(str.size())};
   if (strSize <= availableSpace)
      return str + std::string(availableSpace - strSize, ' ');
   else if (availableSpace < 3)
      return std::string(availableSpace, '.');
   return std::string(str, 0, availableSpace - 3) + "...";
}
