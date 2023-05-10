
#include <stdlib.h>
#include "gtest/gtest.h"
#include "InvariantMassCUDA.h"

#include <Math/Vector4D.h>
// #include <Math/LorentzVector.h>
// #include <Math/PtEtaPhiE4D.h>
#include <ROOT/RVec.hxx>
#include <ROOT/TSeq.hxx>

using namespace ROOT::VecOps;
using namespace ROOT::Math;

// Element-wise comparison between two masses arrays.
#define CHECK_MASSES(a, b, n)                                     \
   {                                                              \
      for (auto i : ROOT::TSeqI(n)) {                             \
         EXPECT_NEAR(a[i], b[i], 1e-4) << "  at index i = " << i; \
      }                                                           \
   }

TEST(InvariantMassTest, PtEtaPhiEVectorComparison)
{
   // Dummy particle collections
   RVec<double> e1 = {50, 50, 50, 50, 100};
   RVec<double> pt1 = {0, 5, 5, 10, 10};
   RVec<double> eta1 = {0.0, 0.0, -1.0, 0.5, 2.5};
   RVec<double> phi1 = {0.0, 0.0, 0.0, -0.5, -2.4};

   RVec<double> e2 = {40, 40, 40, 40, 30};
   RVec<double> pt2 = {0, 5, 5, 10, 2};
   RVec<double> eta2 = {0.0, 0.0, 0.5, 0.4, 1.2};
   RVec<double> phi2 = {0.0, 0.0, 0.0, 0.5, 2.4};

   // Compute invariant mass of two particle system using both collections
   auto p1 = new PtEtaPhiEVector[e1.size()];
   auto p2 = new PtEtaPhiEVector[e1.size()];
   auto expectedInvMass = new double[e1.size()];
   for (size_t i = 0; i < e1.size(); i++) {
      p1[i].SetCoordinates(pt1[i], eta1[i], phi1[i], e1[i]);
      p2[i].SetCoordinates(pt2[i], eta2[i], phi2[i], e2[i]);
      expectedInvMass[i] = (p1[i] + p2[i]).mass();
   }

   const auto CUDAinvMasses =
      ROOT::Experimental::InvariantMassCUDA<PtEtaPhiE4D<double>>::ComputeInvariantMasses(p1, p2, e1.size());
   CHECK_MASSES(expectedInvMass, CUDAinvMasses, e1.size());
}
