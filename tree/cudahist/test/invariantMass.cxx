
#include <stdlib.h>
#include "gtest/gtest.h"
#include "InvariantMassCUDA.h"

#include "TRandom3.h"
#include <Math/Vector4D.h>
#include <ROOT/RVec.hxx>
#include <ROOT/TSeq.hxx>

using namespace ROOT::VecOps;
using namespace ROOT::Math;

// Element-wise comparison between two masses arrays.
// NOTE: accuracy taken from root/math/vecops/test/vecops_rvec.cxx. Tests start to fail with an accuracy higher than
// 1e-6
#define CHECK_MASSES(a, b, n)                                     \
   {                                                              \
      for (auto i : ROOT::TSeqI(n)) {                             \
         EXPECT_NEAR(a[i], b[i], 1e-6) << "  at index i = " << i; \
      }                                                           \
   }

class InvariantMassTestFixture : public ::testing::Test {
protected:
   TRandom3 r;

   InvariantMassTestFixture()
   {
      r.SetSeed(123); // For reproducability
   }

   // Taken from root/test/stressVector.cxx
   RVec<PtEtaPhiEVector> GenRandomVectors(int n)
   {
      RVec<PtEtaPhiEVector> vectors(n);

      // generate n -4 momentum quantities
      for (int i = 0; i < n; ++i) {
         double phi = r.Rndm() * 3.1415926535897931;
         double eta = r.Uniform(-5., 5.);
         double pt = r.Exp(10.);
         double m = r.Uniform(0, 10.);
         if (i % 50 == 0)
            m = r.BreitWigner(1., 0.01);

         double E = sqrt(m * m + pt * pt * cosh(eta) * cosh(eta));

         // fill vectors
         vectors[i].SetCoordinates(pt, eta, phi, E);
      }

      return vectors;
   }
};

TEST_F(InvariantMassTestFixture, LorentzVectorComparison)
{
   const int numMasses = 10000;

   // Compute invariant mass of two particle system using both collections
   auto p1 = this->GenRandomVectors(numMasses);
   auto p2 = this->GenRandomVectors(numMasses);
   auto expectedInvMass = RVec<double>(numMasses);
   for (size_t i = 0; i < numMasses; i++) {
      expectedInvMass[i] = (p1[i] + p2[i]).mass();
   }

   const auto CUDAinvMasses =
      ROOT::Experimental::InvariantMassCUDA<256>::ComputeInvariantMasses(p1.begin(), p2.begin(), numMasses);
   CHECK_MASSES(expectedInvMass, CUDAinvMasses, numMasses);
}
