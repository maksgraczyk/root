#ifndef INVARIANT_MASS_CUDA
#define INVARIANT_MASS_CUDA

#include "Math/Vector4D.h"
#include <Math/PtEtaPhiE4D.h>

namespace ROOT {
namespace Experimental {

template <class CoordSystem, unsigned int BlockSize = 256>
class InvariantMassCUDA {
public:
   static typename CoordSystem::Scalar *ComputeInvariantMasses(const ROOT::Math::LorentzVector<CoordSystem> *v1,
                                                               const ROOT::Math::LorentzVector<CoordSystem> *v2,
                                                               size_t size);
};

} // namespace Experimental
} // namespace ROOT
#endif
