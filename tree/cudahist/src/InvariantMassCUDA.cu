#include "InvariantMassCUDA.h"
#include "Math/Vector4D.h"

#include "TError.h"

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

using ROOT::Math::LorentzVector;
using ROOT::Math::PtEtaPhiE4D;

namespace ROOT {
namespace Experimental {

template <class T>
struct PtEtaPhiE4DCUDA {
   T fPt, fEta, fPhi, fE;
};

template <class T>
__global__ void
InvariantMassesKernel(const PtEtaPhiE4DCUDA<T> *v1, const PtEtaPhiE4DCUDA<T> *v2, size_t size, T *result)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;

   for (int i = tid; i < size; i += stride) {
      // Conversion from (pt, eta, phi, mass) to (x, y, z, e) coordinate system
      const auto x1 = v1[i].fPt * cos(v1[i].fPhi);
      const auto y1 = v1[i].fPt * sin(v1[i].fPhi);
      const auto z1 = v1[i].fPt * sinh(v1[i].fEta);
      const auto e1 = v1[1].fE;

      const auto x2 = v2[i].fPt * cos(v2[i].fPhi);
      const auto y2 = v2[i].fPt * sin(v2[i].fPhi);
      const auto z2 = v2[i].fPt * sinh(v2[i].fEta);
      const auto e2 = v2[i].fE;

      // Addition of particle four-vector elements
      const auto e = e1 + e2;
      const auto x = x1 + x2;
      const auto y = y1 + y2;
      const auto z = z1 + z2;

      auto mm = e * e - x * x - y * y - z * z;
      result[i] = sqrt(mm);
   }
}

template <class CoordSystem, unsigned int BlockSize>
typename CoordSystem::Scalar *
InvariantMassCUDA<CoordSystem, BlockSize>::ComputeInvariantMasses(const LorentzVector<CoordSystem> *v1, const LorentzVector<CoordSystem> *v2,
                                          size_t size)
{
   typedef typename CoordSystem::Scalar Scalar;

   const int numBlocks = ceil(size / float(BlockSize));
   printf("numblocks: %d\n", numBlocks);

   PtEtaPhiE4DCUDA<Scalar> *dV1 = NULL;
   ERRCHECK(cudaMalloc((void **)&dV1, size * sizeof(PtEtaPhiE4DCUDA<Scalar>)));

   PtEtaPhiE4DCUDA<Scalar> *dV2 = NULL;
   ERRCHECK(cudaMalloc((void **)&dV2, size * sizeof(PtEtaPhiE4DCUDA<Scalar>)));

   Scalar *dResult = NULL;
   ERRCHECK(cudaMalloc((void **)&dResult, size * sizeof(Scalar)));

   ERRCHECK(cudaMemcpy(dV1, v1, size * sizeof(PtEtaPhiE4DCUDA<Scalar>), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(dV2, v2, size * sizeof(PtEtaPhiE4DCUDA<Scalar>), cudaMemcpyHostToDevice));

   InvariantMassesKernel<<<numBlocks, BlockSize>>>(dV1, dV2, size, dResult);
   cudaDeviceSynchronize();
   ERRCHECK(cudaPeekAtLastError());

   Scalar *result = (Scalar *)malloc(size * sizeof(Scalar));
   ERRCHECK(cudaMemcpy(result, dResult, size * sizeof(Scalar), cudaMemcpyDeviceToHost));
   return result;
}

// Template instantations
template class InvariantMassCUDA<PtEtaPhiE4D<double>, 256>;

} // namespace Experimental
} // namespace ROOT