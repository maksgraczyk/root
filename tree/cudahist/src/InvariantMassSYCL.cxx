#include <InvariantMassSYCL.h>
#include <CL/sycl.hpp>
#include <iostream>

#include "Math/Vector4D.h"

using ROOT::Math::PtEtaPhiEVector;
namespace sycl = cl::sycl;

auto exception_handler(sycl::exception_list exceptions)
{
   for (std::exception_ptr const &e_ptr : exceptions) {
      try {
         std::rethrow_exception(e_ptr);
      } catch (sycl::exception const &e) {
         std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
      }
   }
}

namespace ROOT {
namespace Experimental {

class invariant_masses;

double *InvariantMassSYCL(const PtEtaPhiEVector *v1, const PtEtaPhiEVector *v2, size_t size)
{
   double *invMasses = new double[size];

   sycl::gpu_selector device_selector;
   sycl::queue queue(device_selector, exception_handler);
   std::cout << "Running InvariantMassSYCL on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

   {
      sycl::buffer<PtEtaPhiEVector, 1> v1_sycl(v1, sycl::range<1>(size));
      sycl::buffer<PtEtaPhiEVector, 1> v2_sycl(v2, sycl::range<1>(size));
      sycl::buffer<double, 1> im_sycl(invMasses, sycl::range<1>(size));

      queue.submit([&](sycl::handler &cgh) {
         auto v1_acc = v1_sycl.get_access<sycl::access::mode::read>(cgh);
         auto v2_acc = v2_sycl.get_access<sycl::access::mode::read>(cgh);
         auto im_acc = im_sycl.get_access<sycl::access::mode::discard_write>(cgh);

         cgh.parallel_for<class invariant_masses>(sycl::range<1>(size), [=](sycl::item<1> item) {
            size_t id = item.get_linear_id();
            auto const local_v1 = v1_acc[id];
            auto const local_v2 = v2_acc[id];

            // Conversion from (pt, Eta(), phi, mass) to (x, y, z, e) coordinate system
            const auto x1 = local_v1.Pt() * sycl::cos(local_v1.Phi());
            const auto y1 = local_v1.Pt() * sycl::sin(local_v1.Phi());
            const auto z1 = local_v1.Pt() * sycl::sinh(local_v1.Eta());
            const auto e1 = local_v1.E();

            const auto x2 = local_v2.Pt() * sycl::cos(local_v2.Phi());
            const auto y2 = local_v2.Pt() * sycl::sin(local_v2.Phi());
            const auto z2 = local_v2.Pt() * sycl::sinh(local_v2.Eta());
            const auto e2 = local_v2.E();

            // Addition of particle four-vector elements
            const auto e = e1 + e2;
            const auto x = x1 + x2;
            const auto y = y1 + y2;
            const auto z = z1 + z2;

            auto mm = e * e - x * x - y * y - z * z;
            im_acc[id] = mm < 0 ? -sycl::sqrt(-mm) : sycl::sqrt(mm);
         });
      });
   }

   try {
      queue.wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }

   return invMasses;
}
} // namespace Experimental
} // namespace ROOT