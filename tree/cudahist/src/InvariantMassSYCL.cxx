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

class vector_addition;

double *InvariantMassSYCL(const PtEtaPhiEVector *v1, const PtEtaPhiEVector *v2, size_t size)
{
   double *invMasses = new double[size];

   sycl::default_selector device_selector;
   sycl::queue queue(device_selector, exception_handler);
   std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

   sycl::float4 a = {1.0, 2.0, 3.0, 4.0};
   sycl::float4 b = {4.0, 3.0, 2.0, 1.0};
   sycl::float4 c = {0.0, 0.0, 0.0, 0.0};
   {
      sycl::buffer<sycl::float4, 1> a_sycl(&a, sycl::range<1>(1));
      sycl::buffer<sycl::float4, 1> b_sycl(&b, sycl::range<1>(1));
      sycl::buffer<sycl::float4, 1> c_sycl(&c, sycl::range<1>(1));

      queue.submit([&](sycl::handler &cgh) {
         auto a_acc = a_sycl.get_access<sycl::access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<sycl::access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<sycl::access::mode::discard_write>(cgh);

         cgh.single_task<class vector_addition>([=]() { c_acc[0] = a_acc[0] + b_acc[0]; });
      });
   }

   try {
      queue.wait_and_throw();
   } catch (sycl::exception const &e) {
      std::cout << "Caught synchronous SYCL exception:\n" << e.what() << std::endl;
   }

   std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
             << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
             << "------------------\n"
             << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }" << std::endl;

   return invMasses;
}
} // namespace Experimental
} // namespace ROOT