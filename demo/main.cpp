#include "alloc.h"
#include "tensor.h"
#include "layer.h"
#include "add_kernel.h"
auto cpu_alloc = mem::CPUDeviceAllocatorFactory::get_instance();

int main() {
    std::cout << "go" << std::endl;
    std::vector<int> dims = {16};
    mem::Tensor A(dims, true, cpu_alloc);
    mem::Tensor B(dims, true, cpu_alloc);
    mem::Tensor C(dims, true, cpu_alloc);
    float A_H[16];
    float B_H[16];
    for (int i = 0; i < 16; i++) {
        A_H[i] = i + 1;
        B_H[i] = i + 2;
    }

    cpu_alloc->memcpy(A_H, A.ptr<float>(), 16*4, base::MemcpyKind::kMemcpyCPU2CPU);
    cpu_alloc->memcpy(B_H, B.ptr<float>(), 16*4, base::MemcpyKind::kMemcpyCPU2CPU);

    kernel::add_kernel_cpu(A, B, C);
    for (int i = 0; i < 16; i++) {
        std::cout << C.index<float>(i) << std::endl; 
    }
    return 0;
}