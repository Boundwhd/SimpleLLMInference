#include "alloc.h"
#include "tensor.h"
#include "add.h"

auto cuda_alloc = mem::CUDADeviceAllocatorFactory::get_instance();

int main() {
    std::vector<int> dims = {16};
    mem::Tensor A(dims, true, cuda_alloc);
    mem::Tensor B(dims, true, cuda_alloc);
    mem::Tensor C(dims, true, cuda_alloc);
    float A_H[16];
    float B_H[16];
    float C_H[16];
    for (int i = 0; i < 16; i++) {
        A_H[i] = i + 1;
        B_H[i] = i + 2;
    }
    cuda_alloc->memcpy(A_H, A.ptr<float>(), 16*4, base::MemcpyKind::kMemcpyCPU2CUDA);
    cuda_alloc->memcpy(B_H, B.ptr<float>(), 16*4, base::MemcpyKind::kMemcpyCPU2CUDA);

    std::shared_ptr<op::Layer> add_layer_;
    add_layer_ = std::make_shared<op::VecAddLayer>(base::DeviceType::kDeviceCUDA);
    add_layer_->forward(A, B, C);
    cuda_alloc->memcpy(C.ptr<float>(), C_H, 16*4, base::MemcpyKind::kMemcpyCUDA2CPU);
    for (int i = 0; i < 16; i++) {
        std::cout << C_H[i] << std::endl;
    }
    return 0;
}