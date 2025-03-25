#include "emb_kernel.h"

namespace kernel {
    void emb_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t vocab_size) {
            
        const int32_t weight_dim = weight.get_dim(1);
        const auto allocator = mem::CPUDeviceAllocatorFactory::get_instance();
        
        int32_t token = *input.ptr<int32_t>();
        if (token > vocab_size) {
            LOG("Token index is greater than vocab size.");
        } else {
            float* dst_ptr = const_cast<float*>(output.ptr<float>());
            float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
            if (weight.device_type() == base::DeviceType::kDeviceCPU) {
                allocator->memcpy(src_ptr, dst_ptr, weight_dim * sizeof(float), base::MemcpyKind::kMemcpyCPU2CPU);
            } else {
                LOG("Unknown device type of weight tensor in the embedding layer.");
            }
        }
    }
}
