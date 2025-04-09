#include "rope.h"
#include "rope_kernel.h"
#include "rope_kernel.cuh"
namespace op {
    RoPELayer::RoPELayer(base::DeviceType device_type, int32_t hidden_dim_size, int32_t head_dim) 
    : hidden_dim_size_(hidden_dim_size), head_dim_(head_dim), Layer(device_type, LayerType::kLayerRoPe, "RoPE") {
        reset_input_size(4);
        reset_output_size(1);
    }

    void RoPELayer::forward() {
        auto input_q = this->get_input(0);
        auto input_k = this->get_input(1);
        auto pos_now = this->get_input(2);
        auto sin_cache = this->get_input(3);
        auto cos_cache = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::rope_kernel_cpu(input_q, input_k, pos_now, sin_cache, cos_cache, hidden_dim_size_, head_dim_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            kernel::rope_kernel_cuda(input_q, input_k, pos_now, sin_cache, cos_cache, hidden_dim_size_, head_dim_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}