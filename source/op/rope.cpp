#include "rope.h"
#include "rope_kernel.h"

namespace op {
    RoPELayer::RoPELayer(base::DeviceType device_type, int32_t dim, int32_t head_size) 
    : dim_(dim), head_size_(head_size), Layer(device_type, LayerType::kLayerRoPe, "RoPE") {
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
            kernel::rope_kernel_cpu(input_q, input_k, pos_now, sin_cache, cos_cache, dim_, head_size_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // 待实现
        } else {
            LOG("Device Type ERROR!");
        }
    }
}