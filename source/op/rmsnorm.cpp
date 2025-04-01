#include "rmsnorm.h"
#include "rms_kernel.h"

namespace op {
    RmsNormLayer::RmsNormLayer(base::DeviceType device_type, int32_t hidden_dim_size, float eps) 
        : hidden_dim_size_(hidden_dim_size), eps_(eps), LayerParam(device_type, LayerType::kLayerRMSNorm, "RMSNorm") {

        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
    }

    void RmsNormLayer::forward() {
        auto input = this->get_input(0);
        auto weight = this->get_weight(0);
        auto output = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::rmsnorm_kernel_cpu(input, weight, output, hidden_dim_size_, eps_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // 待实现
        } else {
            LOG("Device Type ERROR!");
        }
    }
}