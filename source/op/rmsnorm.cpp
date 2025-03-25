#include "rmsnorm.h"
#include "layer.h"
#include "rms_kernel.cuh"
#include "rms_kernel.h"
namespace op {
    RmsNormLayer::RmsNormLayer(base::DeviceType device_type) 
        : LayerParam(device_type, LayerType::kLayerRMSNorm, "RMSNorm") {

        reset_input_size(1);
        reset_output_size(1);
        reset_weight_size(1);
    }

    void RmsNormLayer::forward() {
        auto input = this->get_input(0);
        auto weight = this->get_weight(0);
        auto output = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::rmsnorm_kernel_cpu(input, weight, output);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            kernel::rmsnorm_kernel_cuda(input, weight, output);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}