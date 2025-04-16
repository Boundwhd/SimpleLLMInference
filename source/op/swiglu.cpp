#include "swiglu.h"
#include "swiglu_kernel.h"
#include "swiglu_kernel.cuh"

namespace op {
    SwigluLayer::SwigluLayer(base::DeviceType device_type, int32_t intermediate_size) 
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"), intermediate_size_(intermediate_size) {
        
        reset_input_size(2);
        reset_output_size(1);
    }

    void SwigluLayer::forward() {
        auto up = this->get_input(0);
        auto gate = this->get_input(1);
        auto output = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::swiglu_kernel_cpu(up, gate, output, intermediate_size_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            kernel::swiglu_kernel_cuda(up, gate, output, intermediate_size_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}