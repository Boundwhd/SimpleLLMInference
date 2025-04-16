#include "add.h"
#include "add_kernel.h"
#include "add_kernel.cuh"

namespace op {
    VecAddLayer::VecAddLayer(base::DeviceType device_type, int32_t dim_size) 
        : Layer(device_type, LayerType::kLayerAdd, "Add"), dim_size_(dim_size) {
        reset_input_size(2);
        reset_output_size(1);
    }
    void VecAddLayer::forward() {
        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::add_kernel_cpu(input1, input2, output, dim_size_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            kernel::add_kernel_cuda(input1, input2, output, dim_size_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}