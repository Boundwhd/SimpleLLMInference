#include "add.h"
#include "add_kernel.h"

namespace op {
    VecAddLayer::VecAddLayer(base::DeviceType device_type) : Layer(device_type, LayerType::kLayerAdd, "Add") {
        reset_input_size(2);
        reset_output_size(1);
    }
    void VecAddLayer::forward() {
        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::add_kernel_cpu(input1, input2, output);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // 待实现
        } else {
            LOG("Device Type ERROR!");
        }
    }
}