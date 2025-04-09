#include "matmul.h"
#include "matmul_kernel.h"
#include "matmul_kernel.cuh"

namespace op {
    MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1)
    : dim0_(dim0), dim1_(dim1), LayerParam(device_type, LayerType::kLayerMatmul, "Matmul") {

        reset_input_size(1);
        reset_weight_size(1);
        reset_output_size(1);      
    }

    void MatmulLayer::forward() {
        auto input1 = get_input(0);
        auto weight = get_weight(0);
        auto output = get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::matmul_kernel_cpu(input1, weight, output, dim0_, dim1_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA) {
            kernel::matmul_kernel_cuda(input1, weight, output, dim0_, dim1_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}