#include "embedding.h"
#include "emb_kernel.h"

namespace op {
    EmbeddingLayer::EmbeddingLayer(base::DeviceType device_type, int32_t vocab_size, int32_t hidden_dim_size)
        : vocab_size_(vocab_size), hidden_dim_size_(hidden_dim_size), LayerParam(device_type, LayerType::kLayerEmbedding, "Embedding") {

        reset_weight_size(1);
        reset_input_size(1);
        reset_output_size(1);
    }

    void EmbeddingLayer::forward() {
        auto input1 = get_input(0);
        auto weight = get_weight(0);
        auto output = get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::emb_kernel_cpu(input1, weight, output, vocab_size_, hidden_dim_size_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // 待实现
        } else {
            LOG("Device Type ERROR!");
        }
    }
}