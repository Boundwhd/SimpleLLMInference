#include "mha.h"
#include "mha_kernel.h"
namespace op {
    MultiHeadAttention::MultiHeadAttention(base::DeviceType device_type, int32_t layer_index, 
        int32_t seq_len, int32_t head_num, int32_t head_size, int32_t pos, int32_t dim)
        :   Layer(device_type, LayerType::kLayerMHA, "MultiHeadAttention"),
            layer_index_(layer_index),
            seq_len_(seq_len),
            head_num_(head_num),
            head_size_(head_size),
            pos_(pos), 
            dim_(dim) {
        
        reset_input_size(4);
        reset_output_size(1);
    }
    
    void MultiHeadAttention::set_pos(int32_t pos) {
        pos_ = pos;
    }

    void MultiHeadAttention::set_layer_index(int32_t index) {
        layer_index_ = index;
    }

    void MultiHeadAttention::forward() {
        const mem::Tensor& query = this->get_input(0);
        const mem::Tensor& score = this->get_input(1);
        const mem::Tensor& key_cache = this->get_input(2);
        const mem::Tensor& value_cache = this->get_input(3);
        const mem::Tensor& mha_out = this->get_output(0);

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::mha_kernel_cpu(query, score, key_cache, value_cache, mha_out, layer_index_, pos_, seq_len_, dim_, head_num_, head_size_, device_type_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // kernel::mha_kernel_cuda(query, score, key_cache, value_cache, mha_out, layer_index_, pos_, seq_len_, dim_, head_num_, head_size_, device_type_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}