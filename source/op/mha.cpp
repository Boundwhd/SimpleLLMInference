#include "mha.h"
#include "mha_kernel.h"
#include "mha_kernel.cuh"

namespace op {
    MultiHeadAttention::MultiHeadAttention(
        base::DeviceType device_type,
        int32_t max_seq_len,
        int32_t head_dim,
        int32_t num_attention_heads,
        int32_t num_key_value_heads
    ) : Layer(device_type, LayerType::kLayerMHA, "MultiHeadAttention"),
        max_seq_len_(max_seq_len),
        head_dim_(head_dim),
        num_attention_heads_(num_attention_heads),
        num_key_value_heads_(num_key_value_heads) {
        
        reset_input_size(4);
        reset_output_size(1);

        hidden_dim_ = num_attention_heads * head_dim;
        kv_hidden_dim_ = num_key_value_heads * head_dim;
        att_kv_head_group_ = num_attention_heads / num_key_value_heads;
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
            kernel::mha_kernel_cpu(query, score, key_cache, value_cache, mha_out, layer_index_, pos_, 
                max_seq_len_, head_dim_, hidden_dim_, kv_hidden_dim_, att_kv_head_group_, num_attention_heads_, device_type_);
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            kernel::mha_kernel_cuda(query, score, key_cache, value_cache, mha_out, layer_index_, pos_, 
                max_seq_len_, head_dim_, hidden_dim_, kv_hidden_dim_, att_kv_head_group_, num_attention_heads_, device_type_);
        } else {
            LOG("Device Type ERROR!");
        }
    }
}