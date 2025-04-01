#pragma once
#ifndef _MHA_WHD_H_
#define _MHA_WHD_H_
#include "layer.h"

namespace op {
class MultiHeadAttention : public op::Layer {
public:
    explicit MultiHeadAttention(
        base::DeviceType device_type, 
        int32_t max_seq_len, 
        int32_t head_dim, 
        int32_t num_attention_heads,  
        int32_t num_key_value_heads
    );
    
    void set_pos(int32_t pos);

    void set_layer_index(int32_t index);
    
    void forward() override;
private:
    int32_t layer_index_ = 0;
    int32_t pos_ = 0;
    int32_t max_seq_len_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_dim_ = 0;
    int32_t kv_hidden_dim_ = 0;
    int32_t num_attention_heads_ = 0;
    int32_t num_key_value_heads_ = 0;
    int32_t att_kv_head_group_ = 0;
};
}

#endif 