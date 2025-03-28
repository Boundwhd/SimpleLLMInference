#ifndef _MHA_WHD_H_
#define _MHA_WHD_H_
#include "layer.h"

namespace op {
class MultiHeadAttention : public op::Layer {
public:
    explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index, 
        int32_t seq_len, int32_t head_num, int32_t head_size, int32_t pos, int32_t dim);
    
    void set_pos(int32_t pos);

    void set_layer_index(int32_t index);
    
    void forward() override;
private:
    int32_t layer_index_ = 0;
    int32_t pos_ = 0;
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t head_num_ = 0;
    int32_t head_size_ = 0;
};
};


#endif 