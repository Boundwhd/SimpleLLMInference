#pragma once
#ifndef _ROPE_WHD_H_
#define _ROPE_WHD_H_
#include "layer.h"

namespace op {
class RoPELayer : public Layer {
public:
    explicit RoPELayer(base::DeviceType device_type, int32_t hidden_dim_size, int32_t head_dim);

    void forward() override;
private:
    int32_t hidden_dim_size_ = 0;
    int32_t head_dim_ = 0;
};
}

#endif