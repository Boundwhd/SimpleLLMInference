#pragma once
#ifndef _ROPE_WHD_H_
#define _ROPE_WHD_H_
#include "layer.h"

namespace op {
class RoPELayer : public Layer {
public:
    explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t head_size);

    void forward() override;
private:
    int32_t dim_ = 0;
    int32_t head_size_ = 0;
};
}

#endif