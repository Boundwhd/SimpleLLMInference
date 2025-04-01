#pragma once
#ifndef _RMSNORM_WHD_H_
#define _RMSNORM_WHD_H_
#include "layer.h"

namespace op {
class RmsNormLayer : public LayerParam {
public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t hidden_dim_size, float eps);

    void forward() override;

private:
    int32_t hidden_dim_size_ = 0;
    float eps_ = 0;
};

};
#endif