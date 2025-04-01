#pragma once
#ifndef _MATMUL_WHD_H_
#define _MATMUL_WHD_H_
#include "layer.h"

namespace op {
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1);

    void forward() override;

private:
    int32_t dim0_ = 0;  // weight 0
    int32_t dim1_ = 0;  // weight 1
};
}

#endif