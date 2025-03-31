#pragma once
#ifndef _MATMUL_WHD_H_
#define _MATMUL_WHD_H_
#include "layer.h"

namespace op {
class MatmulLayer : public LayerParam {
public:
    explicit MatmulLayer(base::DeviceType device_type);

    void forward() override;
};

}

#endif