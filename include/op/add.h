#pragma once
#ifndef _ADD_WHD_H_
#define _ADD_WHD_H_
#include "layer.h"

namespace op {
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type, int32_t dim_size);

    void forward() override;
private:
    int32_t dim_size_;
};
}
#endif