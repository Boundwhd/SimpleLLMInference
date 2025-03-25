#ifndef _RMSNORM_WHD_H_
#define _RMSNORM_WHD_H_
#include "layer.h"

namespace op {
class RmsNormLayer : public LayerParam {
public:
    explicit RmsNormLayer(base::DeviceType device_type);

    void forward() override;
};
};
#endif