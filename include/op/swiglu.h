#ifndef _SWIGLU_WHD_H_
#define _SWIGLU_WHD_H_
#include "layer.h"

namespace op {
class SwigluLayer : public Layer {
public:
    explicit SwigluLayer(base::DeviceType, int32_t intermediate_size);

    void forward() override;
private:
    int32_t intermediate_size_ = 0;
};
}

#endif