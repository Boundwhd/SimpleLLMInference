#ifndef _ARGMAX_WHD_H_
#define _ARGMAX_WHD_H_
#include "layer.h"
namespace op {
class argmaxLayer {
public:
    explicit argmaxLayer(base::DeviceType device_type, int32_t hidden_dim_size);

    void forward(const mem::Tensor& logits, const mem::Tensor& input_idx);

private:
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    int32_t hidden_dim_size_ = 0;
};
}
#endif