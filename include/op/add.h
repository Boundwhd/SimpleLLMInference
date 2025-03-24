#ifndef _ADD_WHD_H_
#define _ADD_WHD_H_
#include "layer.h"

namespace op {
class VecAddLayer : public Layer {
public:
    explicit VecAddLayer(base::DeviceType device_type);

    void forward() override;
};
}
#endif