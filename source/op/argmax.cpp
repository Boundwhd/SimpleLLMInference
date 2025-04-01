#include "argmax.h"
#include <algorithm>
namespace op {
    argmaxLayer::argmaxLayer(base::DeviceType device_type, int32_t hidden_dim_size) 
        : device_type_(device_type), hidden_dim_size_(hidden_dim_size) {}
    
    void argmaxLayer::forward(const mem::Tensor& logits, const mem::Tensor& input_idx) {
        const float* logits_ptr = logits.ptr<float>();

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            size_t next = std::distance(logits_ptr, std::max_element(logits_ptr, logits_ptr + hidden_dim_size_));
            *(const_cast<int32_t*>(input_idx.ptr<int32_t>())) = next;
        } else if (device_type_ == base::DeviceType::kDeviceCUDA){
            // 待实现
        } else {
            LOG("wrong device!\n");
        }
        
    }
}