#ifndef _MHA_CPU_H_
#define _MHA_CPU_H_
#include "tensor.h"
namespace kernel {
    void mha_kernel_cpu(const mem::Tensor& query,
        const mem::Tensor& score, const mem::Tensor& key_cache, const mem::Tensor& value_cache, 
        const mem::Tensor& mha_out, int32_t layer_index, int32_t pos, int32_t seq_len, int32_t dim,
        int32_t head_num, int32_t head_size, base::DeviceType device_type);
}
#endif