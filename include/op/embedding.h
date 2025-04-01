#pragma once
#ifndef _EMBEDDING_WHD_H_
#define _EMBEDDING_WHD_H_
#include "layer.h"

namespace op {
class EmbeddingLayer : public LayerParam {
public:
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t vocab_size, int32_t hidden_dim_size);
    
    void forward() override;
private:
    int32_t vocab_size_ = 0;
    int32_t hidden_dim_size_ = 0;
};
};

#endif