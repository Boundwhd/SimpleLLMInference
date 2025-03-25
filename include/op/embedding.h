#ifndef _EMBEDDING_WHD_H_
#define _EMBEDDING_WHD_H_
#include <utility>
#include "layer.h"

namespace op {
struct EmbeddingOutput {
    mem::Tensor input_tokens;
    mem::Tensor input_embeddings;
    mem::Tensor input_token_num;
    explicit EmbeddingOutput(mem::Tensor input_tokens, mem::Tensor input_embeddings, mem::Tensor input_token_num)
        : input_tokens(std::move(input_tokens)), input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam {
public:
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);

    void forward() override;
private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};
};
#endif