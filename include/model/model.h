#ifndef _MODEL_WHD_H_
#define _MODEL_WHD_H_

#include "alloc.h"
#include "tensor.h"
#include "config.h"
#include "weight_loader.h"
#include <map>
#include "layer.h"
#include "encode.h"
#include "argmax.h"
namespace model {

enum class ModelBufferType {
    input_token = 0,        // 输入 token 序号，大小为 {1}
    position    = 1,        // 输入 token 位置，大小为 {1}
    key_cache   = 2,        // key cache，大小为 {layer_num，seq_len, kv_dim}
    value_cache = 3,        // value cache，大小为 {layer_num，seq_len, kv_dim}
    emb_output  = 4,        // embedding 层输出，大小为 {dim}
    rms_output  = 5,        // rmsnorm 层输出，大小为 {dim}
    query       = 6,        // wq 层输出, 大小为 {dim}
    score       = 7,        // attention 中间量，大小为 {head_dim, seq_len}
    mha_output  = 8,        // mha 层输出，大小为 {dim}
    att_output  = 9,        // attention 输出，大小为 {dim}
    ffn_input   = 10,       // ffn_input 过完 residual，大小为 {dim}
    up_output   = 11,       // up 层输出，大小为 {4 * dim}
    gate_output = 12,       // gate 层输出，大小为 {4 * dim}
    down_output = 13,       // down 层输出，大小为 {dim}
    swi_output  = 14,       // swiglu 层输出，大小为 {4 * dim}
    ffn_output  = 15,       // ffn 输出，大小为 {dim}
    model_pred  = 16,       // 每个词的概率，大小为 {vocab_size}
    sin_cache   = 17,       // 旋转位置编码，sin cacahe 大小 {seq_len, head_dim / 2}
    cos_cache   = 18,       // 旋转位置编码，cos cacahe 大小 {seq_len, head_dim / 2}
};

struct LlamaLayer {
    std::shared_ptr<op::SPELayer> encode_layer_;
    std::shared_ptr<op::argmaxLayer> argmax_layer_;

    std::shared_ptr<op::Layer> add_layer_;
    std::shared_ptr<op::Layer> rope_layer_;
    std::shared_ptr<op::Layer> swiglu_layer_;
    std::shared_ptr<op::Layer> mha_layer_;
    std::shared_ptr<op::Layer> emb_layer_;
    std::shared_ptr<op::Layer> cls_layer;

    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;
    
    std::vector<std::shared_ptr<op::Layer>> up_layers_;
    std::vector<std::shared_ptr<op::Layer>> gate_layers_;
    std::vector<std::shared_ptr<op::Layer>> down_layers_;

    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
};

class LlamaModel {
public:
    explicit LlamaModel(std::string tokenizer_path, std::string model_path, base::DeviceType device_type);

    void init();

    void forward();

    void predict(const std::string prompt, const int max_length);

protected:
    void init_mem();

    void insert_buffer(ModelBufferType buffer_idx, const mem::Tensor& tensor);

    const mem::Tensor& get_buffer(ModelBufferType buffer_idx);

    void read_model_file();

    void create_param_layers();
    
    void create_nonparam_layers();

    std::unique_ptr<LlamaModelConfig> config_;
    std::string tokenizer_path_;
    std::string model_path_;
    std::map<ModelBufferType, mem::Tensor> buffers_;
    std::shared_ptr<RawModelData> raw_model_data_;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    std::unique_ptr<LlamaLayer> Llama_layers_;
};
};
#endif