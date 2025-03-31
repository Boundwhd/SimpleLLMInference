#include "alloc.h"
#include "layer.h"
#include "weight_loader.h"
#include "rope_kernel.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
// ------------------------------------------------------------------------------------------------
#include "add.h"
#include "embedding.h"
#include "matmul.h"
#include "rope.h"
#include "rmsnorm.h"
#include "mha.h"


// 配置参数--------------------------------------------------------------------------------------------
int32_t dim = 768;
int32_t num_layer = 1;
int32_t max_seq_len = 1024;
int32_t vocab_size = 32;
int32_t head_num = 12;
int32_t head_size = dim / head_num;

int main() {
    // 内存管理器初始化--------------------------------------------------------------------------------
    auto alloc= mem::CPUDeviceAllocatorFactory::get_instance();
    auto device_type = base::DeviceType::kDeviceCPU;

    // 外部模型权重读入管理------------------------------------------------------------------------------
    const char* model_path = "../pytorch/export/model_weights.bin";
    int32_t fd = open(model_path, O_RDONLY);
    FILE* file = fopen(model_path, "rb");
    struct stat sb;
    std::shared_ptr<model::RawModelData> raw_model_data_;
    raw_model_data_ = std::make_shared<model::RawModelDataFp32>();

    // 内存资源初始化-----------------------------------------------------------------------------------
    mem::Tensor input_token({1}, true, alloc);                              // 使用CPU分配，不参与CUDA计算
    mem::Tensor pos_tensor({1}, true, alloc);                               // 使用CPU分配，不参与CUDA计算

    mem::Tensor key_cache({num_layer, max_seq_len, dim}, true, alloc);      // key cache
    mem::Tensor value_cache({num_layer, max_seq_len, dim}, true, alloc);    // value cache

    mem::Tensor emb_output({dim}, true, alloc);
    mem::Tensor rms_output({dim}, true, alloc);
    mem::Tensor query({dim}, true, alloc);
    mem::Tensor score({head_num, max_seq_len}, true, alloc);
    mem::Tensor attn_output({dim}, true, alloc);

    mem::Tensor sin_cache({max_seq_len, head_size / 2}, true, alloc);       // sin 旋转位置编码 cache   初始化大小 head_size / 2
    mem::Tensor cos_cache({max_seq_len, head_size / 2}, true, alloc);       // cos 旋转位置编码 cache   初始化大小 head_size / 2

    kernel::rope_cache_cal(head_size, max_seq_len, sin_cache, cos_cache);   

    // 权重--------------------------------------------------------------------------------------------
    fstat(fd, &sb);
    raw_model_data_->file_size = sb.st_size;
    raw_model_data_->fd = fd;
    raw_model_data_->weight_data = mmap(nullptr, raw_model_data_->file_size, 
        PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

    // 创建层------------------------------------------------------------------------------------------
    std::shared_ptr<op::Layer> emb_layer;
    std::shared_ptr<op::Layer> rope_layer;
    std::shared_ptr<op::Layer> mha_layer;
    std::vector<std::shared_ptr<op::Layer>> rms_layer_;
    std::vector<std::shared_ptr<op::Layer>> wq_;
    std::vector<std::shared_ptr<op::Layer>> wk_;
    std::vector<std::shared_ptr<op::Layer>> wv_;

    int32_t pos_weight = 0;

    // 无权重层--------------------------------------------------------------------------------------------
    rope_layer = std::make_shared<op::RoPELayer>(device_type, vocab_size, head_size);
    mha_layer = std::make_shared<op::MultiHeadAttention>(device_type, 0, max_seq_len, head_num, head_size, 0, dim);

    // 带权重层--------------------------------------------------------------------------------------------
    emb_layer = std::make_shared<op::EmbeddingLayer>(device_type, vocab_size);
    emb_layer->set_weight(0, {vocab_size, dim}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
    pos_weight += vocab_size * dim;

    for (int i = 0; i < num_layer; i++) {
        rms_layer_.emplace_back(std::make_shared<op::RmsNormLayer>(device_type));
        rms_layer_[i]->set_weight(0, {dim}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
        pos_weight += dim;
    }
    
    for (int i = 0; i < num_layer; i++) {
        wq_.emplace_back(std::make_shared<op::MatmulLayer>(device_type));
        wq_[i]->set_weight(0, {dim, dim}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
        pos_weight += dim * dim;
    }

    for (int i = 0; i < num_layer; i++) {
        wk_.emplace_back(std::make_shared<op::MatmulLayer>(device_type));
        wk_[i]->set_weight(0, {dim, dim}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
        pos_weight += dim * dim;
    }

    for (int i = 0; i < num_layer; i++) {
        wv_.emplace_back(std::make_shared<op::MatmulLayer>(device_type));
        wv_[i]->set_weight(0, {dim, dim}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
        pos_weight += dim * dim;
    }

    // 模型配置完成，开始测试-------------------------------------------------------------------------------------------

    int pos = 0;        // 第一个token
    int max_len = 2;    // 最大输出限制长度

    int next = 2;       // 经过encode后，第一个词语的 ids = 2
    while(pos < max_len) {    
        input_token.index<int32_t>(0) = next;
        pos_tensor.index<int32_t>(0) = pos;

        emb_layer->forward(input_token, attn_output);
        for (int id_layer = 0; id_layer < num_layer; id_layer++) {
            // 过rmsnorm
            rms_layer_[id_layer]->forward(attn_output, rms_output);

            // 生成q, k, v tensor (k, v 要从kv cache切分出)
            const auto& [key, value] = mem::slice_KV_cache(id_layer, pos, max_seq_len, dim, key_cache, value_cache);

            wq_[id_layer]->forward(rms_output, query);
            wk_[id_layer]->forward(rms_output, key);
            wv_[id_layer]->forward(rms_output, value);

            rope_layer->forward(query, key, pos_tensor, sin_cache, cos_cache);
            std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
            std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_index(id_layer);
            mha_layer->forward(query, score, key_cache, value_cache, attn_output);
        }
        pos++;
        next = 3;
    }


    std::ofstream out_file("result.txt");
    if (out_file.is_open()) {
        out_file << std::fixed << std::setprecision(6);
        for (int i = 0; i < dim; i++) {
            out_file << attn_output.index<float>(i) << std::endl;
        }
    }
    
    return 0;
}

