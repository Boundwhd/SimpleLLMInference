#include "model.h"
#include "add.h"
#include "mha.h"
#include "embedding.h"
#include "rope.h"
#include "swiglu.h"
#include "matmul.h"
#include "rmsnorm.h"
#include "rope_kernel.h"
#include "rope_kernel.cuh"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <unistd.h>

namespace model {
    LlamaModel::LlamaModel(std::string tokenizer_path, std::string model_path, base::DeviceType device_type) 
    : tokenizer_path_(std::move(tokenizer_path)), model_path_(std::move(model_path)), device_type_(device_type){}

    void LlamaModel::init() {
        // if (tokenizer_path_.empty()) {
        //     LOG("Path is invalid!");
        // }

        config_ = std::make_unique<LlamaModelConfig>();

        read_model_file();

        Llama_layers_ = std::make_unique<LlamaLayer>();

        create_param_layers();

        create_nonparam_layers();

        init_mem();
    }

    void LlamaModel::forward() {

        std::ofstream log_file((device_type_ == base::DeviceType::kDeviceCPU) ? "layer_outputs_cpu.txt" : "layer_outputs_cuda.txt");
        
        auto input_token = get_buffer(ModelBufferType::input_token);
        auto pos_tensor = get_buffer(ModelBufferType::position);
        int pos = pos_tensor.index<int>(0);
    
        Llama_layers_->emb_layer_->forward(input_token, get_buffer(ModelBufferType::emb_output));
        
        for (int id_layer = 0; id_layer < config_->num_hidden_layers; id_layer++) {

            Llama_layers_->rmsnorm_layers_[2*id_layer]->forward(get_buffer(ModelBufferType::emb_output), get_buffer(ModelBufferType::rms_output));

            const auto& [key, value] = mem::slice_KV_cache(id_layer, pos, config_->max_length, config_->kv_hidden_size, 
                get_buffer(ModelBufferType::key_cache), get_buffer(ModelBufferType::value_cache));
            

            Llama_layers_->wq_layers_[id_layer]->forward(get_buffer(ModelBufferType::rms_output), get_buffer(ModelBufferType::query));
            
            Llama_layers_->wk_layers_[id_layer]->forward(get_buffer(ModelBufferType::rms_output), key);

            Llama_layers_->wv_layers_[id_layer]->forward(get_buffer(ModelBufferType::rms_output), value);
            auto query = get_buffer(ModelBufferType::query);


            Llama_layers_->rope_layer_->forward(get_buffer(ModelBufferType::query), key, pos_tensor, 
                get_buffer(ModelBufferType::sin_cache), get_buffer(ModelBufferType::cos_cache));


            std::dynamic_pointer_cast<op::MultiHeadAttention>(Llama_layers_->mha_layer_)->set_pos(pos);
            std::dynamic_pointer_cast<op::MultiHeadAttention>(Llama_layers_->mha_layer_)->set_layer_index(id_layer);
            Llama_layers_->mha_layer_->forward(
                get_buffer(ModelBufferType::query),
                get_buffer(ModelBufferType::score),
                get_buffer(ModelBufferType::key_cache),
                get_buffer(ModelBufferType::value_cache),
                get_buffer(ModelBufferType::mha_output)
            );

            Llama_layers_->wo_layers_[id_layer]->forward(
                get_buffer(ModelBufferType::mha_output),
                get_buffer(ModelBufferType::att_output)
            );

    
            Llama_layers_->add_layer_->forward(
                get_buffer(ModelBufferType::emb_output), 
                get_buffer(ModelBufferType::att_output),
                get_buffer(ModelBufferType::ffn_input)
            );


            Llama_layers_->rmsnorm_layers_[2 * id_layer + 1]->forward(
                get_buffer(ModelBufferType::ffn_input),
                get_buffer(ModelBufferType::rms_output)
            );


            Llama_layers_->up_layers_[id_layer]->forward(
                get_buffer(ModelBufferType::rms_output),
                get_buffer(ModelBufferType::up_output)
            );


            Llama_layers_->gate_layers_[id_layer]->forward(
                get_buffer(ModelBufferType::rms_output),
                get_buffer(ModelBufferType::gate_output)
            );


            Llama_layers_->swiglu_layer_->forward(
                get_buffer(ModelBufferType::up_output),
                get_buffer(ModelBufferType::gate_output),
                get_buffer(ModelBufferType::swi_output)
            );


            Llama_layers_->down_layers_[id_layer]->forward(
                get_buffer(ModelBufferType::swi_output),
                get_buffer(ModelBufferType::ffn_output)
            );


            Llama_layers_->add_layer_->forward(
                get_buffer(ModelBufferType::ffn_output), 
                get_buffer(ModelBufferType::ffn_input),
                get_buffer(ModelBufferType::emb_output)
            );
        }

        Llama_layers_->rmsnorm_layers_[2 * config_->num_hidden_layers]->forward(
            get_buffer(ModelBufferType::emb_output),
            get_buffer(ModelBufferType::rms_output)
        );

        Llama_layers_->cls_layer->forward(
            get_buffer(ModelBufferType::rms_output),
            get_buffer(ModelBufferType::model_pred)
        );
    }

    void LlamaModel::predict(const std::string prompt, const int max_length) {

        // 初始提示词数量
        std::vector<int> input_ids = Llama_layers_->encode_layer_->encode(prompt);
        int prompt_nums = input_ids.size();

        int32_t pos  = 0;
        auto input_tensor = get_buffer(ModelBufferType::input_token);
        auto pos_tensor = get_buffer(ModelBufferType::position);
        input_tensor.index<int32_t>(0) = input_ids[pos]; 
        pos_tensor.index<int32_t>(0) = pos; 

        std::string out_str = Llama_layers_->encode_layer_->decode({input_ids[pos]});
        std::cout << out_str << " ";

        while (pos < max_length) {
            this->forward();
            if (pos < prompt_nums - 1) {
                pos++;
                int32_t next = input_ids[pos];
                out_str = Llama_layers_->encode_layer_->decode({next});
                std::cout << out_str << " ";
                pos_tensor.index<int32_t>(0) = pos;
                input_tensor.index<int32_t>(0) = next;
            } else {
                pos++;
                pos_tensor.index<int32_t>(0) = pos;
                if (device_type_ == base::DeviceType::kDeviceCPU) {
                    Llama_layers_->argmax_layer_->forward(get_buffer(ModelBufferType::model_pred), input_tensor);
                    int32_t next = input_tensor.index<int32_t>(0);
                    out_str = Llama_layers_->encode_layer_->decode({next});
                    std::cout << out_str << " ";
                } else {
                    auto cpu_alloc = mem::CPUDeviceAllocatorFactory::get_instance();
                    auto model_pred_cuda = get_buffer(ModelBufferType::model_pred);
                    auto model_pred_cpu = mem::Tensor({config_->vocab_size}, true, cpu_alloc);
                    cpu_alloc->memcpy(model_pred_cuda.ptr<float>(), model_pred_cpu.ptr<float>(), config_->vocab_size * sizeof(float), base::MemcpyKind::kMemcpyCUDA2CPU);
                    Llama_layers_->argmax_layer_->forward(model_pred_cpu, input_tensor);
                    int32_t next = input_tensor.index<int32_t>(0);
                    out_str = Llama_layers_->encode_layer_->decode({next});
                    std::cout << out_str << " ";
                }
            }
        }
        std::cout << std::endl;
    }

    void LlamaModel::insert_buffer(ModelBufferType buffer_idx, const mem::Tensor& tensor) {
        if (buffers_.count(buffer_idx) > 0) {
            LOG(std::to_string(int(buffer_idx)) + " has exits in the buffers\n");
        }

        if (tensor.is_empty()) {
            LOG("The tensor is empty for inserting buffer.");
        }
        buffers_.insert({buffer_idx, tensor});
    }

    const mem::Tensor& LlamaModel::get_buffer(ModelBufferType buffer_idx) {
        return buffers_.at(buffer_idx);   
    }

    void LlamaModel::read_model_file() {
        if (model_path_.empty()) {
            LOG("No model weigth file!\n");
        }

        int32_t fd = open(model_path_.data(), O_RDONLY);
        if (fd == -1) {
            LOG("Fail to open the weight file!\n");
        }

        FILE* file = fopen(model_path_.data(), "rb");
        if (!file) {
            LOG("Failed to open the file. The path may be invalid.");
        }

        auto config = LlamaModelConfig();
        config_->head_dim = config.head_dim;
        config_->hidden_size = config.hidden_size;
        config_->intermediate_size = config.intermediate_size;
        config_->kv_hidden_size = config.kv_hidden_size;
        config_->max_length = config.max_length;
        config_->num_attention_heads = config.num_attention_heads;
        config_->num_hidden_layers = config.num_hidden_layers;
        config_->num_key_value_heads = config.num_key_value_heads;
        config_->rms_norm_eps = config.rms_norm_eps;
        config_->rope_theta = config.rope_theta;
        config_->vocab_size = config.vocab_size;

        raw_model_data_ = std::make_shared<RawModelDataFp32>();
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            LOG("Failed to retrieve the file size information from the model file\n.");
        }
        raw_model_data_->file_size = sb.st_size;
        raw_model_data_->fd = fd;
        raw_model_data_->weight_data = mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

        if (!raw_model_data_->weight_data) {
            LOG("Weight file wrong!\n");
        }
    }

    void LlamaModel::init_mem() {
        std::shared_ptr<mem::DeviceAllocator> alloc;
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            alloc = mem::CPUDeviceAllocatorFactory::get_instance();
        } else {
            alloc = mem::CUDADeviceAllocatorFactory::get_instance();
        }

        std::shared_ptr<mem::DeviceAllocator> alloc_cpu = mem::CPUDeviceAllocatorFactory::get_instance();
        std::shared_ptr<mem::DeviceAllocator> alloc_cu = mem::CUDADeviceAllocatorFactory::get_instance();
        
        mem::Tensor input_token({1}, true, alloc_cpu);              // cpu 分配
        insert_buffer(ModelBufferType::input_token, input_token);

        mem::Tensor pos_tensor({1}, true, alloc_cpu);               // cpu 分配
        insert_buffer(ModelBufferType::position, pos_tensor);

        mem::Tensor key_cache({config_->num_hidden_layers, config_->max_length, config_->kv_hidden_size}, true, alloc);
        mem::Tensor value_cache({config_->num_hidden_layers, config_->max_length, config_->kv_hidden_size}, true, alloc);

        insert_buffer(ModelBufferType::key_cache, key_cache);
        insert_buffer(ModelBufferType::value_cache, value_cache);

        mem::Tensor emb_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::emb_output, emb_output);

        mem::Tensor rms_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::rms_output, rms_output);

        mem::Tensor query({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::query, query);

        mem::Tensor score({config_->head_dim, config_->max_length}, true, alloc);
        insert_buffer(ModelBufferType::score, score);

        mem::Tensor mha_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::mha_output, mha_output);

        mem::Tensor attn_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::att_output, attn_output);

        mem::Tensor ffn_input({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::ffn_input, ffn_input);

        mem::Tensor up_output({config_->intermediate_size}, true, alloc);
        insert_buffer(ModelBufferType::up_output, up_output);

        mem::Tensor gate_output({config_->intermediate_size}, true, alloc);
        insert_buffer(ModelBufferType::gate_output, gate_output);

        mem::Tensor down_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::down_output, down_output);

        mem::Tensor swiglu_output({config_->intermediate_size}, true, alloc);
        insert_buffer(ModelBufferType::swi_output, swiglu_output);

        mem::Tensor ffn_output({config_->hidden_size}, true, alloc);
        insert_buffer(ModelBufferType::ffn_output, ffn_output);

        mem::Tensor model_pred({config_->vocab_size}, true, alloc);
        insert_buffer(ModelBufferType::model_pred, model_pred);

        mem::Tensor sin_cache({config_->max_length, config_->head_dim / 2}, true, alloc);       
        mem::Tensor cos_cache({config_->max_length, config_->head_dim / 2}, true, alloc);      

        if (device_type_ == base::DeviceType::kDeviceCPU) {
            kernel::rope_cache_cal(config_->head_dim, config_->max_length, sin_cache, cos_cache, config_->rope_theta);
        } else {
            kernel::rope_cache_cal_cuda(config_->head_dim, config_->max_length, sin_cache, cos_cache, config_->rope_theta);
        }
        

        insert_buffer(ModelBufferType::sin_cache, sin_cache);
        insert_buffer(ModelBufferType::cos_cache, cos_cache);
    }

    void LlamaModel::create_nonparam_layers() {
        if (!Llama_layers_) {
            LOG("Llama layers are not initiallized\n");
        }
        Llama_layers_->argmax_layer_ = std::make_shared<op::argmaxLayer>(base::DeviceType::kDeviceCPU, config_->vocab_size);
        Llama_layers_->encode_layer_ = std::make_shared<op::SPELayer>(tokenizer_path_);
        Llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_, config_->hidden_size);
        Llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(device_type_, config_->max_length, config_->head_dim, 
            config_->num_attention_heads, config_->num_key_value_heads);
        Llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(device_type_, config_->hidden_size, config_->head_dim);
        Llama_layers_->swiglu_layer_ = std::make_shared<op::SwigluLayer>(device_type_, config_->intermediate_size);
    }

    void LlamaModel::create_param_layers() {
        if (!Llama_layers_) {
            LOG("Llama layers are not initiallized\n");
        }
        size_t pos_weight = 0;

        Llama_layers_->emb_layer_ = std::make_shared<op::EmbeddingLayer>(device_type_, config_->vocab_size, config_->hidden_size);
        Llama_layers_->emb_layer_->set_weight(0, {config_->vocab_size, config_->hidden_size}, 
            raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            Llama_layers_->emb_layer_->to_cuda();
        }

        Llama_layers_->cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size, config_->hidden_size);
        Llama_layers_->cls_layer->set_weight(0, {config_->vocab_size, config_->hidden_size}, 
            raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            Llama_layers_->cls_layer->to_cuda();
        }

        pos_weight += config_->vocab_size * config_->hidden_size;

        for (int i = 0; i < 2 * config_->num_hidden_layers + 1; i++) {
            Llama_layers_->rmsnorm_layers_.emplace_back(std::make_shared<op::RmsNormLayer>(device_type_, config_->hidden_size, config_->rms_norm_eps));
            Llama_layers_->rmsnorm_layers_[i]->set_weight(0, {config_->hidden_size}, raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < 2 * config_->num_hidden_layers + 1; i++) {
                Llama_layers_->rmsnorm_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->wq_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->hidden_size, config_->hidden_size));
            Llama_layers_->wq_layers_[i]->set_weight(0, {config_->hidden_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->hidden_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->wq_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->wk_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->kv_hidden_size, config_->hidden_size));
            Llama_layers_->wk_layers_[i]->set_weight(0, {config_->kv_hidden_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->kv_hidden_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->wk_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->wv_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->kv_hidden_size, config_->hidden_size));
            Llama_layers_->wv_layers_[i]->set_weight(0, {config_->kv_hidden_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->kv_hidden_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->wv_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->wo_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->hidden_size, config_->hidden_size));
            Llama_layers_->wo_layers_[i]->set_weight(0, {config_->hidden_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->hidden_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->wo_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->up_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->intermediate_size, config_->hidden_size));
            Llama_layers_->up_layers_[i]->set_weight(0, {config_->intermediate_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->intermediate_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->up_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->gate_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->intermediate_size, config_->hidden_size));
            Llama_layers_->gate_layers_[i]->set_weight(0, {config_->intermediate_size, config_->hidden_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->intermediate_size * config_->hidden_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->gate_layers_[i]->to_cuda();
            }
        }

        for (int i = 0; i < config_->num_hidden_layers; i++) {
            Llama_layers_->down_layers_.emplace_back(std::make_shared<op::MatmulLayer>(
                device_type_, config_->hidden_size, config_->intermediate_size));
            Llama_layers_->down_layers_[i]->set_weight(0, {config_->hidden_size, config_->intermediate_size}, 
                raw_model_data_->weight(pos_weight), base::DeviceType::kDeviceCPU);
            pos_weight += config_->hidden_size * config_->intermediate_size;
        }

        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            for (int i = 0; i < config_->num_hidden_layers; i++) {
                Llama_layers_->down_layers_[i]->to_cuda();
            }
        }
    }   
}