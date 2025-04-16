#ifndef _CONFIG_WHD_H_
#define _CONFIG_WHD_H_

namespace model {
struct LlamaModelConfig {
    int vocab_size = 128256;
    int head_dim = 128;
    int hidden_size = 3072;
    int kv_hidden_size = 1024;
    int intermediate_size = 8192;
    int max_length = 1024;
    int num_hidden_layers = 28;
    int num_attention_heads = 24;
    int num_key_value_heads = 8;
    float rms_norm_eps = 1e-05;
    float rope_theta = 100000.0f;
};
}
#endif