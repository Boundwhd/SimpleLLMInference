#ifndef _ENCODE_WHD_H_
#define _ENCODE_WHD_H_
#include <string>
#include <vector>
#include <memory>
#include <sentencepiece_processor.h>

// Llama 3
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include "tiktoken.h"
#include "unordered_dense.h"
#include "nlohmann/json.hpp"

namespace op {
class SPELayer {
public:
    SPELayer(const std::string& model_file);

    std::vector<int32_t> encode(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;

    int GetVocabularySize() const;
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
};
}

#endif