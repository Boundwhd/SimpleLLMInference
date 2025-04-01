#include "encode.h"
#include <stdexcept>

namespace op{
    SPELayer::SPELayer(const std::string& model_file) {
        processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
        const auto status = processor_->Load(model_file);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
    }

    std::vector<int32_t> SPELayer::encode(const std::string& text) const {
        std::vector<int> ids;
        processor_->Encode(text, &ids);
        return ids;
    }

    std::string SPELayer::decode(const std::vector<int32_t>& ids) const {
        std::string text;
        processor_->Decode(ids, &text);
        return text;
    }

    int SPELayer::GetVocabularySize() const {
        return processor_->GetPieceSize();
    }
}