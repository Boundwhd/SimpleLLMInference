#include "layer.h"
#include <cstdarg>
namespace op {
    // BaseLayer----------------------------------------------------------------------------------------------------
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, std::string layer_name) 
        : device_type_(device_type), layer_type_(layer_type), layer_name_(std::move(layer_name)) {}
    
    LayerType BaseLayer::layer_type() const { 
        return layer_type_; 
    }

    const std::string& BaseLayer::get_layer_name() const {
        return layer_name_;
    }

    void BaseLayer::set_layer_name(const std::string& layer_name) {
        layer_name_ = layer_name;
    }

    void BaseLayer::set_device_type(base::DeviceType device_type) {
        device_type_ = device_type;
    }

    base::DeviceType BaseLayer::device_type() const {
        return device_type_;
    }
    
    // Layer----------------------------------------------------------------------------------------------------
    Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name) 
        : BaseLayer(device_type, layer_type, std::move(layer_name)) {}

    void Layer::check_tensor(const mem::Tensor& tensor, base::DeviceType device_type) const {
        if (tensor.is_empty()) {
            LOG("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            LOG("The tensor has a wrong device type.");
        }
    }

    void Layer::check_tensor_with_dim(const mem::Tensor& tensor, base::DeviceType device_type, ...) const {
        std::va_list args;
        if (tensor.is_empty()) {
            LOG("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            LOG("The tensor has a wrong device type.");
        }

        va_start(args, device_type);
        int32_t dims = tensor.dims_size();
        for (int32_t i = 0; i < dims; ++i) {
            int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.get_dim(i)) {
                LOG("The tensor has a wrong dim in dim" + std::to_string(i));
            }
        }
        va_end(args);
    }

    void Layer::set_input(int32_t idx, const mem::Tensor& input) {
        this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const mem::Tensor& output) {
        this->outputs_.at(idx) = output;
    }

    const mem::Tensor& Layer::get_input(int32_t idx) const {
        return inputs_.at(idx);
    }

    mem::Tensor& Layer::get_input(int32_t idx) {
        return inputs_.at(idx);
    }

    mem::Tensor& Layer::get_output(int32_t idx) {
        return outputs_.at(idx);
    }

    const mem::Tensor& Layer::get_output(int32_t idx) const {
        return outputs_.at(idx);
    }

    size_t Layer::input_size() const {
        return inputs_.size();
    }

    size_t Layer::output_size() const {
        return outputs_.size();
    }

    void Layer::reset_input_size(size_t size) {
        inputs_.resize(size);
    }

    void Layer::reset_output_size(size_t size) {
        outputs_.resize(size);
    }
    
    void Layer::to_cuda() {
        for (auto& input : inputs_) {
            input.to_cuda();
        }
        for (auto& output : outputs_) {
            output.to_cuda();
        }
    }

    void Layer::forward(const mem::Tensor& input1, const mem::Tensor& output1) {
        this->set_input(0, input1);
        this->set_output(0, output1);
        return;
    }

    void Layer::forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_output(0, output1);
        return;
    }

    void Layer::forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_output(0, output1);
        return;
    }

    void Layer::forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& input4, const mem::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_output(0, output1);
        return;
    }

    void Layer::forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& input4, const mem::Tensor& input5, const mem::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);
        this->set_output(0, output1);
        return;
    }
    // ParamLayer----------------------------------------------------------------------------------------------------
    LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
        : Layer(device_type, layer_type, std::move(layer_name)) {}
    
    size_t LayerParam::weight_size() const {
        return weights_.size();
    }

    void LayerParam::reset_weight_size(size_t size) {
        weights_.resize(size);
    }

    mem::Tensor& LayerParam::get_weight(int32_t idx) {
        return weights_.at(idx);
    }

    const mem::Tensor& LayerParam::get_weight(int32_t idx) const {
        return weights_.at(idx);
    }

    void LayerParam::to_cuda() {
        Layer::to_cuda();
        for (auto& weight : weights_) {
            weight.to_cuda();
        }
    }

    void LayerParam::set_weight(int32_t idx, const mem::Tensor& weight) {
        if (!weight.is_empty()) {
            if (weight.device_type() != device_type_) {
                LOG("Device not the same!");
            }
            weights_.at(idx) = weight;
        }
    }

    void LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, base::DeviceType device_type) {
        if (weight_ptr == nullptr) {
            LOG("Ptr is empty!");
        }
        size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
        std::shared_ptr<mem::Buffer> buffer =std::make_shared<mem::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
        if (device_type != base::DeviceType::kDeviceUnknown) {
            buffer->set_device_type(device_type);
        }
        mem::Tensor weight(dims);
        weight.set_device_type(device_type);
        weight.assign(buffer);
        weights_.at(idx) = weight;
    }
};