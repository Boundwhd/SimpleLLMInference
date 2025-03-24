#ifndef _LAYER_WHD_H_
#define _LAYER_WHD_H_
#include "alloc.h"
#include "tensor.h"

namespace op {
// Llama 中使用到的所有网络层
enum class LayerType : uint8_t {
    kLayerUnknown = 0,
    kLayerLinear = 1,
    kLayerEncode = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm = 4,
    kLayerMatmul = 5,
    kLayerRoPe = 6,
    kLayerMHA = 7,
    kLayerSoftmax = 8,
    kLayerAdd = 9,
    kLayerSwiGLU = 10,
};

class BaseLayer {
protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;   

public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

    LayerType layer_type() const;

    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name);

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

    virtual void init() = 0;

    virtual void forward() = 0;

    virtual void forward(const mem::Tensor& input1, const mem::Tensor& output1) = 0;

    virtual void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output1) = 0;
        
    virtual void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3,
        const mem::Tensor& output1) = 0;
    
    virtual void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3,
        const mem::Tensor& input4, const mem::Tensor& output1) = 0;
    
    virtual void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3,
        const mem::Tensor& input4, const mem::Tensor& input5, const mem::Tensor& output1) = 0;
    
    virtual void set_input(int32_t idx, const mem::Tensor& input) = 0;

    virtual void set_output(int32_t idx, const mem::Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual void check() const = 0;

    virtual mem::Tensor& get_input(int32_t idx) = 0;

    virtual mem::Tensor& get_output(int32_t idx) = 0;

    virtual const mem::Tensor& get_input(int32_t idx) const = 0;

    virtual const mem::Tensor& get_output(int32_t idx) const = 0;

    virtual void set_weight(int32_t idx, const mem::Tensor& weight) = 0;

    virtual void set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, 
        base::DeviceType device_type = base::DeviceType::kDeviceUnknown) = 0;
};

class Layer : public BaseLayer {
public:
    explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

    void check_tensor(const mem::Tensor& tensor, base::DeviceType device_type) const;

    void check_tensor_with_dim(const mem::Tensor& tensor, base::DeviceType device_type, ...) const;

    void set_input(int32_t idx, const mem::Tensor& input) override;

    void set_output(int32_t idx, const mem::Tensor& output) override;

    const mem::Tensor& get_input(int32_t idx) const override;

    const mem::Tensor& get_output(int32_t idx) const override;
  
    mem::Tensor& get_input(int32_t idx) override;
  
    mem::Tensor& get_output(int32_t idx) override;

    size_t input_size() const override;

    size_t output_size() const override;

    void reset_input_size(size_t size);

    void reset_output_size(size_t size);

    virtual void to_cuda();

    void forward(const mem::Tensor& input1, const mem::Tensor& output1) override;

    void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output1) override;

    void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& output1) override;
    
    void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& input4, const mem::Tensor& output1) override;
    
    void forward(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& input3, 
        const mem::Tensor& input4, const mem::Tensor& input5, const mem::Tensor& output1) override;
        
protected:
    std::vector<mem::Tensor> inputs_;
    std::vector<mem::Tensor> outputs_;
};

class LayerParam : public Layer {
public:
    explicit LayerParam(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

    size_t weight_size() const;

    void reset_weight_size(size_t size);

    mem::Tensor& get_weight(int32_t idx);

    const mem::Tensor& get_weight(int32_t idx) const;

    void to_cuda() override;

    void set_weight(int32_t idx, const mem::Tensor& weight) override;

    void set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;
        
protected:
    std::vector<mem::Tensor> weights_;
};
};
#endif