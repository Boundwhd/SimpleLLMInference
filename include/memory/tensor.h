#ifndef _TENSOR_WHD_H_
#define _TENSOR_WHD_H_
#include <driver_types.h>
#include "alloc.h"
#include "buffer.h"
#include <numeric>

namespace mem {
class Tensor {
public:
    explicit Tensor() = default;

    explicit Tensor(std::vector<int32_t> dims, bool need_alloc = false,
        std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    
    void to_cpu();

    void to_cuda();

    bool is_empty() const;

    void init_buffer(std::shared_ptr<DeviceAllocator> alloc, bool need_alloc, void* ptr);

    template <typename T>
    T* ptr();
    
    template <typename T>
    const T* ptr() const;
    
    void reshape(const std::vector<int32_t>& dims);

    std::shared_ptr<Buffer> get_buffer() const;

    size_t size() const;

    size_t byte_size() const;

    int32_t dims_size() const;

    int32_t get_dim(int32_t idx) const;

    const std::vector<int32_t>& dims() const;

    std::vector<size_t> strides() const;

    bool assign(std::shared_ptr<Buffer> buffer);

    void reset(const std::vector<int32_t>& dims);

    void set_device_type(base::DeviceType device_type) const;

    base::DeviceType device_type() const;

    bool allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc = false);

    template <typename T>
    T* ptr(int64_t index);

    template <typename T>
    T& index(int64_t offset);

    Tensor clone() const;
private:
    size_t size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<Buffer> buffer_;
};

template <typename T>
T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return const_cast<const T*>(reinterpret_cast<T*>(buffer_->ptr()));
}

template <typename T>
T* Tensor::ptr(int64_t index) {
    if (!buffer_ || !buffer_->ptr()) {
        LOG("ERROR Get Ptr!");
    }
    return const_cast<T*>(reinterpret_cast<const T*>(buffer_->ptr())) + index;
}

template <typename T>
T& Tensor::index(int64_t offset) {
    if (offset < 0 || offset >= this->size()) {
        LOG("ERROR Index!");
    } 
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}
};
#endif