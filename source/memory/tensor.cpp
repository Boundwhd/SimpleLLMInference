#include "tensor.h"

namespace mem{
    template<typename T, typename Tp>
    static size_t reduce_dimention(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    Tensor::Tensor(std::vector<int32_t> dims, bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void* ptr) 
        : dims_(std::move(dims)) {
        
        size_ = reduce_dimention(dims_.begin(), dims_.end(), 1);
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, need_alloc, ptr);
        }
    }

    bool Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc) {
        if (!allocator) {
            std::cout<< " Allocator is Empty! " << std::endl;
            return false;
        }

        size_t byte_size = this->byte_size();
        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
                return true;
            }
        }
        buffer_ = std::make_shared<Buffer>(byte_size, allocator, nullptr);
        if (!buffer_->ptr()) {
            LOG("The memory allocated is a null pointer!");
            return false;
        }
        return true;
    }

    void Tensor::init_buffer(std::shared_ptr<DeviceAllocator> alloc, bool need_alloc, void* ptr) {
        if (!alloc && !need_alloc) {
            std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(DaraTypeSize * size_, nullptr, ptr, true);
            this->buffer_ = buffer;
        } else {
            allocate(alloc, true);
        }
    }

    size_t Tensor::byte_size() const {
        return DaraTypeSize * this->size_;
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_) {
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }

    void Tensor::to_cpu() {
        if (buffer_ == nullptr) {
            LOG(" No buffer in Tensor! ");
        }
        
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(" The device type of the tensor is unknown. ");
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            std::cout << " The device type of the tensor is already cpu. " << std::endl;
        } else {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        }
    }

    void Tensor::to_cuda() {
        if (buffer_ == nullptr) {
            LOG(" No buffer in Tensor! ");
        }
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(" The device type of the tensor is unknown. ");
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            std::cout << " The device type of the tensor is already cuda. " << std::endl;
        } else {
            size_t byte_size = this->byte_size();
            auto cu_alloc = CUDADeviceAllocatorFactory::get_instance();
            auto cu_buffer = std::make_shared<Buffer>(byte_size, cu_alloc);
            cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA);
            this->buffer_ = cu_buffer;
        }
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    void Tensor::reshape(const std::vector<int32_t>& dims) {
        size_t size = reduce_dimention(dims.begin(), dims.end(), 1);
        if (!buffer_) {
            this->dims_ = dims;
            this->size_ = size;
            return;
        }
        if (size > size_) {
            auto new_buffer = std::make_shared<Buffer>(size * DaraTypeSize, buffer_->allocator());
            new_buffer->copy_from(buffer_.get());
            this->buffer_ = new_buffer;
        }
        this->dims_ = dims;
        this->size_ = size;
    }

    std::shared_ptr<Buffer> Tensor::get_buffer() const {
        return buffer_;
    }

    size_t Tensor::size() const {
        return this->size_;
    }

    int32_t Tensor::dims_size() const {
        return static_cast<int32_t>(dims_.size());
    }

    int32_t Tensor::get_dim(int32_t idx) const {
        if (idx < 0 || idx >= this->dims_size()) {
            LOG("idx is wrong!");
        }
        return this->dims_.at(idx);
    }

    const std::vector<int32_t>& Tensor::dims() const {
        return this->dims_;
    }

    bool Tensor::assign(std::shared_ptr<Buffer> buffer) {
        if (!buffer) {
            std::cout << "The buffer parameter in the assign function is null pointer!" << std::endl;
            return false;
        }

        if (buffer_) {
            if (buffer_->device_type() != buffer->device_type()) {
                std::cout << "The device type of the new buffer is different from the original one." << std::endl;
                return false;
            }
        }
        size_t byte_size = this->byte_size();
            if (byte_size > buffer->byte_size()) {
                std::cout << "The size of buffer is too small for the tensor!" << std::endl;
                return false;
            }
        buffer_ = buffer;
        return true;
    }

    void Tensor::reset(const std::vector<int32_t>& dims) {
        this->dims_ = dims;
        this->size_ = reduce_dimention(dims.begin(), dims.end(), 1);
        this->buffer_ = nullptr;
    }

    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = reduce_dimention(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<Buffer>(byte_size, allocator);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }
};