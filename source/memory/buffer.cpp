#include "buffer.h"

namespace mem {
    Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr, bool use_external)
        : byte_size_(byte_size), allocator_(allocator), ptr_(ptr), use_external_(use_external) {

        if (!ptr_ && allocator_) {
            device_type_ = allocator_->device_type();
            use_external_ = false;
            ptr_ = allocator_->allocate(byte_size);
        }
    }

    Buffer::~Buffer() {
        if (!use_external_) {
            if (ptr_ && allocator_) {
                allocator_->release(ptr_);
                ptr_ = nullptr;
            }
        }
    }

    void* Buffer::ptr() {
        return ptr_;
    }

    const void* Buffer::ptr() const {
        return ptr_;
    }

    size_t Buffer::byte_size() const {
        return byte_size_;
    }

    bool Buffer::allocate() {
        if (allocator_ && byte_size_ != 0) {
            use_external_ = false;
            ptr_ = allocator_->allocate(byte_size_);
            if (!ptr_) {
                return false;
            } else {
                return true;
            }
        } else {
            return false;
        }
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
        return allocator_;
    }

    void Buffer::copy_from(const Buffer& buffer) const {
        if (allocator_ == nullptr) {
            LOG(" ERROR! alloccator is empty while coping! ");
        }
        if (buffer.ptr_ == nullptr) {
            LOG(" ERROR! ptr is empty while coping! ");
        }

        size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
        const base::DeviceType& buffer_device = buffer.device_type();
        const base::DeviceType& current_device = this->device_type();

        if (buffer_device == base::DeviceType::kDeviceUnknown || 
            current_device == base::DeviceType::kDeviceUnknown) {
                LOG(" ERROR! DeviceType is unknown! ");
        }

        if (buffer_device == base::DeviceType::kDeviceCPU &&
            current_device == base::DeviceType::kDeviceCPU) {
                return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, base::MemcpyKind::kMemcpyCPU2CPU);
        } else if (buffer_device == base::DeviceType::kDeviceCUDA &&
                   current_device == base::DeviceType::kDeviceCPU) {
                return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, base::MemcpyKind::kMemcpyCUDA2CPU);
        } else if (buffer_device == base::DeviceType::kDeviceCPU &&
                   current_device == base::DeviceType::kDeviceCUDA) {
                return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, base::MemcpyKind::kMemcpyCPU2CUDA);
        } else {
                return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, base::MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    void Buffer::copy_from(const Buffer* buffer) const {
        if (allocator_ == nullptr) {
            LOG(" ERROR! alloccator is empty while coping! ");
        }
        if (buffer->ptr_ == nullptr) {
            LOG(" ERROR! ptr is empty while coping! ");
        }
      
        size_t dest_size = byte_size_;
        size_t src_size = buffer->byte_size_;
        size_t byte_size = src_size < dest_size ? src_size : dest_size;
      
        const base::DeviceType& buffer_device = buffer->device_type();
        const base::DeviceType& current_device = this->device_type();
      
        if (buffer_device == base::DeviceType::kDeviceCPU &&
            current_device == base::DeviceType::kDeviceCPU) {
          return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, base::MemcpyKind::kMemcpyCPU2CPU);
        } else if (buffer_device == base::DeviceType::kDeviceCUDA &&
            current_device == base::DeviceType::kDeviceCPU) {
          return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, base::MemcpyKind::kMemcpyCUDA2CPU);
        } else if (buffer_device == base::DeviceType::kDeviceCPU &&
            current_device == base::DeviceType::kDeviceCUDA) {
          return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, base::MemcpyKind::kMemcpyCPU2CUDA);
        } else {
          return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, base::MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    base::DeviceType Buffer::device_type() const {
        return device_type_;
    }

    void Buffer::set_device_type(base::DeviceType device_type) {
        device_type_ = device_type;
    }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
        return shared_from_this();
    }

    bool Buffer::is_external() const {
        return this->use_external_;
    }
}