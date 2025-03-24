#ifndef _BUFFER_WHD_H_
#define _BUFFER_WHD_H_
#include "alloc.h"
namespace mem {

class NoCopyable {
protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer>{
private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;

public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
        void* ptr = nullptr, bool use_external = false);
    
    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer& buffer) const;

    void copy_from(const Buffer* buffer) const;

    void* ptr();

    const void* ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();
    
    bool is_external() const;
};
};
#endif