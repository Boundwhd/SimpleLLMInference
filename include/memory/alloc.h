#ifndef _ALLOC_WHD_H_
#define _ALLOC_WHD_H_
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <cuda_runtime_api.h>
#include "base.h"

#define DaraTypeSize 4
/**
 * mem 命名空间：
 * 1. CPU 设备管理，内存申请，释放
 * 2. GPU 设备管理，内存申请，释放
 */

constexpr size_t kMinBlockSize = 512;
constexpr size_t kSmallSize = 1048576;
constexpr size_t kSmallBuffer = 2097152;
constexpr size_t kLargeBuffer = 20971520;

namespace mem {

inline size_t round_size(size_t size) {
    if (size < kMinBlockSize) return kMinBlockSize;
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

// 设备管理类
class DeviceAllocator {
public:
    explicit DeviceAllocator(base::DeviceType device_type) : device_type_(device_type) {}

    virtual base::DeviceType device_type();

    virtual void release(void* ptr) const = 0;

    virtual void* allocate(size_t byte_size) const = 0;

    void memcpy(const void* src_ptr, void* dst_ptr, size_t byte_size, base::MemcpyKind memcpy_kind) const;

    virtual void memset_zero(void* ptr, size_t byte_size);
private:
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

class CPUDeviceAllocator : public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

struct CUDAMemBlock {
    void* ptr;
    size_t size;
    bool allocated;
    CUDAMemBlock* prev;
    CUDAMemBlock* next;

    CUDAMemBlock(size_t size, void* ptr) : ptr(ptr), size(size), allocated(false), prev(nullptr), next(nullptr) {}
};

class CUDAMemBlockPool {
public:
    std::set<CUDAMemBlock*, bool(*)(CUDAMemBlock*, CUDAMemBlock*)> blocks;

    static bool CompareBlocks(CUDAMemBlock* a, CUDAMemBlock* b) {
        if (a->size != b->size) {
            return a->size < b->size;  
        }
        return a->ptr < b->ptr;
    }

    CUDAMemBlockPool();

    void insert(CUDAMemBlock* block);

    CUDAMemBlock* find_best_fit(size_t size);

    void erase(CUDAMemBlock* block);
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;

private:
    mutable std::mutex mutex;
    mutable CUDAMemBlockPool small_blocks;
    mutable CUDAMemBlockPool large_blocks;
    mutable std::unordered_map<void*, CUDAMemBlock*> allocated_blocks;

    void* malloc_impl(size_t size) const ;

    void* split_block(CUDAMemBlock* block, size_t size) const ;

    void* allocate_new_block(size_t req_size, size_t alloc_size) const ;
    
    void release_cached_memory() const ;

    void merge_blocks(CUDAMemBlock* block) const ;
};

class CPUDeviceAllocatorFactory {
public:
     static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
        instance = std::make_shared<CPUDeviceAllocator>();
       }
    return instance;
    }
   
private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
        instance = std::make_shared<CUDADeviceAllocator>();
    }
       return instance;
    }
   
private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};
};

#endif