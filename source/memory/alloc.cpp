#include "alloc.h"

namespace mem {
    // 设备管理基类
    base::DeviceType DeviceAllocator::device_type() {
        return device_type_;
    }

    void DeviceAllocator::memcpy(const void* src_ptr, void* dst_ptr, size_t byte_size, base::MemcpyKind memcpy_kind) const {
        if (!src_ptr || !dst_ptr) {
            LOG(" ERROR! Ptr is empty! ");
        }   
        if (memcpy_kind == base::MemcpyKind::kMemcpyCPU2CPU) {
            std::memcpy(dst_ptr, src_ptr, byte_size);
        } else if (memcpy_kind == base::MemcpyKind::kMemcpyCPU2CUDA) {
            cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
        } else if (memcpy_kind == base::MemcpyKind::kMemcpyCUDA2CPU) {
            cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
        } else {
            cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
        }
    }

    void DeviceAllocator::memset_zero(void* ptr, size_t byte_size) {
        if (!ptr) {
            LOG(" ERROR! Ptr is Empty! ");
        }
        if (device_type_ == base::DeviceType::kDeviceUnknown) {
            LOG(" ERROR! Device Type Unknown! ");
        } else if (device_type_ == base::DeviceType::kDeviceCPU) {
            std::memset(ptr, 0, byte_size);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }
    }

    // CPU 设备管理类
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(base::DeviceType::kDeviceCPU) {}

    void* CPUDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }
        void* data = malloc(byte_size);
        return data;
    }

    void CPUDeviceAllocator::release(void* ptr) const {
        if (ptr) {
            free(ptr);
        }
    }

    //CUDA 设备管理类
    // ---- CUDA 内存池类
    CUDAMemBlockPool::CUDAMemBlockPool() : blocks(CompareBlocks) {}

    void CUDAMemBlockPool::insert(CUDAMemBlock* block) {
        blocks.insert(block);
    }

    CUDAMemBlock* CUDAMemBlockPool::find_best_fit(size_t size) {
        CUDAMemBlock key(size, nullptr);
        auto it = blocks.lower_bound(&key);
        return (it != blocks.end() && (*it)->size >= size) ? *it : nullptr;
    }

    void CUDAMemBlockPool::erase(CUDAMemBlock* block) {
        blocks.erase(block);
    }

    // ---- 内存管理算法
    void* CUDADeviceAllocator::malloc_impl (size_t size) const {
        std::lock_guard<std::mutex> lock(mutex);

        // 确定内存池
        CUDAMemBlockPool& pool = (size <= kSmallSize) ? small_blocks : large_blocks;
        size_t alloc_size = 0;
        
        
        if (size <= kSmallSize) {
            alloc_size = kSmallBuffer;
        } else if (size <= 10 * kSmallSize) {
            alloc_size = kLargeBuffer;
        } else {
            alloc_size = ((size + kSmallBuffer - 1) / kSmallBuffer) * kSmallBuffer;
        }


        if (CUDAMemBlock* block = pool.find_best_fit(size)) {
            pool.erase(block);
            return split_block(block, size);
        }

        return allocate_new_block(size, alloc_size);
    }

    void* CUDADeviceAllocator::split_block(CUDAMemBlock* block, size_t size) const {
        if (block->size - size >= kMinBlockSize) {
            CUDAMemBlock* remaining = new CUDAMemBlock(block->size - size, static_cast<char*>(block->ptr) + size);

            remaining->prev = block;
            remaining->next = block->next;
            if (block->next) {
                block->next->prev = remaining;
            }
            block->next = remaining;
            block->size = size;

            (remaining->size <= kSmallSize ? small_blocks : large_blocks).insert(remaining);
        }

        block->allocated = true;
        allocated_blocks[block->ptr] = block;
        return block->ptr;
    }

    void* CUDADeviceAllocator::allocate_new_block(size_t req_size, size_t alloc_size) const {
        void* ptr;
        cudaError_t err = cudaMalloc(&ptr, alloc_size);

        if (err != cudaSuccess) {
            cudaGetLastError();
            release_cached_memory();
            err = cudaMalloc(&ptr, alloc_size);
            if (err != cudaSuccess) throw std::bad_alloc();
        }
        
        CUDAMemBlock* block = new CUDAMemBlock(alloc_size, ptr);
        return split_block(block, req_size);
    }

    void CUDADeviceAllocator::release_cached_memory() const {
        auto release_pool = [](CUDAMemBlockPool& pool) {
            std::vector<CUDAMemBlock*> to_delete;
            for (CUDAMemBlock* block : pool.blocks) {
                if (!block->allocated) {
                    cudaFree(block->ptr);
                    to_delete.push_back(block);
                }
            }
            for (CUDAMemBlock* block : to_delete) {
                pool.erase(block);
                delete block;
            }
        };
    
        release_pool(small_blocks);
        release_pool(large_blocks);
    }

    void CUDADeviceAllocator::merge_blocks(CUDAMemBlock* block) const {
        CUDAMemBlockPool& pool = (block->size <= kSmallSize) ? small_blocks : large_blocks;

        while (block->prev && !block->prev->allocated) {
            CUDAMemBlock* prev = block->prev;
            pool.erase(prev);  
            pool.erase(block);
            prev->size += block->size;
            prev->next = block->next;
            if (block->next) {
                block->next->prev = prev;
            }
            delete block;
            block = prev;
        }

        while (block->next && !block->next->allocated) {
            CUDAMemBlock* next = block->next;
            pool.erase(block); 
            pool.erase(next);
            block->size += next->size;
            block->next = next->next;
            if (next->next) {
                next->next->prev = block;
            }
            delete next;
        }

        (block->size <= kSmallSize ? small_blocks : large_blocks).insert(block);
    }

    CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(base::DeviceType::kDeviceCUDA) {}

    void* CUDADeviceAllocator::allocate(size_t byte_size) const {
        size_t size = round_size(byte_size);
        return malloc_impl(size);
    }

    void CUDADeviceAllocator::release(void* ptr) const {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = allocated_blocks.find(ptr);
        if (it == allocated_blocks.end()) return;
        
        CUDAMemBlock* block = it->second;
        block->allocated = false;
        allocated_blocks.erase(it);
        
        merge_blocks(block);
    }
    std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
    std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
};