#include "weight_loader.h"
#include <sys/mman.h>
#include <unistd.h>

namespace model {
    RawModelData::~RawModelData() {
        if (data != nullptr && data != MAP_FAILED) {
            munmap(data, file_size);
            data = nullptr;
        }
        if (fd != -1) {
            close(fd);
            fd = -1;
        }
    }
    
    const void* RawModelDataFp32::weight(size_t offset) const {
        return static_cast<float*>(weight_data) + offset;
    }
}