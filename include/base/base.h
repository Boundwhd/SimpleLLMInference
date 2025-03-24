#ifndef _BASE_WHD_H_
#define _BASE_WHD_H_
#include <iostream>
#include <string>

inline void log_message(const std::string& message, const std::string& file, int line) {
    std::cout << "file: " << file << " line: " << line << " - " << message << std::endl;
    std::exit(EXIT_FAILURE);
}
#define LOG(message) log_message(message, __FILE__, __LINE__)

namespace base {
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

enum DeviceType {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};
};

#endif 