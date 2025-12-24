#pragma once

#include <cstddef>

namespace yaal {

class Buffer {
public:
    Buffer(const char* data, size_t len) : data_(data), len_(len) {}

    const char* start() const { return data_; }
    size_t len() const { return len_; }

private:
    const char* data_;
    size_t len_;
};

} // namespace yaal
