#pragma once

#include "parser_base.hpp"
#include <cstddef>
#include <cstdint>

namespace yaal {

class CountingParser : public ParserBase<CountingParser> {
public:
    struct Counts {
        uint64_t bod = 0;
        uint64_t bos = 0;
        uint64_t eol = 0;
        uint64_t eod = 0;
    };

    CountingParser() = default;

    __attribute__((always_inline)) void on_bod(size_t) { counts_.bod++; }
    __attribute__((always_inline)) void on_bos(size_t) { counts_.bos++; }
    __attribute__((always_inline)) void on_eol(size_t) { counts_.eol++; }
    __attribute__((always_inline)) void on_eod(size_t) { counts_.eod++; }

    const Counts& counts() const { return counts_; }
    void reset() { counts_ = Counts{}; }

private:
    Counts counts_;
};

} // namespace yaal
