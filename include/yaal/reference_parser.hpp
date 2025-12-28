#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace yaal {

class ReferenceParser {
public:
    struct Counts {
        uint64_t bod = 0;
        uint64_t bos = 0;
        uint64_t eol = 0;
        uint64_t eod = 0;
    };

    ReferenceParser() = default;

    __attribute__((flatten, hot, noinline))
    void parse(const Buffer& buf) {
        const char* data = buf.start();
        const size_t len = buf.len();

        counts_.bod++;

        if (len == 0) {
            counts_.eod++;
            return;
        }

        const __m256i newline_vec = _mm256_set1_epi8('\n');
        const __m256i space_vec = _mm256_set1_epi8(' ');

        size_t pos = 0;
        uint8_t need_bos = true;

        // Local accumulators to avoid memory stores in hot loop
        uint64_t local_eol = 0;
        uint64_t local_bos = 0;

        // Main loop: 192 bytes at a time
        //
        // This is unrolled 3x to minimize the latency impact of the add-with-carry
        // chains in count_bos_fast.
        while (pos + 192 <= len) {
            __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 32));
            __m256i c2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 64));
            __m256i c3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 96));
            __m256i c4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 128));
            __m256i c5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 160));

            uint64_t nl_mask_0 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, newline_vec)));
            uint64_t nl_mask_1 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c3, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, newline_vec)));
            uint64_t nl_mask_2 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c5, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c4, newline_vec)));

            uint64_t sp_mask_0 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, space_vec)));
            uint64_t sp_mask_1 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c3, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, space_vec)));
            uint64_t sp_mask_2 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c5, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c4, space_vec)));

            local_eol += _mm_popcnt_u64(nl_mask_0);
            local_eol += _mm_popcnt_u64(nl_mask_1);
            local_eol += _mm_popcnt_u64(nl_mask_2);
            local_bos += count_bos_fast(nl_mask_0, sp_mask_0 | nl_mask_0, need_bos, need_bos);
            local_bos += count_bos_fast(nl_mask_1, sp_mask_1 | nl_mask_1, need_bos, need_bos);
            local_bos += count_bos_fast(nl_mask_2, sp_mask_2 | nl_mask_2, need_bos, need_bos);

            pos += 192;
        }

        // 32-byte remainders
        while (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            uint32_t nl_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, newline_vec)));
            uint32_t sp_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, space_vec)));

            local_eol += _mm_popcnt_u32(nl_mask);
            local_bos += count_bos_fast_32(nl_mask, sp_mask | nl_mask, need_bos, need_bos);

            pos += 32;
        }

        // Store accumulated counts
        counts_.eol += local_eol;
        counts_.bos += local_bos;

        // Scalar tail
        while (pos < len) {
            char c = data[pos];
            if (c == '\n') {
                counts_.eol++;
                need_bos = true;
            } else if (c != ' ' && need_bos) {
                counts_.bos++;
                need_bos = false;
            }
            pos++;
        }

        counts_.eod++;
    }

    const Counts& counts() const { return counts_; }
    void reset() { counts_ = Counts{}; }

private:
    Counts counts_;

    __attribute__((always_inline, hot))
    static uint64_t count_bos_fast(uint64_t nl_mask, uint64_t ws_mask, uint8_t need_bos_in, uint8_t& need_bos_out) {
        unsigned long long sum;
        need_bos_out = _addcarry_u64(need_bos_in, ws_mask, nl_mask, &sum);
        uint64_t bos_mask = sum & ~ws_mask;
        return _mm_popcnt_u64(bos_mask);
    }

    __attribute__((always_inline, hot))
    static uint32_t count_bos_fast_32(uint32_t nl_mask, uint32_t ws_mask, uint8_t need_bos_in, uint8_t& need_bos_out) {
        unsigned sum;
        need_bos_out = _addcarry_u32(need_bos_in, ws_mask, nl_mask, &sum);
        uint32_t bos_mask = sum & ~ws_mask;
        return _mm_popcnt_u32(bos_mask);
    }
};

} // namespace yaal
