#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace yaal {

template<typename Derived>
class ParserBase {
public:
    __attribute__((flatten, hot, noinline))
    void parse(const Buffer& buf) {
        const char* data = buf.start();
        const size_t len = buf.len();

        derived().on_bod(0);

        if (len == 0) {
            derived().on_eod(0);
            return;
        }

        const __m256i newline_vec = _mm256_set1_epi8('\n');
        const __m256i space_vec = _mm256_set1_epi8(' ');

        size_t pos = 0;
        uint8_t need_bos = true;

        // Main loop: 64 bytes at a time
        while (pos + 64 <= len) {
            __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 32));

            uint64_t nl_mask = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, newline_vec)));

            uint64_t sp_mask = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, space_vec)));

            uint64_t ws_mask = sp_mask | nl_mask;

            // Compute BOS mask using add-with-carry
            uint64_t bos_mask = compute_bos_mask(nl_mask, ws_mask, need_bos, need_bos);

            // Emit EOL events
            uint64_t nl_tmp = nl_mask;
            while (nl_tmp) {
                uint64_t nl_pos = _tzcnt_u64(nl_tmp);
                derived().on_eol(pos + nl_pos);
                nl_tmp &= nl_tmp - 1;
            }

            // Emit BOS events
            while (bos_mask) {
                uint64_t bos_pos = _tzcnt_u64(bos_mask);
                derived().on_bos(pos + bos_pos);
                bos_mask &= bos_mask - 1;
            }

            pos += 64;
        }

        // Handle remaining 32-byte chunk
        if (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            uint32_t nl_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, newline_vec)));
            uint32_t sp_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, space_vec)));

            uint32_t ws_mask = sp_mask | nl_mask;

            // Compute BOS mask using add-with-carry
            uint32_t bos_mask = compute_bos_mask_32(nl_mask, ws_mask, need_bos, need_bos);

            // Emit EOL events
            uint32_t nl_tmp = nl_mask;
            while (nl_tmp) {
                uint32_t nl_pos = _tzcnt_u32(nl_tmp);
                derived().on_eol(pos + nl_pos);
                nl_tmp &= nl_tmp - 1;
            }

            // Emit BOS events
            while (bos_mask) {
                uint32_t bos_pos = _tzcnt_u32(bos_mask);
                derived().on_bos(pos + bos_pos);
                bos_mask &= bos_mask - 1;
            }

            pos += 32;
        }

        // Scalar tail
        while (pos < len) {
            char c = data[pos];
            if (c == '\n') {
                derived().on_eol(pos);
                need_bos = true;
            } else if (c != ' ' && need_bos) {
                derived().on_bos(pos);
                need_bos = false;
            }
            pos++;
        }

        derived().on_eod(len);
    }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }

    __attribute__((always_inline, hot))
    static uint64_t compute_bos_mask(uint64_t nl_mask, uint64_t ws_mask, uint8_t need_bos_in, uint8_t& need_bos_out) {
        unsigned long long sum;
        need_bos_out = _addcarry_u64(need_bos_in, ws_mask, nl_mask, &sum);
        return sum & ~ws_mask;
    }

    __attribute__((always_inline, hot))
    static uint32_t compute_bos_mask_32(uint32_t nl_mask, uint32_t ws_mask, uint8_t need_bos_in, uint8_t& need_bos_out) {
        unsigned sum;
        need_bos_out = _addcarry_u32(need_bos_in, ws_mask, nl_mask, &sum);
        return sum & ~ws_mask;
    }
};

} // namespace yaal
