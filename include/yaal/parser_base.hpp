#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace yaal {

template<typename Derived>
class ParserBase {
public:
    __attribute__((flatten, hot))
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
        bool need_bos = true;

        while (pos + 64 <= len) {
            __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 32));

            uint64_t nl_mask = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, newline_vec)));

            uint64_t sp_mask = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, space_vec)));

            uint64_t ns_mask = ~(sp_mask | nl_mask);

            // Fast path: no newlines in chunk (common case)
            if (__builtin_expect(nl_mask == 0, 1)) {
                if (need_bos & (ns_mask != 0)) {
                    derived().on_bos(pos + _tzcnt_u64(ns_mask));
                    need_bos = false;
                }
                pos += 64;
                continue;
            }

            // Handle bos before first newline if needed
            uint64_t first_nl_pos = _tzcnt_u64(nl_mask);
            if (need_bos) {
                uint64_t ns_before = ns_mask & ((1ULL << first_nl_pos) - 1);
                if (ns_before) {
                    derived().on_bos(pos + _tzcnt_u64(ns_before));
                }
            }

            // Process all newlines
            do {
                uint64_t nl_pos = _tzcnt_u64(nl_mask);
                derived().on_eol(pos + nl_pos);
                nl_mask &= nl_mask - 1;

                // Find bos after this newline
                uint64_t after_mask = (nl_pos < 63) ? (~0ULL << (nl_pos + 1)) : 0;
                uint64_t next_nl_pos = nl_mask ? _tzcnt_u64(nl_mask) : 64;
                uint64_t before_next = (next_nl_pos < 64) ? ((1ULL << next_nl_pos) - 1) : ~0ULL;
                uint64_t search = ns_mask & after_mask & before_next;

                if (search) {
                    derived().on_bos(pos + _tzcnt_u64(search));
                    need_bos = false;
                } else {
                    need_bos = true;
                }
            } while (nl_mask);

            pos += 64;
        }

        // Handle remaining 32-byte chunk
        if (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            uint32_t nl_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, newline_vec)));
            uint32_t sp_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, space_vec)));
            uint32_t ns_mask = ~(sp_mask | nl_mask);

            if (nl_mask == 0) {
                if (need_bos && ns_mask) {
                    derived().on_bos(pos + _tzcnt_u32(ns_mask));
                    need_bos = false;
                }
            } else {
                uint32_t first_nl_pos = _tzcnt_u32(nl_mask);
                if (need_bos) {
                    uint32_t ns_before = ns_mask & ((1u << first_nl_pos) - 1);
                    if (ns_before) derived().on_bos(pos + _tzcnt_u32(ns_before));
                }

                do {
                    uint32_t nl_pos = _tzcnt_u32(nl_mask);
                    derived().on_eol(pos + nl_pos);
                    nl_mask &= nl_mask - 1;

                    uint32_t after_mask = (nl_pos < 31) ? (~0u << (nl_pos + 1)) : 0;
                    uint32_t next_nl_pos = nl_mask ? _tzcnt_u32(nl_mask) : 32;
                    uint32_t before_next = (next_nl_pos < 32) ? ((1u << next_nl_pos) - 1) : ~0u;
                    uint32_t search = ns_mask & after_mask & before_next;

                    if (search) {
                        derived().on_bos(pos + _tzcnt_u32(search));
                        need_bos = false;
                    } else {
                        need_bos = true;
                    }
                } while (nl_mask);
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
};

} // namespace yaal
