#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace yaal {

template<typename Derived>
class FastEventParser {
public:
    __attribute__((flatten, hot))
    void parse(const Buffer& buf) {
        const char* data = buf.start();
        const size_t len = buf.len();

        derived().on_bod();

        if (len == 0) {
            derived().on_eod();
            return;
        }

        const __m256i newline_vec = _mm256_set1_epi8('\n');
        const __m256i space_vec = _mm256_set1_epi8(' ');

        size_t pos = 0;
        bool need_bos = true;
        uint64_t local_eol = 0;
        uint64_t local_bos = 0;

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

            uint64_t ns_mask = ~(sp_mask | nl_mask);

            // Detect BOS events
            uint64_t bos_mask = detect_bos(nl_mask, ns_mask, need_bos);

            // Accumulate locally
            local_eol += _mm_popcnt_u64(nl_mask);
            local_bos += _mm_popcnt_u64(bos_mask);

            pos += 64;
        }

        // 32-byte remainder
        if (pos + 32 <= len) {
            __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            uint32_t nl_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, newline_vec)));
            uint32_t sp_mask = static_cast<uint32_t>(_mm256_movemask_epi8(
                _mm256_cmpeq_epi8(chunk, space_vec)));
            uint32_t ns_mask = ~(sp_mask | nl_mask);

            uint32_t bos_mask = detect_bos_32(nl_mask, ns_mask, need_bos);

            local_eol += _mm_popcnt_u32(nl_mask);
            local_bos += _mm_popcnt_u32(bos_mask);

            pos += 32;
        }

        // Fire accumulated events to derived
        derived().on_eol(local_eol);
        derived().on_bos(local_bos);

        // Scalar tail
        while (pos < len) {
            char c = data[pos];
            if (c == '\n') {
                derived().on_eol_single();
                need_bos = true;
            } else if (c != ' ' && need_bos) {
                derived().on_bos_single();
                need_bos = false;
            }
            pos++;
        }

        derived().on_eod();
    }

private:
    __attribute__((always_inline))
    Derived& derived() { return static_cast<Derived&>(*this); }

    __attribute__((always_inline, hot))
    uint64_t detect_bos(uint64_t nl_mask, uint64_t ns_mask, bool& need_bos) {
        uint64_t start = (nl_mask << 1) | 1ULL;
        uint64_t seg_or = ns_mask | (((start & ~ns_mask) - ns_mask) & ~start);
        uint64_t prev_seg = (seg_or << 1) & ~start;
        uint64_t bos_mask = ns_mask & ~prev_seg;

        // Clear first segment if !need_bos
        uint64_t first_nl_pos = _tzcnt_u64(nl_mask);
        uint64_t first_seg_mask = _bzhi_u64(~0ULL, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint64_t>(!need_bos));

        // Update need_bos for next chunk
        uint64_t last_nl_pos = 63 - _lzcnt_u64(nl_mask);
        uint64_t after_last = _bzhi_u64(ns_mask, last_nl_pos + 1) ^ ns_mask;
        uint64_t has_nl = -static_cast<uint64_t>(nl_mask != 0);
        uint64_t no_nl_result = need_bos & (ns_mask == 0);
        need_bos = (has_nl & (after_last == 0)) | (~has_nl & no_nl_result);

        return bos_mask;
    }

    __attribute__((always_inline, hot))
    uint32_t detect_bos_32(uint32_t nl_mask, uint32_t ns_mask, bool& need_bos) {
        uint32_t start = (nl_mask << 1) | 1u;
        uint32_t seg_or = ns_mask | (((start & ~ns_mask) - ns_mask) & ~start);
        uint32_t prev_seg = (seg_or << 1) & ~start;
        uint32_t bos_mask = ns_mask & ~prev_seg;

        uint32_t first_nl_pos = _tzcnt_u32(nl_mask);
        uint32_t first_seg_mask = _bzhi_u32(~0u, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint32_t>(!need_bos));

        uint32_t last_nl_pos = 31 - _lzcnt_u32(nl_mask);
        uint32_t after_last = _bzhi_u32(ns_mask, last_nl_pos + 1) ^ ns_mask;
        uint32_t has_nl = -static_cast<uint32_t>(nl_mask != 0);
        uint32_t no_nl_result = need_bos & (ns_mask == 0);
        need_bos = (has_nl & (after_last == 0)) | (~has_nl & no_nl_result);

        return bos_mask;
    }
};

// Counting implementation using CRTP
class FastCountingParserV2 : public FastEventParser<FastCountingParserV2> {
public:
    struct Counts {
        uint64_t bod = 0;
        uint64_t bos = 0;
        uint64_t eol = 0;
        uint64_t eod = 0;
    };

    __attribute__((always_inline)) void on_bod() { counts_.bod++; }
    __attribute__((always_inline)) void on_eod() { counts_.eod++; }

    __attribute__((always_inline)) void on_eol(uint64_t count) {
        counts_.eol += count;
    }

    __attribute__((always_inline)) void on_bos(uint64_t count) {
        counts_.bos += count;
    }

    __attribute__((always_inline)) void on_eol_single() { counts_.eol++; }
    __attribute__((always_inline)) void on_bos_single() { counts_.bos++; }

    const Counts& counts() const { return counts_; }
    void reset() { counts_ = Counts{}; }

private:
    Counts counts_;
};

} // namespace yaal
