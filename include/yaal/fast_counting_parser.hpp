#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace yaal {

class FastCountingParser {
public:
    struct Counts {
        uint64_t bod = 0;
        uint64_t bos = 0;
        uint64_t eol = 0;
        uint64_t eod = 0;
    };

    FastCountingParser() = default;

    __attribute__((flatten, hot))
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
        bool need_bos = true;

        // Local accumulators to avoid memory stores in hot loop
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

            local_eol += _mm_popcnt_u64(nl_mask);
            local_bos += count_bos_fast(nl_mask, ns_mask, need_bos, need_bos);

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

            local_eol += _mm_popcnt_u32(nl_mask);
            local_bos += count_bos_fast_32(nl_mask, ns_mask, need_bos, need_bos);

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

    // O(1) segmented OR-scan with SBB for cross-chunk state
    // borrow=0 means "need BOS", borrow=1 means "don't need"
    __attribute__((always_inline, hot))
    static uint64_t count_bos_sbb(uint64_t nl_mask, uint64_t ns_mask, unsigned char& borrow) {
        unsigned long long start = (nl_mask << 1) | 1ULL;
        unsigned long long a = start & ~ns_mask;
        unsigned long long b = ns_mask;
        unsigned long long diff;

        // Save need_first BEFORE modifying borrow
        bool need_first = !borrow;

        // Segmented OR-scan using SBB
        borrow = _subborrow_u64(borrow, a, b, &diff);
        unsigned long long seg_or = ns_mask | (diff & ~start);

        // Correct borrow for newline at position 63
        borrow &= ~static_cast<unsigned char>(nl_mask >> 63);

        // BOS = first ns in each segment
        unsigned long long prev_seg = (seg_or << 1) & ~start;
        unsigned long long bos_mask = ns_mask & ~prev_seg;

        // Clear first segment BOS if !need_first
        uint64_t first_nl_pos = _tzcnt_u64(nl_mask);
        uint64_t first_seg_mask = _bzhi_u64(~0ULL, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint64_t>(!need_first));

        return _mm_popcnt_u64(bos_mask);
    }

    __attribute__((always_inline, hot))
    static uint32_t count_bos_sbb_32(uint32_t nl_mask, uint32_t ns_mask, unsigned char& borrow) {
        unsigned int start = (nl_mask << 1) | 1u;
        unsigned int a = start & ~ns_mask;
        unsigned int b = ns_mask;
        unsigned int diff;

        bool need_first = !borrow;

        borrow = _subborrow_u32(borrow, a, b, &diff);
        unsigned int seg_or = ns_mask | (diff & ~start);
        borrow &= ~static_cast<unsigned char>(nl_mask >> 31);

        unsigned int prev_seg = (seg_or << 1) & ~start;
        unsigned int bos_mask = ns_mask & ~prev_seg;

        uint32_t first_nl_pos = _tzcnt_u32(nl_mask);
        uint32_t first_seg_mask = _bzhi_u32(~0u, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint32_t>(!need_first));

        return _mm_popcnt_u32(bos_mask);
    }

    // Legacy interface
    __attribute__((always_inline, hot))
    static uint64_t count_bos_fast(uint64_t nl_mask, uint64_t ns_mask, bool need_bos_in, bool& need_bos_out) {
        // Segment starts: position after each newline, plus position 0
        uint64_t start = (nl_mask << 1) | 1ULL;

        // Segmented OR-scan: seg_or[i] = OR of ns bits from segment start to i
        uint64_t seg_or = ns_mask | (((start & ~ns_mask) - ns_mask) & ~start);

        // BOS = first ns in each segment
        uint64_t prev_seg = (seg_or << 1) & ~start;
        uint64_t bos_mask = ns_mask & ~prev_seg;

        // Clear first segment if !need_bos_in (branchless)
        // tzcnt(0) = 64, bzhi(x, 64) = x (all bits)
        uint64_t first_nl_pos = _tzcnt_u64(nl_mask);
        uint64_t first_seg_mask = _bzhi_u64(~0ULL, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint64_t>(!need_bos_in));

        // need_bos_out: no ns after last newline (fully branchless)
        uint64_t last_nl_pos = 63 - _lzcnt_u64(nl_mask);
        uint64_t after_last = _bzhi_u64(ns_mask, last_nl_pos + 1) ^ ns_mask;
        // has_nl: -1 if nl_mask!=0, 0 otherwise
        uint64_t has_nl = -static_cast<uint64_t>(nl_mask != 0);
        // no_nl_result: need_bos_in && !ns_mask
        uint64_t no_nl_result = need_bos_in & (ns_mask == 0);
        // Select: has_nl ? (after_last==0) : no_nl_result
        need_bos_out = (has_nl & (after_last == 0)) | (~has_nl & no_nl_result);

        return _mm_popcnt_u64(bos_mask);
    }

    __attribute__((always_inline, hot))
    static uint32_t count_bos_fast_32(uint32_t nl_mask, uint32_t ns_mask, bool need_bos_in, bool& need_bos_out) {
        uint32_t start = (nl_mask << 1) | 1u;
        uint32_t seg_or = ns_mask | (((start & ~ns_mask) - ns_mask) & ~start);
        uint32_t prev_seg = (seg_or << 1) & ~start;
        uint32_t bos_mask = ns_mask & ~prev_seg;

        uint32_t first_nl_pos = _tzcnt_u32(nl_mask);
        uint32_t first_seg_mask = _bzhi_u32(~0u, first_nl_pos);
        bos_mask &= ~(first_seg_mask & -static_cast<uint32_t>(!need_bos_in));

        uint32_t last_nl_pos = 31 - _lzcnt_u32(nl_mask);
        uint32_t after_last = _bzhi_u32(ns_mask, last_nl_pos + 1) ^ ns_mask;
        uint32_t has_nl = -static_cast<uint32_t>(nl_mask != 0);
        uint32_t no_nl_result = need_bos_in & (ns_mask == 0);
        need_bos_out = (has_nl & (after_last == 0)) | (~has_nl & no_nl_result);

        return _mm_popcnt_u32(bos_mask);
    }
};

} // namespace yaal
