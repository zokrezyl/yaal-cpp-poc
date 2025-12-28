#pragma once

#include "buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

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

        // Main loop: 192 bytes at a time (3x64 bytes)
        // Unrolled to allow parallel execution of add-with-carry chains
        while (pos + 192 <= len) {
            // Load 6 chunks (192 bytes)
            __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
            __m256i c1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 32));
            __m256i c2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 64));
            __m256i c3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 96));
            __m256i c4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 128));
            __m256i c5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos + 160));

            // Compute newline masks (3 x 64-bit)
            uint64_t nl_mask_0 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, newline_vec)));
            uint64_t nl_mask_1 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c3, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, newline_vec)));
            uint64_t nl_mask_2 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c5, newline_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c4, newline_vec)));

            // Compute space masks (3 x 64-bit)
            uint64_t sp_mask_0 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, space_vec)));
            uint64_t sp_mask_1 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c3, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c2, space_vec)));
            uint64_t sp_mask_2 = (static_cast<uint64_t>(static_cast<uint32_t>(
                _mm256_movemask_epi8(_mm256_cmpeq_epi8(c5, space_vec)))) << 32) |
                static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(c4, space_vec)));

            // Compute whitespace masks
            uint64_t ws_mask_0 = sp_mask_0 | nl_mask_0;
            uint64_t ws_mask_1 = sp_mask_1 | nl_mask_1;
            uint64_t ws_mask_2 = sp_mask_2 | nl_mask_2;

            // Compute BOS masks using add-with-carry (3 parallel chains)
            uint64_t bos_mask_0 = compute_bos_mask(nl_mask_0, ws_mask_0, need_bos, need_bos);
            uint64_t bos_mask_1 = compute_bos_mask(nl_mask_1, ws_mask_1, need_bos, need_bos);
            uint64_t bos_mask_2 = compute_bos_mask(nl_mask_2, ws_mask_2, need_bos, need_bos);

            // Emit events for chunk 0
            emit_events(nl_mask_0, bos_mask_0, pos);
            // Emit events for chunk 1
            emit_events(nl_mask_1, bos_mask_1, pos + 64);
            // Emit events for chunk 2
            emit_events(nl_mask_2, bos_mask_2, pos + 128);

            pos += 192;
        }

        // Handle 64-byte chunks
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
            uint64_t bos_mask = compute_bos_mask(nl_mask, ws_mask, need_bos, need_bos);

            emit_events(nl_mask, bos_mask, pos);
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
            uint32_t bos_mask = compute_bos_mask_32(nl_mask, ws_mask, need_bos, need_bos);

            emit_events_32(nl_mask, bos_mask, pos);
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

    // SFINAE helper to detect batch support
    template<typename T, typename = void>
    struct has_batch_support : std::false_type {};

    template<typename T>
    struct has_batch_support<T, std::void_t<decltype(T::supports_batch)>>
        : std::bool_constant<T::supports_batch> {};

    __attribute__((always_inline, hot))
    void emit_events(uint64_t nl_mask, uint64_t bos_mask, [[maybe_unused]] size_t base_pos) {
        if constexpr (has_batch_support<Derived>::value) {
            // Fast path: use batch callbacks with popcnt
            derived().on_eol_batch(_mm_popcnt_u64(nl_mask));
            derived().on_bos_batch(_mm_popcnt_u64(bos_mask));
        } else {
            // Slow path: iterate through each bit
            while (nl_mask) {
                uint64_t nl_pos = _tzcnt_u64(nl_mask);
                derived().on_eol(base_pos + nl_pos);
                nl_mask &= nl_mask - 1;
            }
            while (bos_mask) {
                uint64_t bos_pos = _tzcnt_u64(bos_mask);
                derived().on_bos(base_pos + bos_pos);
                bos_mask &= bos_mask - 1;
            }
        }
    }

    __attribute__((always_inline, hot))
    void emit_events_32(uint32_t nl_mask, uint32_t bos_mask, [[maybe_unused]] size_t base_pos) {
        if constexpr (has_batch_support<Derived>::value) {
            // Fast path: use batch callbacks with popcnt
            derived().on_eol_batch(_mm_popcnt_u32(nl_mask));
            derived().on_bos_batch(_mm_popcnt_u32(bos_mask));
        } else {
            // Slow path: iterate through each bit
            while (nl_mask) {
                uint32_t nl_pos = _tzcnt_u32(nl_mask);
                derived().on_eol(base_pos + nl_pos);
                nl_mask &= nl_mask - 1;
            }
            while (bos_mask) {
                uint32_t bos_pos = _tzcnt_u32(bos_mask);
                derived().on_bos(base_pos + bos_pos);
                bos_mask &= bos_mask - 1;
            }
        }
    }
};

} // namespace yaal
