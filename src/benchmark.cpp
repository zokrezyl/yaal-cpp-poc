#include "yaal/counting_parser.hpp"
#include "yaal/fast_counting_parser.hpp"
#include "yaal/fast_event_parser.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <random>
#include <immintrin.h>

// Fast xorshift64 PRNG
class FastRandom {
public:
    explicit FastRandom(uint64_t seed = 12345) : state_(seed) {}

    uint64_t next() {
        uint64_t x = state_;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state_ = x;
        return x;
    }

    uint64_t next(uint64_t max) { return next() % max; }

private:
    uint64_t state_;
};

std::vector<std::string> load_words(const char* path) {
    std::vector<std::string> words;
    std::ifstream file(path);
    std::string word;
    while (std::getline(file, word)) {
        if (!word.empty()) words.push_back(word);
    }
    return words;
}

std::vector<char> generate_document(
    const std::vector<std::string>& words,
    size_t target_size,
    size_t avg_words_per_line,
    size_t avg_lines_per_indent_level,
    uint64_t seed = 42
) {
    FastRandom rng(seed);
    std::vector<char> doc;
    doc.reserve(target_size + 1024);

    const size_t num_words = words.size();
    int current_indent = 0;
    size_t lines_at_current_indent = 0;

    while (doc.size() < target_size) {
        lines_at_current_indent++;
        if (lines_at_current_indent >= avg_lines_per_indent_level) {
            uint64_t r = rng.next(3);
            if (r == 0 && current_indent > 0) current_indent--;
            else if (r == 1 && current_indent < 10) current_indent++;
            lines_at_current_indent = 0;
        }

        for (int i = 0; i < current_indent * 4; i++) doc.push_back(' ');

        size_t num_words_this_line = 1 + rng.next(avg_words_per_line * 2);
        for (size_t w = 0; w < num_words_this_line; w++) {
            if (w > 0) doc.push_back(' ');
            const std::string& word = words[rng.next(num_words)];
            for (char c : word) doc.push_back(c);
        }
        doc.push_back('\n');
    }
    return doc;
}

// Measure READ-ONLY memory throughput using sum (not memcpy!)
uint64_t sum_bytes_simd(const char* data, size_t len) {
    __m256i sum = _mm256_setzero_si256();
    size_t pos = 0;

    while (pos + 32 <= len) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
        sum = _mm256_add_epi64(sum, _mm256_sad_epu8(chunk, _mm256_setzero_si256()));
        pos += 32;
    }

    // Extract sum
    uint64_t result = 0;
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), sum);
    result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    while (pos < len) {
        result += static_cast<uint8_t>(data[pos]);
        pos++;
    }
    return result;
}

double measure_read_throughput(const char* data, size_t len, int iterations) {
    // Warmup
    volatile uint64_t dummy = sum_bytes_simd(data, len);
    (void)dummy;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        volatile uint64_t s = sum_bytes_simd(data, len);
        (void)s;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    return (static_cast<double>(len) * iterations) / elapsed_sec;
}

// Simple newline counter
uint64_t count_newlines_simd(const char* data, size_t len) {
    const __m256i newline_vec = _mm256_set1_epi8('\n');
    uint64_t count = 0;
    size_t pos = 0;

    while (pos + 32 <= len) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + pos));
        uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(chunk, newline_vec)));
        count += _mm_popcnt_u32(mask);
        pos += 32;
    }
    while (pos < len) {
        if (data[pos] == '\n') count++;
        pos++;
    }
    return count;
}

double measure_newline_throughput(const char* data, size_t len, int iterations) {
    volatile uint64_t dummy = count_newlines_simd(data, len);
    (void)dummy;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        volatile uint64_t c = count_newlines_simd(data, len);
        (void)c;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    return (static_cast<double>(len) * iterations) / elapsed_sec;
}

double measure_parser_throughput(const yaal::Buffer& buf, yaal::CountingParser& parser, int iterations) {
    parser.reset();
    parser.parse(buf);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        parser.reset();
        parser.parse(buf);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    return (static_cast<double>(buf.len()) * iterations) / elapsed_sec;
}

double measure_fast_parser_throughput(const yaal::Buffer& buf, yaal::FastCountingParser& parser, int iterations) {
    parser.reset();
    parser.parse(buf);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        parser.reset();
        parser.parse(buf);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    return (static_cast<double>(buf.len()) * iterations) / elapsed_sec;
}

double measure_crtp_parser_throughput(const yaal::Buffer& buf, yaal::FastCountingParserV2& parser, int iterations) {
    parser.reset();
    parser.parse(buf);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        parser.reset();
        parser.parse(buf);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(end - start).count();
    return (static_cast<double>(buf.len()) * iterations) / elapsed_sec;
}

void print_throughput(double bytes_per_sec) {
    double gb_per_sec = bytes_per_sec / (1024.0 * 1024.0 * 1024.0);
    std::cout << std::fixed << std::setprecision(2) << gb_per_sec << " GB/s";
}

int main(int argc, char* argv[]) {
    size_t target_size = 1024ULL * 1024 * 1024;
    size_t avg_words_per_line = 8;
    size_t avg_lines_per_indent = 5;
    const char* dict_path = "/usr/share/dict/words";
    int iterations = 5;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
            target_size = std::stoull(argv[++i]) * 1024 * 1024;
        else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
            iterations = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dict") == 0 && i + 1 < argc)
            dict_path = argv[++i];
    }

    std::cout << "=== YAAL Parser Benchmark ===" << std::endl << std::endl;

    std::cout << "Loading dictionary..." << std::endl;
    auto words = load_words(dict_path);
    std::cout << "Loaded " << words.size() << " words" << std::endl << std::endl;

    std::cout << "Generating " << (target_size / (1024*1024)) << " MB document..." << std::endl;
    auto doc = generate_document(words, target_size, avg_words_per_line, avg_lines_per_indent);
    std::cout << "Generated " << doc.size() << " bytes" << std::endl << std::endl;

    yaal::Buffer buf(doc.data(), doc.size());

    std::cout << "Running benchmarks (" << iterations << " iterations each)..." << std::endl << std::endl;

    double read_tp = measure_read_throughput(doc.data(), doc.size(), iterations);
    double nl_tp = measure_newline_throughput(doc.data(), doc.size(), iterations);

    yaal::CountingParser parser;
    double parser_tp = measure_parser_throughput(buf, parser, iterations);

    yaal::FastCountingParser fast_parser;
    double fast_tp = measure_fast_parser_throughput(buf, fast_parser, iterations);

    yaal::FastCountingParserV2 crtp_parser;
    double crtp_tp = measure_crtp_parser_throughput(buf, crtp_parser, iterations);

    parser.reset();
    parser.parse(buf);
    fast_parser.reset();
    fast_parser.parse(buf);
    crtp_parser.reset();
    crtp_parser.parse(buf);

    std::cout << "=== Results ===" << std::endl << std::endl;

    std::cout << "Memory read bandwidth: ";
    print_throughput(read_tp);
    std::cout << " (baseline)" << std::endl;

    std::cout << "Newline scan:          ";
    print_throughput(nl_tp);
    std::cout << " (" << std::fixed << std::setprecision(1) << (nl_tp / read_tp * 100) << "%)" << std::endl;

    std::cout << "Full parser (old):     ";
    print_throughput(parser_tp);
    std::cout << " (" << std::fixed << std::setprecision(1) << (parser_tp / read_tp * 100) << "%)" << std::endl;

    std::cout << "Fast parser (new):     ";
    print_throughput(fast_tp);
    std::cout << " (" << std::fixed << std::setprecision(1) << (fast_tp / read_tp * 100) << "%)" << std::endl;

    std::cout << "CRTP parser:           ";
    print_throughput(crtp_tp);
    std::cout << " (" << std::fixed << std::setprecision(1) << (crtp_tp / read_tp * 100) << "%)" << std::endl;

    std::cout << std::endl << "Old parser counts: eol=" << parser.counts().eol
              << " bos=" << parser.counts().bos << std::endl;
    std::cout << "Fast parser counts: eol=" << fast_parser.counts().eol
              << " bos=" << fast_parser.counts().bos << std::endl;
    std::cout << "CRTP parser counts: eol=" << crtp_parser.counts().eol
              << " bos=" << crtp_parser.counts().bos << std::endl;

    if (parser.counts().eol == fast_parser.counts().eol &&
        parser.counts().bos == fast_parser.counts().bos &&
        parser.counts().eol == crtp_parser.counts().eol &&
        parser.counts().bos == crtp_parser.counts().bos) {
        std::cout << "Counts MATCH!" << std::endl;
    } else {
        std::cout << "WARNING: Counts MISMATCH!" << std::endl;
    }

    return 0;
}
