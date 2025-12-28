#include <boost/ut.hpp>
#include <string>
#include <vector>

#include "yaal/counting_parser.hpp"
#include "yaal/fast_counting_parser.hpp"

using namespace boost::ut;

// Helper to compare parsers
struct ParseResult {
    uint64_t bos;
    uint64_t eol;
};

ParseResult parse_with_reference(const std::string& input) {
    yaal::CountingParser parser;
    yaal::Buffer buf(input.data(), input.size());
    parser.parse(buf);
    return {parser.counts().bos, parser.counts().eol};
}

ParseResult parse_with_fast(const std::string& input) {
    yaal::FastCountingParser parser;
    yaal::Buffer buf(input.data(), input.size());
    parser.parse(buf);
    return {parser.counts().bos, parser.counts().eol};
}

suite parser_tests = [] {
    "basic_single_line"_test = [] {
        std::string input = "hello\n";
        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    "basic_indented_line"_test = [] {
        std::string input = "  hello\n";
        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    "multiple_lines"_test = [] {
        std::string input = "hello\nworld\n";
        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    // Tests specifically targeting the 32-byte remainder loop
    // The bug: ns_mask is passed instead of sp_mask to count_bos_fast_32

    "remainder_32bytes_with_spaces"_test = [] {
        // Exactly 32 bytes - processed only by 32-byte remainder loop
        // "  hello world  test string!\n" + padding to 32 bytes
        std::string input = "  hello world test string!!\n";  // 28 chars
        input += "abc\n";  // 32 chars total

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for 32-byte input with spaces";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch for 32-byte input with spaces";
    };

    "remainder_64bytes_with_spaces"_test = [] {
        // 64 bytes - two iterations of 32-byte remainder loop
        std::string input(64, 'x');
        // Add some spaces and newlines
        input[0] = ' ';
        input[1] = ' ';
        input[10] = '\n';
        input[11] = ' ';
        input[12] = ' ';
        input[20] = '\n';
        input[40] = '\n';
        input[41] = ' ';
        input[63] = '\n';

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for 64-byte input";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch for 64-byte input";
    };

    "remainder_100bytes_indented_lines"_test = [] {
        // 100 bytes - exercises 32-byte remainder loop (100 < 192)
        std::string input;
        input += "    first line with indent\n";      // 27 chars
        input += "  second line\n";                   // 14 chars
        input += "third\n";                           // 6 chars
        input += "    fourth with spaces\n";          // 23 chars
        input += "  fifth line here\n";               // 18 chars
        // Pad to exactly 100 bytes
        while (input.size() < 99) input += 'x';
        input += '\n';

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for 100-byte indented input";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch for 100-byte indented input";
    };

    "remainder_after_192byte_main_loop"_test = [] {
        // 250 bytes = 192 (main loop) + 58 (remainder handled by 32-byte loop)
        // This tests that the remainder after main loop is handled correctly
        std::string input(250, 'a');
        // Add spaces and newlines in the remainder portion (after byte 192)
        input[200] = ' ';
        input[201] = ' ';
        input[210] = '\n';
        input[211] = ' ';
        input[212] = ' ';
        input[213] = ' ';
        input[220] = '\n';
        input[240] = '\n';
        input[249] = '\n';

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for 250-byte input (192+58 remainder)";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch for 250-byte input";
    };

    "remainder_heavy_indentation"_test = [] {
        // Heavy indentation pattern in 32-byte remainder range
        std::string input;
        for (int i = 0; i < 5; i++) {
            input += std::string(i * 2, ' ');  // increasing indent
            input += "text\n";
        }
        // Pad to 80 bytes (within 32-byte remainder range, < 192)
        while (input.size() < 79) input += ' ';
        input += '\n';

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for heavily indented 80-byte input";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    "remainder_empty_lines_with_spaces"_test = [] {
        // Lines that are only spaces (no BOS should be detected for these)
        std::string input;
        input += "real\n";          // has BOS
        input += "     \n";         // only spaces, no BOS
        input += "  text\n";        // has BOS at 't'
        input += "   \n";           // only spaces, no BOS
        input += "end\n";           // has BOS
        // Pad to 50 bytes
        while (input.size() < 49) input += ' ';
        input += '\n';

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch for empty lines with spaces";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    "remainder_all_spaces_between_newlines"_test = [] {
        // Specifically tests space mask vs non-space mask handling
        std::string input = "a\n";
        input += std::string(30, ' ');  // 30 spaces
        input += "\nb\n";
        // Total: 2 + 30 + 2 = 34 bytes, hits 32-byte remainder

        auto ref = parse_with_reference(input);
        auto fast = parse_with_fast(input);
        expect(eq(ref.bos, fast.bos)) << "BOS mismatch: spaces between newlines";
        expect(eq(ref.eol, fast.eol)) << "EOL mismatch";
    };

    "stress_various_sizes_32_to_191"_test = [] {
        // Test all sizes from 32 to 191 (all handled by remainder loop only)
        for (size_t size = 32; size < 192; size++) {
            std::string input(size, 'x');
            // Add some structure
            if (size > 10) {
                input[0] = ' ';
                input[1] = ' ';
            }
            if (size > 20) {
                input[size/2] = '\n';
                input[size/2 + 1] = ' ';
            }
            if (size > 5) {
                input[size - 1] = '\n';
            }

            auto ref = parse_with_reference(input);
            auto fast = parse_with_fast(input);
            expect(eq(ref.bos, fast.bos)) << "BOS mismatch at size " << size;
            expect(eq(ref.eol, fast.eol)) << "EOL mismatch at size " << size;
        }
    };
};

int main() {
    return 0;
}
