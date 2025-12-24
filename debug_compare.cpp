// Debug comparison of old vs new parser
#include "yaal/counting_parser.hpp"
#include "yaal/fast_counting_parser.hpp"
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    // Test with known input
    std::vector<std::string> tests = {
        "hello\n",
        "hello\nworld\n",
        "  hello\n",
        "\n\n\n",
        "a\nb\nc\n",
        "abc",
        "  \n  \n  \n",
        "hello world\n  indented\n",
        std::string(100, 'a') + "\n",  // Long line
        std::string(64, 'a') + "\n",   // Exactly 64 chars + newline
        std::string(65, 'a') + "\n",   // 65 chars + newline (spans chunks)
    };

    for (size_t i = 0; i < tests.size(); i++) {
        const std::string& test = tests[i];
        yaal::Buffer buf(test.data(), test.size());

        yaal::CountingParser old_parser;
        old_parser.parse(buf);

        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        bool match = (old_parser.counts().eol == fast_parser.counts().eol &&
                     old_parser.counts().bos == fast_parser.counts().bos);

        std::cout << "Test " << i << " (" << test.size() << " bytes): ";
        if (match) {
            std::cout << "PASS";
        } else {
            std::cout << "FAIL";
        }
        std::cout << " old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;

        if (!match) {
            // Show the input
            std::cout << "  Input: \"";
            for (char c : test) {
                if (c == '\n') std::cout << "\\n";
                else if (c == ' ') std::cout << "_";
                else std::cout << c;
            }
            std::cout << "\"" << std::endl;
        }
    }

    // Test with multi-chunk data
    std::cout << "\nMulti-chunk tests:" << std::endl;

    // Create 128-byte input that spans exactly 2 chunks
    std::string multi(128, 'x');
    multi[63] = '\n';  // Newline at end of first chunk
    multi[127] = '\n'; // Newline at end of second chunk

    {
        yaal::Buffer buf(multi.data(), multi.size());
        yaal::CountingParser old_parser;
        old_parser.parse(buf);
        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        std::cout << "128 bytes, newlines at 63,127: "
                  << "old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;
    }

    // Newline at start of second chunk
    multi[63] = 'x';
    multi[64] = '\n';
    {
        yaal::Buffer buf(multi.data(), multi.size());
        yaal::CountingParser old_parser;
        old_parser.parse(buf);
        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        std::cout << "128 bytes, newlines at 64,127: "
                  << "old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;
    }

    // Test with generated document pattern (similar to benchmark)
    std::cout << "\nGenerated document pattern test:" << std::endl;

    // Generate a small document with same pattern as benchmark
    std::string doc;
    doc.reserve(10000);
    int indent = 0;
    for (int line = 0; line < 100; line++) {
        // Add indentation
        for (int i = 0; i < indent * 4; i++) doc += ' ';
        // Add some words
        doc += "word1 word2 word3";
        doc += '\n';
        // Vary indent
        if (line % 5 == 0) indent = (indent + 1) % 4;
    }

    {
        yaal::Buffer buf(doc.data(), doc.size());
        yaal::CountingParser old_parser;
        old_parser.parse(buf);
        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        bool match = (old_parser.counts().eol == fast_parser.counts().eol &&
                     old_parser.counts().bos == fast_parser.counts().bos);
        std::cout << "Generated doc (" << doc.size() << " bytes): "
                  << (match ? "PASS" : "FAIL")
                  << " old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;
    }

    // Test with more varied content
    doc.clear();
    for (int i = 0; i < 1000; i++) {
        // Random spaces (0-8)
        int spaces = i % 9;
        for (int s = 0; s < spaces; s++) doc += ' ';
        // Some content
        if (i % 10 != 0) {  // 90% non-empty lines
            doc += "content";
        }
        doc += '\n';
    }

    {
        yaal::Buffer buf(doc.data(), doc.size());
        yaal::CountingParser old_parser;
        old_parser.parse(buf);
        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        bool match = (old_parser.counts().eol == fast_parser.counts().eol &&
                     old_parser.counts().bos == fast_parser.counts().bos);
        std::cout << "Varied doc (" << doc.size() << " bytes): "
                  << (match ? "PASS" : "FAIL")
                  << " old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;
    }

    // Stress test: 1MB of varied data
    doc.clear();
    doc.reserve(1024 * 1024);
    while (doc.size() < 1024 * 1024) {
        int spaces = (doc.size() / 7) % 12;
        for (int s = 0; s < spaces; s++) doc += ' ';
        int words = 1 + (doc.size() / 11) % 10;
        for (int w = 0; w < words; w++) {
            if (w > 0) doc += ' ';
            doc += "word";
        }
        doc += '\n';
    }

    {
        yaal::Buffer buf(doc.data(), doc.size());
        yaal::CountingParser old_parser;
        old_parser.parse(buf);
        yaal::FastCountingParser fast_parser;
        fast_parser.parse(buf);

        bool match = (old_parser.counts().eol == fast_parser.counts().eol &&
                     old_parser.counts().bos == fast_parser.counts().bos);
        std::cout << "1MB stress (" << doc.size() << " bytes): "
                  << (match ? "PASS" : "FAIL")
                  << " old(eol=" << old_parser.counts().eol << ",bos=" << old_parser.counts().bos
                  << ") fast(eol=" << fast_parser.counts().eol << ",bos=" << fast_parser.counts().bos
                  << ")" << std::endl;

        if (!match) {
            // Find first divergence
            std::cout << "  Finding first divergence..." << std::endl;
            for (size_t chunk_start = 0; chunk_start < doc.size(); chunk_start += 64) {
                size_t chunk_end = std::min(chunk_start + 64, doc.size());
                std::string chunk = doc.substr(chunk_start, chunk_end - chunk_start);

                yaal::Buffer chunk_buf(doc.data(), chunk_start + chunk.size());
                yaal::CountingParser old_p;
                old_p.parse(chunk_buf);
                yaal::FastCountingParser fast_p;
                fast_p.parse(chunk_buf);

                if (old_p.counts().bos != fast_p.counts().bos) {
                    std::cout << "  Divergence at chunk starting " << chunk_start << std::endl;
                    std::cout << "  old_bos=" << old_p.counts().bos << " fast_bos=" << fast_p.counts().bos << std::endl;
                    break;
                }
            }
        }
    }

    // Trace first 64 bytes of stress test pattern
    std::cout << "\nDetailed trace of first 64 bytes:" << std::endl;
    doc.clear();
    while (doc.size() < 64) {
        int spaces = (doc.size() / 7) % 12;
        for (int s = 0; s < spaces; s++) doc += ' ';
        int words = 1 + (doc.size() / 11) % 10;
        for (int w = 0; w < words; w++) {
            if (w > 0) doc += ' ';
            doc += "word";
        }
        doc += '\n';
    }

    std::cout << "Content (first 64 chars):" << std::endl;
    for (size_t i = 0; i < 64 && i < doc.size(); i++) {
        if (doc[i] == '\n') std::cout << "\\n";
        else if (doc[i] == ' ') std::cout << "_";
        else std::cout << doc[i];
        if ((i+1) % 32 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    // Show newline and non-space positions
    std::cout << "Newlines at: ";
    for (size_t i = 0; i < 64 && i < doc.size(); i++) {
        if (doc[i] == '\n') std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << "Non-spaces at: ";
    for (size_t i = 0; i < 64 && i < doc.size(); i++) {
        if (doc[i] != ' ' && doc[i] != '\n') std::cout << i << " ";
    }
    std::cout << std::endl;

    // Parse just first 64 bytes
    std::string first64 = doc.substr(0, 64);
    {
        yaal::Buffer buf(first64.data(), first64.size());
        yaal::CountingParser old_p;
        old_p.parse(buf);
        yaal::FastCountingParser fast_p;
        fast_p.parse(buf);

        std::cout << "First 64 bytes: old(eol=" << old_p.counts().eol << ",bos=" << old_p.counts().bos
                  << ") fast(eol=" << fast_p.counts().eol << ",bos=" << fast_p.counts().bos
                  << ")" << std::endl;
    }

    return 0;
}
