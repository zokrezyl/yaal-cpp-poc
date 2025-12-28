// Test and verify the parallel prefix BOS counting algorithm
#include <cstdint>
#include <iostream>
#include <cassert>
#include <immintrin.h>
#include <x86intrin.h>

// Naive O(n) algorithm - iterate through each newline
uint64_t count_bos_naive(uint64_t nl_mask, uint64_t ns_mask, bool need_bos) {
    uint64_t count = 0;

    // Handle start
    if (need_bos && ns_mask) {
        count++;
        // Find first ns position
        uint64_t first_ns = _tzcnt_u64(ns_mask);
        // Find first nl position
        uint64_t first_nl = nl_mask ? _tzcnt_u64(nl_mask) : 64;
        // If first ns is before first nl, we found bos at start
        if (first_ns < first_nl) {
            need_bos = false;
        } else {
            count--; // Undo the count, will be handled by nl loop
        }
    }

    // Process each newline
    uint64_t remaining = nl_mask;
    while (remaining) {
        uint64_t nl_pos = _tzcnt_u64(remaining);
        remaining &= remaining - 1;

        uint64_t next_nl_pos = remaining ? _tzcnt_u64(remaining) : 64;

        // Search for non-space in (nl_pos, next_nl_pos)
        // Clear bits 0..nl_pos and bits next_nl_pos..63
        uint64_t search_mask = ns_mask;
        // Clear bits 0 to nl_pos (we want positions AFTER nl_pos)
        if (nl_pos < 63) {
            search_mask &= (~0ULL << (nl_pos + 1));
        } else {
            search_mask = 0; // nl at position 63, nothing after it
        }
        // Clear bits from next_nl_pos onward
        if (next_nl_pos < 64) search_mask &= ((1ULL << next_nl_pos) - 1);

        if (search_mask) {
            count++;
        }
    }

    return count;
}

// arithmetic algorithm
uint64_t count_bos_carry(uint64_t nl_mask, uint64_t sp_mask, bool need_bos) {
    unsigned long long e2;
    _addcarry_u64(need_bos, sp_mask, nl_mask, &e2);
    uint64_t bos_mask = e2 & ~sp_mask;
    return _mm_popcnt_u64(bos_mask);
}

void test_case(const char* name, uint64_t nl_mask, uint64_t ns_mask, bool need_bos) {
    uint64_t naive = count_bos_naive(nl_mask, ns_mask, need_bos);
    uint64_t carry = count_bos_carry(nl_mask, ~ns_mask, need_bos);

    std::cout << name << ": ";
    if (naive == carry) {
        std::cout << "PASS (count=" << naive << ")" << std::endl;
    } else {
        std::cout << "FAIL! naive=" << naive << " carry=" << carry << std::endl;
        std::cout << "  nl_mask=0x" << std::hex << nl_mask << std::endl;
        std::cout << "  ns_mask=0x" << std::hex << ns_mask << std::dec << std::endl;
    }
}

int main() {
    // Test case 1: "ab\n  cd\nef\n\ngh" (conceptual)
    // Positions: 0=a, 1=b, 2=\n, 3=sp, 4=sp, 5=c, 6=d, 7=\n, 8=e, 9=f, 10=\n, 11=\n, 12=g, 13=h
    // nl at: 2, 7, 10, 11
    // ns at: 0,1, 5,6, 8,9, 12,13
    // Expected BOS: 0 (start), 5 (after \n@2), 8 (after \n@7), 12 (after \n@11)
    // Note: \n@10 followed by \n@11, so no bos there
    {
        uint64_t nl = (1ULL<<2) | (1ULL<<7) | (1ULL<<10) | (1ULL<<11);
        uint64_t ns = (1ULL<<0) | (1ULL<<1) | (1ULL<<5) | (1ULL<<6) |
                      (1ULL<<8) | (1ULL<<9) | (1ULL<<12) | (1ULL<<13);
        test_case("Test 1: mixed content", nl, ns, true);
    }

    // Test case 2: No newlines "abcdef"
    {
        uint64_t nl = 0;
        uint64_t ns = 0x3F; // bits 0-5
        test_case("Test 2: no newlines, need_bos=true", nl, ns, true);
        test_case("Test 2: no newlines, need_bos=false", nl, ns, false);
    }

    // Test case 3: Only newlines "\n\n\n"
    {
        uint64_t nl = 0x7; // bits 0,1,2
        uint64_t ns = 0;
        test_case("Test 3: only newlines", nl, ns, true);
    }

    // Test case 4: Alternating "a\nb\nc\n"
    {
        uint64_t nl = (1ULL<<1) | (1ULL<<3) | (1ULL<<5);
        uint64_t ns = (1ULL<<0) | (1ULL<<2) | (1ULL<<4);
        test_case("Test 4: alternating", nl, ns, true);
    }

    // Test case 5: Leading spaces "  ab\n"
    {
        uint64_t nl = (1ULL<<4);
        uint64_t ns = (1ULL<<2) | (1ULL<<3);
        test_case("Test 5: leading spaces, need_bos=true", nl, ns, true);
        test_case("Test 5: leading spaces, need_bos=false", nl, ns, false);
    }

    // Test case 6: Spaces after newline "\n  ab"
    {
        uint64_t nl = (1ULL<<0);
        uint64_t ns = (1ULL<<3) | (1ULL<<4);
        test_case("Test 6: spaces after newline", nl, ns, false);
    }

    // Test case 7: Empty (no content)
    {
        test_case("Test 7: empty", 0, 0, true);
        test_case("Test 7: empty", 0, 0, false);
    }

    // Test case 8: Complex pattern with many newlines
    {
        // Pattern: every 4th position is newline, rest is ns except some spaces
        uint64_t nl = 0x1111111111111111ULL; // newline every 4th position
        uint64_t ns = 0x6666666666666666ULL; // non-space at positions 1,2, 5,6, 9,10, etc.
        test_case("Test 8: regular pattern", nl, ns, true);
    }

    // Test case 9: All non-spaces
    {
        uint64_t nl = 0;
        uint64_t ns = ~0ULL;
        test_case("Test 9: all non-space, need_bos=true", nl, ns, true);
        test_case("Test 9: all non-space, need_bos=false", nl, ns, false);
    }

    // Test case 10: Single newline at end
    {
        uint64_t nl = (1ULL << 63);
        uint64_t ns = 0x7FFFFFFFFFFFFFFFULL;
        test_case("Test 10: single newline at end", nl, ns, true);
    }

    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}
