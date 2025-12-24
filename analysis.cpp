// Mathematical proof: can we count bos without iterating through newlines?
//
// Given:
//   nl_mask: bits set at newline positions
//   ns_mask: bits set at non-space, non-newline positions
//
// bos_count = number of newlines that have at least one non-space AFTER them
//             (before the next newline or end of chunk)
//
// For a 64-bit chunk with newlines at positions n0, n1, n2, ...
// We need to check:
//   - Is there any bit set in ns_mask in range (n0+1, n1-1]?
//   - Is there any bit set in ns_mask in range (n1+1, n2-1]?
//   - etc.
//
// This is a SEGMENT problem - we need to check each segment independently.
//
// Key insight: We can use the "carry-propagate" trick!
//
// If we compute: (ns_mask + nl_mask) XOR ns_mask XOR nl_mask
// This gives us carry bits that propagate through non-space regions.
//
// Actually, simpler approach:
//
// For each newline, we want to know if there's a non-space before the next newline.
//
// Let's define:
//   after_nl = nl_mask shifted right by 1 (positions just after newlines)
//
// The "fill" from after_nl until blocked by another newline can be computed with:
//   fill = after_nl
//   fill |= (fill << 1) & ~nl_mask   // propagate unless blocked by newline
//   fill |= (fill << 2) & ~nl_mask
//   fill |= (fill << 4) & ~nl_mask
//   ... up to 64 bits
//
// Then: bos_positions = fill & ns_mask (first non-space in each segment)
// But we want COUNT, not positions!
//
// Alternative: count segments that contain at least one non-space
//
// Observation: if a segment [n_i+1, n_{i+1}] contains a non-space,
// then (ns_mask & segment_mask) != 0
//
// We can compute segment_mask for ALL segments simultaneously using parallel prefix.
//
// Actually, the cleanest approach:
//
// 1. Mark "reachable from newline" positions:
//    reach = nl_mask
//    reach |= (reach << 1) & ~nl_mask  // can reach pos+1 if not blocked
//    reach |= (reach << 2) & ~nl_mask
//    reach |= (reach << 4) & ~nl_mask
//    reach |= (reach << 8) & ~nl_mask
//    reach |= (reach << 16) & ~nl_mask
//    reach |= (reach << 32) & ~nl_mask
//
// 2. Find first non-space in each reachable region:
//    candidates = reach & ns_mask
//
// 3. But we need to count UNIQUE bos (one per newline), not total candidates
//
// The count of unique bos = count of newlines that have at least one reachable non-space
//
// Hmm, this is still tricky because multiple positions can be "reached" from one newline.
//
// NEW APPROACH: Think about it from the non-space perspective
//
// A position is a bos if:
//   - It's a non-space
//   - All positions between it and the previous newline are spaces or newlines
//
// In other words: a non-space at position i is a bos if there's no non-space
// in the range (prev_newline, i).
//
// This means: bos positions are the FIRST non-space after each newline.
//
// We can compute this as:
//   prev_nl_or_ns = nl_mask | ns_mask  // positions that "reset" the search
//   For each bit in ns_mask, check if all bits between it and prev are 0 in ns_mask
//
// Using parallel prefix from the LEFT (finding leftmost set bit before each position):
//   1. Compute "most recent newline or ns before each position"
//   2. A ns is bos if the most recent is a newline (not another ns)
//
// This CAN be computed with O(log n) parallel prefix operations!
//
// PROOF SKETCH:
//
// Let combined = nl_mask | ns_mask  (all "interesting" positions)
//
// For each position i in ns_mask, we want to know if the closest set bit
// to the left in "combined" is in nl_mask (meaning it's a newline, not another ns).
//
// Step 1: Compute "closest set bit to the left" for each position
//         This is a parallel prefix OR from left to right, masked appropriately.
//
// Step 2: Check if that closest bit is in nl_mask
//
// The count can then be computed with popcnt!
//
// Let me implement this...

#include <cstdint>
#include <immintrin.h>
#include <iostream>

// Compute bos_count without iterating through individual newlines
// Returns the number of non-space characters that are the FIRST non-space after a newline
uint64_t count_bos_parallel(uint64_t nl_mask, uint64_t ns_mask, bool need_bos_at_start) {
    if (ns_mask == 0) return 0;

    // A position i in ns_mask is a bos if:
    // - Either i=0 and need_bos_at_start, OR
    // - The closest set bit to the left in (nl_mask | ns_mask) is in nl_mask

    // Strategy: for each bit in ns_mask, find if there's a newline between it
    // and the previous non-space (or start of chunk).

    // Key insight: ns position i is bos iff there's NO other ns in range (prev_nl, i)
    // Equivalently: the previous set bit in (nl_mask | ns_mask) is a newline

    // Compute: for each ns position, is there a newline "just before" it (with only spaces between)?

    // Method: propagate newline "influence" rightward until blocked by ns
    // nl_reach[i] = 1 if position i can be "reached" from a newline through spaces only

    uint64_t nl_reach = nl_mask;
    // Parallel prefix: propagate right, blocked by ns
    uint64_t blocker = ns_mask;
    nl_reach |= (nl_reach << 1) & ~blocker;
    nl_reach |= (nl_reach << 2) & ~blocker;
    nl_reach |= (nl_reach << 4) & ~blocker;
    nl_reach |= (nl_reach << 8) & ~blocker;
    nl_reach |= (nl_reach << 16) & ~blocker;
    nl_reach |= (nl_reach << 32) & ~blocker;

    // A non-space is bos if it's in the "reach" of a newline
    // But we need to shift by 1 because bos is AFTER the newline
    uint64_t bos_candidates = (nl_reach << 1) & ns_mask;

    // Handle start of chunk
    if (need_bos_at_start) {
        // Position 0 is bos if it's non-space and we need_bos
        bos_candidates |= (ns_mask & 1);
        // Also, any leading non-spaces (before first newline) - only the first one
        // This is already handled if we consider "virtual newline at position -1"
    }

    // Actually wait - we're double counting. Each newline can produce at most 1 bos.
    // The above gives us ALL non-spaces reachable from newlines, not just the FIRST one.

    // We need: first ns in each "segment" after a newline
    //
    // Better approach: bos = ns positions that are NOT preceded by another ns (in the same segment)
    //
    // bos = ns_mask & ~(ns_reach_from_nl)
    // where ns_reach_from_nl propagates ns influence (blocked by newlines)

    // Let me reconsider...
    //
    // Define: segment[i] = the segment containing position i (between two newlines)
    // bos in segment = first ns in that segment
    //
    // For each ns at position i:
    //   - It's bos if there's no other ns between prev_newline and i
    //   - Equivalently: propagate ns leftward (blocked by nl), if it reaches a nl, it's bos

    // Propagate ns leftward until blocked by newline
    uint64_t ns_reach_left = ns_mask;
    uint64_t nl_blocker = nl_mask;
    ns_reach_left |= (ns_reach_left >> 1) & ~nl_blocker;
    ns_reach_left |= (ns_reach_left >> 2) & ~nl_blocker;
    ns_reach_left |= (ns_reach_left >> 4) & ~nl_blocker;
    ns_reach_left |= (ns_reach_left >> 8) & ~nl_blocker;
    ns_reach_left |= (ns_reach_left >> 16) & ~nl_blocker;
    ns_reach_left |= (ns_reach_left >> 32) & ~nl_blocker;

    // A ns is bos if it "reaches" a newline position when propagated left
    // ns_reach_left has all positions reachable from ns going left until newline
    // We want to check if any of these overlap with nl_mask

    // Hmm, this gives us reachability, not the bos itself.

    // SIMPLER:
    // bos = first ns in each segment = ns that is NOT immediately preceded by another ns (in same segment)
    //
    // ns at position i is bos iff (ns_mask >> 1) & (1 << i) == 0 OR there's a newline at i-1
    // In other words: bit i-1 in ns_mask is 0, OR bit i-1 in nl_mask is 1
    //
    // bos_mask = ns_mask & (~(ns_mask >> 1) | (nl_mask >> 1))
    // Wait, this doesn't handle segments correctly...

    // Actually the simplest formulation:
    // Position i is bos iff:
    //   - ns_mask[i] = 1 (it's a non-space)
    //   - For all j in (prev_nl, i): ns_mask[j] = 0 (no non-space before it in segment)
    //
    // This is exactly "first set bit in ns_mask after each newline"
    //
    // We can compute this as:
    //   bos_mask = ns_mask & ~(ns_mask - nl_shifted)  where nl_shifted fills segments
    //
    // Hmm, let me think differently.
    //
    // Use subtraction trick:
    // If we subtract 1 from a mask, it flips all bits from LSB up to and including first set bit.
    //
    // For each segment, we want to keep only the first set bit in ns_mask.
    //
    // If we had segment boundaries, we could use: first_in_segment = x & ~(x - 1) per segment
    //
    // With multiple segments... this is the "parallel first-set-bit per segment" problem.

    // FINAL APPROACH using PDEP/PEXT:
    // This is getting complicated. Let me just verify the theory first.

    return _mm_popcnt_u64(bos_candidates); // placeholder
}

// Test
int main() {
    // Test case: "ab\n  cd\nef\n\ngh"
    //             0123456789...
    // nl at: 2, 6, 9, 10
    // ns at: 0,1, 4,5, 7,8, 11,12
    // bos should be at: 0 (start), 4 (after nl@2), 7 (after nl@6), 11 (after nl@10)
    // (nl@9 followed by nl@10, so no bos there)

    uint64_t nl_mask = (1ULL << 2) | (1ULL << 6) | (1ULL << 9) | (1ULL << 10);
    uint64_t ns_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 4) | (1ULL << 5) |
                       (1ULL << 7) | (1ULL << 8) | (1ULL << 11) | (1ULL << 12);

    std::cout << "nl_mask: " << std::hex << nl_mask << std::endl;
    std::cout << "ns_mask: " << std::hex << ns_mask << std::endl;

    // Expected bos: 0, 4, 7, 11 = 4 total
    // Let's trace through manually...

    return 0;
}
