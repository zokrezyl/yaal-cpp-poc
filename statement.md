# Problem: O(1) Bit-Parallel BOS Detection

## Goal

Achieve near-100% memory bandwidth throughput for detecting **both**:
1. Newline positions (trivial - already solved)
2. "Beginning of Statement" (BOS) = first non-space character after each newline

## Current Performance

| Operation | Throughput | % of Memory BW |
|-----------|------------|----------------|
| Memory read baseline | 18 GB/s | 100% |
| Newline-only scan | 16 GB/s | ~87% |
| Newline + BOS (current) | 9 GB/s | ~50% |

## Input

For each 64-byte chunk, we have two 64-bit masks:
- `nl_mask`: bit `i` = 1 if position `i` is a newline (`\n`)
- `ns_mask`: bit `i` = 1 if position `i` is a non-space, non-newline character

## Output

Count of BOS positions in the chunk, where:
- **BOS at position `i`** iff `ns_mask[i] == 1` AND there exists `j < i` where `nl_mask[j] == 1` AND all positions in `(j, i)` have `ns_mask == 0`

In other words: a non-space character is a BOS if the nearest "interesting" position before it (either newline or non-space) is a newline.

## Cross-Chunk State

- Input: `need_bos_in` (bool) - true if previous chunk ended with a newline or spaces
- Output: `need_bos_out` (bool) - true if this chunk ends with newline followed by only spaces

When `need_bos_in == true`, the first non-space in the chunk is also a BOS (even without a preceding newline in this chunk).

## Example

```
Input: "ab\n  cd\nef\n\ngh"
Positions: 0=a, 1=b, 2=\n, 3=sp, 4=sp, 5=c, 6=d, 7=\n, 8=e, 9=f, 10=\n, 11=\n, 12=g, 13=h

nl_mask = 0b00110010000100  (bits 2, 7, 10, 11)
ns_mask = 0b11001100110011  (bits 0,1, 5,6, 8,9, 12,13)

BOS positions (with need_bos_in=true):
- Position 0: first non-space at start (need_bos_in=true)
- Position 5: first non-space after \n at position 2
- Position 8: first non-space after \n at position 7
- Position 12: first non-space after \n at position 11

BOS count = 4
```

## Constraint

**Must be O(1) operations per 64-byte chunk** - no loops proportional to number of newlines.

The current O(k) solution (k = newline count) achieves only 50% throughput because the loop creates data dependencies the CPU cannot parallelize.

## Failed Attempt: Parallel Prefix

Tried propagating newline "reachability" using doubling:
```cpp
reach = nl_mask;
reach |= (reach << 1) & ~ns_mask;
reach |= (reach << 2) & ~ns_mask;
reach |= (reach << 4) & ~ns_mask;
// ... etc
bos_mask = (reach << 1) & ns_mask;
```

**Why it fails:** The shift-by-2^k steps can "jump over" blockers. For example, if there's a blocker at position 10, `reach << 4` can still propagate from position 5 to position 9, incorrectly skipping the blocker.

The blocking must be **cumulative through all intermediate positions**, not just checked at the destination.

## Question

Is there a true O(1) bit-parallel algorithm to count BOS positions given `nl_mask`, `ns_mask`, and `need_bos_in`?

Alternatively: is there a proof that this problem requires O(k) operations where k = popcount(nl_mask)?

## Environment

- x86-64 with AVX2, BMI1, BMI2, POPCNT
- Available intrinsics: `_mm_popcnt_u64`, `_tzcnt_u64`, `_lzcnt_u64`, `_blsr_u64`, `_pdep_u64`, `_pext_u64`, etc.
