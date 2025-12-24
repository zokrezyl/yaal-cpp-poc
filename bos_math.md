# Mathematical Model: Can BOS be Computed Without Per-Newline Iteration?

## Problem Statement

Given a 64-bit chunk with:
- `nl_mask`: bits set at newline positions
- `ns_mask`: bits set at non-space, non-newline positions

Count BOS = number of "first non-space after each newline"

## Key Observation

BOS at position i iff:
1. `ns_mask[i] == 1` (position is non-space)
2. There is NO other non-space between the previous newline and position i

Equivalently: position i is BOS iff the most recent "interesting" position before i is a newline (not another non-space).

## Algorithm (O(log n) parallel prefix)

### Step 1: Compute "reachability" from newlines

Propagate newline influence rightward, BLOCKED by non-spaces:

```
reach = nl_mask
reach |= (reach << 1) & ~ns_mask
reach |= (reach << 2) & ~ns_mask
reach |= (reach << 4) & ~ns_mask
reach |= (reach << 8) & ~ns_mask
reach |= (reach << 16) & ~ns_mask
reach |= (reach << 32) & ~ns_mask
```

After this, `reach[i] == 1` means position i can be "reached" from some newline through only spaces.

### Step 2: Find BOS positions

```
bos_mask = (reach << 1) & ns_mask
```

A non-space is BOS if the position just before it is reachable from a newline.

### Step 3: Handle start of chunk

If `need_bos_at_start == true`, we add the first non-space (if any):
```
if (need_bos_at_start && ns_mask) {
    bos_mask |= ns_mask & (-ns_mask);  // isolate lowest set bit
}
```

### Step 4: Count

```
bos_count = popcnt(bos_mask)
```

## Complexity Analysis

- 6 OR operations
- 6 shift operations
- 6 AND-NOT operations
- 2 more AND/shift for final step
- 1 popcnt

Total: ~20 ALU operations per 64-bit chunk (independent of newline count!)

Compare to newline-only scan: ~8 operations per 64 bits (load, compare, movemask, popcnt × 2 halves)

Overhead ratio: 20/8 = 2.5x

But this is CONSTANT, not proportional to newline count!

## Theoretical Maximum Throughput

With 2.5x more ALU work, if we're memory-bound at 90%, we should still be at 90%!

The ALU operations are fast (1 cycle each), and we have ILP (instruction-level parallelism).

Memory bandwidth: ~20 GB/s
ALU throughput: ~3 GHz × 4 ports = 12 billion ops/sec

For 1GB: 1e9 / 64 = 15.6M chunks
ALU work: 15.6M × 20 = 312M operations
Time for ALU: 312M / 12B = 0.026 sec
Time for memory: 1e9 / 20e9 = 0.05 sec

Memory dominates! So we SHOULD achieve ~90% throughput!

## Why Current Implementation is Slow

Current implementation uses `while (nl_mask)` loop with:
- ~5-10 ops per newline
- 10M newlines = 50-100M extra ops
- This breaks the O(1) per chunk property

## Conclusion

YES, it IS mathematically possible to achieve near-100% memory throughput for BOS counting using parallel prefix bit operations, without per-newline iteration.

The key is replacing the `while (nl_mask)` loop with O(log n) parallel prefix operations.
