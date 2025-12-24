# Bit-Parallel Challenge: O(1) "First Set Bit After Each Set Bit"

Given two 64-bit masks `A` and `B`, count positions where `B[i]=1` and the nearest set bit in `A|B` before position `i` is in `A`.

Equivalently: for each segment between consecutive bits in `A`, does `B` have any bit set?

**Example:** `A=0b10010000`, `B=0b01100110` → answer is 2 (positions 1 and 5)

Newline scan alone: 87% memory bandwidth. Adding this drops to 50%.

Is there an O(1) bit-parallel solution using x86 BMI/AVX2, or is O(popcount(A)) the lower bound?
