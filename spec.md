# YAAL Parser Specification

YAAL (Yet Another Awesome Language) is a simplified YAML-like language designed for ultra-fast SIMD parsing.

## Language Definition

### Core Concepts

- **Document**: A sequence of bytes to be parsed
- **Statement**: A line containing at least one non-space character. The statement value starts at the first non-space character.
- **Indentation**: Number of leading space characters (0x20) before the first non-space character

### Delimiters

| Character | Byte | Role |
|-----------|------|------|
| Space | 0x20 | Indentation delimiter (only spaces, NOT tabs) |
| Newline | 0x0A | Line delimiter (only LF, NOT CRLF) |

### Rules

1. Tab (0x09) is treated as a non-space character
2. UTF-8 content is allowed (space and newline bytes don't conflict with UTF-8 multi-byte sequences)
3. Empty lines (containing only spaces or nothing) do not constitute statements
4. Trailing whitespace handling is delegated to higher-level parsers

## SIMD Parser Base Class

The base parser is a low-level event emitter that detects structural boundaries using AVX2 SIMD instructions. It uses CRTP (Curiously Recurring Template Pattern) for zero-overhead callbacks.

### Events

| Event | Signature | Description |
|-------|-----------|-------------|
| `on_bod` | `void on_bod(size_t offset)` | Begin of document. Always emitted first. Offset is 0. |
| `on_bos` | `void on_bos(size_t offset)` | Begin of statement. First non-space character after bod or eol. |
| `on_eos` | `void on_eos(size_t offset)` | End of statement. Emitted only if a bos was emitted. Offset points to the terminating `\n` or end position. |
| `on_eol` | `void on_eol(size_t offset)` | End of line. Emitted for every `\n` character. Offset points to the `\n`. |
| `on_eod` | `void on_eod(size_t offset)` | End of document. Always emitted last. Offset is buffer length (one past last byte). |

### Event Ordering

Events are emitted in strict positional order with the following rules:

1. `on_bod` is always first
2. `on_eos` is emitted **before** `on_eol` (if a statement was open)
3. `on_eos` is emitted **before** `on_eod` (if a statement was open and no trailing newline)
4. `on_eod` is always last

### State Machine

```
State: looking_for_bos = true (initially)

PROCESS(byte, offset):
    if byte == '\n':
        if not looking_for_bos:
            emit on_eos(offset)
        emit on_eol(offset)
        looking_for_bos = true

    else if byte != ' ' and looking_for_bos:
        emit on_bos(offset)
        looking_for_bos = false

START:
    emit on_bod(0)
    looking_for_bos = true

END(length):
    if not looking_for_bos:
        emit on_eos(length)
    emit on_eod(length)
```

### Examples

#### Example 1: Simple lines
Input: `"hello\nworld\n"` (length = 12)

```
on_bod(0)
on_bos(0)    // 'h' at position 0
on_eos(5)    // '\n' at position 5
on_eol(5)    // '\n' at position 5
on_bos(6)    // 'w' at position 6
on_eos(11)   // '\n' at position 11
on_eol(11)   // '\n' at position 11
on_eod(12)
```

#### Example 2: Indented lines
Input: `"  hello\n  world\n"` (length = 16)

```
on_bod(0)
on_bos(2)    // 'h' at position 2 (2 spaces before)
on_eos(7)    // '\n' at position 7
on_eol(7)    // '\n' at position 7
on_bos(10)   // 'w' at position 10 (2 spaces before)
on_eos(15)   // '\n' at position 15
on_eol(15)   // '\n' at position 15
on_eod(16)
```

#### Example 3: Empty lines and whitespace-only lines
Input: `"hello\n\n   \nworld\n"` (length = 18)

```
on_bod(0)
on_bos(0)    // 'h'
on_eos(5)    // first '\n'
on_eol(5)    // first '\n'
on_eol(6)    // second '\n' (empty line - no bos/eos)
on_eol(10)   // third '\n' (whitespace-only line - no bos/eos)
on_bos(11)   // 'w'
on_eos(16)   // fourth '\n'
on_eol(16)   // fourth '\n'
on_eod(18)
```

#### Example 4: No trailing newline
Input: `"hello"` (length = 5)

```
on_bod(0)
on_bos(0)    // 'h'
on_eos(5)    // end of document triggers eos
on_eod(5)
```

#### Example 5: Trailing newlines
Input: `"hello\n\n\n"` (length = 8)

```
on_bod(0)
on_bos(0)    // 'h'
on_eos(5)    // '\n'
on_eol(5)    // '\n'
on_eol(6)    // '\n' (no eos - no statement)
on_eol(7)    // '\n' (no eos - no statement)
on_eod(8)    // no eos before eod - no open statement
```

## Buffer Interface

```cpp
class Buffer {
public:
    const char* start() const;
    size_t len() const;
};
```

## SIMD Implementation Strategy (AVX2)

### Algorithm

1. Process 32 bytes at a time using AVX2 (256-bit registers)
2. Create bitmasks:
   - `newline_mask`: bit set where byte == `\n` (0x0A)
   - `space_mask`: bit set where byte == ` ` (0x20)
   - `non_space_mask`: ~space_mask & ~newline_mask
3. Use bit manipulation (`tzcnt`, `blsr`) to iterate set bits in positional order
4. Maintain `looking_for_bos` state across 32-byte chunks
5. Handle tail bytes (< 32) with scalar fallback or masked load

### Key Insight

The parser is memory-bandwidth limited because:
- SIMD comparison and mask extraction is O(1) per 32 bytes
- Bit iteration is O(events) which is typically << O(bytes)
- No branching in the hot path except for event emission

### CRTP Pattern

```cpp
template<typename Derived>
class YaalParserBase {
public:
    void parse(const Buffer& buf) {
        // SIMD implementation
        // Calls: derived().on_bod(offset), derived().on_bos(offset), etc.
    }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
};

// Usage
class MyParser : public YaalParserBase<MyParser> {
public:
    void on_bod(size_t offset) { /* ... */ }
    void on_bos(size_t offset) { /* ... */ }
    void on_eos(size_t offset) { /* ... */ }
    void on_eol(size_t offset) { /* ... */ }
    void on_eod(size_t offset) { /* ... */ }
};
```

## Higher-Level Semantics (Future)

The base parser only emits low-level events. Higher-level parsers will build on this to:

1. Track indentation levels (offset of bos minus offset of previous eol + 1)
2. Build parent-child relationships based on indentation changes
3. Distinguish simple statements from compound statements
4. Extract statement values (from bos to eos, optionally trimming trailing whitespace)
