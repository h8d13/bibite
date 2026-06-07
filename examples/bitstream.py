"""bitstream.py

Pack mixed sub-byte fields into one dense bitstream, then read them back.

When fields are narrower than a byte (a 3-bit type tag, a 1-bit flag, a 12-bit
id), storing each in its own byte wastes most of the space. write/read move
fields at their exact widths so they sit shoulder to shoulder with no padding
between them. pad_to_byte aligns only the final tail, once.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bite import Bits

# (value, width) records with mixed sub-byte widths.
RECORDS = [
    (0b101, 3),     # 3-bit tag
    (0b10110, 5),   # 5-bit code
    (0xABC, 12),    # 12-bit id
    (0b1, 1),       # 1-bit flag
    (0b011, 3),
    (0xF, 5),
]


def pack(records: list[tuple[int, int]]) -> Bits:
    b = Bits()
    for value, width in records:
        b.write(value, width)
    b.pad_to_byte()          # align the tail so the stream is whole bytes
    return b


def unpack(b: Bits, records: list[tuple[int, int]]) -> list[int]:
    b.rewind()
    return [b.read(width) for _, width in records]


def main() -> None:
    payload_bits = sum(width for _, width in RECORDS)
    packed = pack(RECORDS)

    naive_bytes = len(RECORDS)          # one byte per field
    print(f"Records        : {len(RECORDS)} fields, {payload_bits} payload bits")
    print(f"Packed         : {packed.to_hex()}  ({len(packed)} bits, {len(packed) // 8} bytes)")
    print(f"Naive layout   : {naive_bytes} bytes (one per field)")
    print(f"Saved          : {naive_bytes - len(packed) // 8} bytes\n")

    values = unpack(packed, RECORDS)
    originals = [v for v, _ in RECORDS]
    assert values == originals, f"roundtrip mismatch: {values} != {originals}"
    print(f"Read back      : {values}")
    print("Roundtrip OK\n")

    # The padded stream is byte-aligned, so chunks(8) walks it byte by byte.
    packed.rewind()
    octets = list(packed.chunks(8))
    print(f"As bytes       : {' '.join(f'{o:02x}' for o in octets)}")


if __name__ == "__main__":
    main()
