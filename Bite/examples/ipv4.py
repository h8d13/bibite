"""ipv4.py 

Using Bits to encode and decode an IPv4 packet header.
The IPv4 header is the classic case where bit-level access pays off. Several fields are not byte-aligned:

    Version          4 bits  ─┐ packed into the first byte
    IHL              4 bits  ─┘
    DSCP/ECN (ToS)   8 bits
    Total Length    16 bits
    Identification  16 bits
    Flags            3 bits  ─┐ these two cross a byte boundary
    Fragment Offset 13 bits  ─┘
    TTL              8 bits
    Protocol         8 bits
    Header Checksum 16 bits
    Source IP       32 bits
    Destination IP  32 bits

With raw bytes you'd be writing shifts and masks (`byte0 = (version << 4) | ihl`,
`flags = (word >> 13) & 0x7`, etc.) by hand. With Bits, you just write the
fields in order, at their declared widths, and decoding is symmetric.
"""
from dataclasses import dataclass

from bite import Bits

# ---------------------------------------------------------------------------
# Header Data Struc
# ---------------------------------------------------------------------------

@dataclass
class IPv4Header:
    version: int = 4
    ihl: int = 5                  # 5 * 32-bit words = 20-byte header (no options)
    tos: int = 0                  # combined DSCP + ECN
    total_length: int = 20
    identification: int = 0
    flags: int = 0b010            # `0b` prefix = binary literal. Bits: Reserved=0, DF=1, MF=0
    fragment_offset: int = 0
    ttl: int = 64
    protocol: int = 6             # TCP
    checksum: int = 0
    src_ip: int = 0
    dst_ip: int = 0

    def encode(self, fill_checksum: bool = True) -> bytes:
        """Pack the header into 20 bytes (no options). If `fill_checksum`,
        compute and embed the IPv4 header checksum."""
        if fill_checksum:
            self.checksum = 0                                       # zero first, else field skews its own sum
            self.checksum = ipv4_checksum(self.encode(fill_checksum=False))  # encode w/o checksum, hash it, re-encode below

        b = Bits()                          # empty bit-stream buffer
        b.write(self.version, 4)            # write 4 bits, MSB-first
        b.write(self.ihl, 4)                # next 4 bits, fills out byte 0
        b.write(self.tos, 8)                # byte 1
        b.write(self.total_length, 16)      # bytes 2 to 3
        b.write(self.identification, 16)    # bytes 4 to 5
        b.write(self.flags, 3)              # 3 bits, then...
        b.write(self.fragment_offset, 13)   # ...13 more, crossing a byte boundary
        b.write(self.ttl, 8)
        b.write(self.protocol, 8)
        b.write(self.checksum, 16)
        b.write(self.src_ip, 32)
        b.write(self.dst_ip, 32)
        return bytes(b)                     # bit-stream into raw bytes

    @classmethod
    def decode(cls, data: bytes) -> "IPv4Header":
        b = Bits(data)                      # wrap raw bytes; cursor starts at bit 0
        return cls(
            version=b.read(4),              # consume 4 bits, return as unsigned int
            ihl=b.read(4),
            tos=b.read(8),
            total_length=b.read(16),
            identification=b.read(16),
            flags=b.read(3),                # same widths and order as encode (symmetric)
            fragment_offset=b.read(13),
            ttl=b.read(8),
            protocol=b.read(8),
            checksum=b.read(16),
            src_ip=b.read(32),
            dst_ip=b.read(32),
        )


# ---------------------------------------------------------------------------
# Checksum (RFC 791): 16-bit one's-complement of the one's-complement sum
# ---------------------------------------------------------------------------


def ipv4_checksum(header: bytes) -> int:
    """Compute the IPv4 header checksum. Returns 0 when run over a header
    whose own checksum field is already correct useful for verification."""
    if len(header) % 2:                             # checksum reads 16-bit words, need even byte count
        header += b"\x00"                           # `b"..."` = bytes literal; `\x00` = one zero byte
    total = sum(Bits(header).chunks(16))            # sum every 16-bit word
    while total >> 16:                              # `>> 16` shifts right 16 bits. nonzero = sum overflowed 16 bits
        total = (total & 0xFFFF) + (total >> 16)    # `& 0xFFFF` keeps low 16, `>> 16` is the carry; add them back (one's-complement add). `0xFFFF` = 65535 = sixteen 1-bits
    return total ^ 0xFFFF                           # `^` = XOR. XOR with all 1s flips every bit (one's complement)


# ---------------------------------------------------------------------------
# IP address helpers
# ---------------------------------------------------------------------------


def ip_to_int(s: str) -> int:
    b = Bits()
    for octet in s.split("."):     # "192.168.1.10" -> ["192", "168", "1", "10"]
        b.write(int(octet), 8)     # pack each octet as 8 bits, MSB-first
    return b.to_int()              # 32 bits as one big-endian int


def int_to_ip(n: int) -> str:
    # slice 32-bit int into 4 octets, join as dotted quad
    return ".".join(str(c) for c in Bits(n, nbits=32).chunks(8))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    # Build a header: TCP packet, 192.168.1.10 -> 8.8.8.8
    hdr = IPv4Header(
        total_length=40,             # 20 IP + 20 TCP
        identification=0x1C46,       # `0x` prefix = hexadecimal literal. 0x1C46 = 7238 in decimal
        flags=0b010,                 # Don't Fragment
        ttl=64,
        protocol=6,                  # TCP
        src_ip=ip_to_int("192.168.1.10"),
        dst_ip=ip_to_int("8.8.8.8"),
    )

    raw = hdr.encode()
    print(f"Encoded ({len(raw)} bytes):")
    print(f"  {raw.hex(' ')}\n")     # `.hex(' ')` formats bytes as space-separated hex pairs

    # The same 20 bytes you'd see in Wireshark — let's pull them apart.
    decoded = IPv4Header.decode(raw)
    # `&` = bitwise AND. ANDing w/ a single-bit mask isolates that bit; bool() turns 0/non-0 into False/True
    df = bool(decoded.flags & 0b010)
    mf = bool(decoded.flags & 0b001)
    # f-string format spec ":02x" = hex (x), pad to width 2 with zeros. ":04x" = 4-wide hex. ":03b" = 3-wide binary
    print("Decoded:")
    print(f"  Version        : {decoded.version}")
    print(f"  IHL            : {decoded.ihl}  ({decoded.ihl * 4} bytes)")
    print(f"  ToS            : 0x{decoded.tos:02x}")
    print(f"  Total length   : {decoded.total_length}")
    print(f"  Identification : 0x{decoded.identification:04x}")
    print(f"  Flags          : 0b{decoded.flags:03b}  (DF={df}, MF={mf})")
    print(f"  Fragment offset: {decoded.fragment_offset}")
    print(f"  TTL            : {decoded.ttl}")
    print(f"  Protocol       : {decoded.protocol}  (TCP)")
    print(f"  Checksum       : 0x{decoded.checksum:04x}")
    print(f"  Source IP      : {int_to_ip(decoded.src_ip)}")
    print(f"  Destination IP : {int_to_ip(decoded.dst_ip)}\n")

    # A correct header re-checksummed should sum to zero.
    assert ipv4_checksum(raw) == 0, "checksum mismatch"
    print("Checksum verifies ✓  (the header is intact)\n")

    # Mutate a byte, prove the checksum catches it.
    tampered = bytearray(raw)    # bytearray is the mutable cousin of bytes (we need to edit in place)
    tampered[8] ^= 0x01          # `^=` = XOR-assign. XOR with 0x01 flips the lowest bit of byte 8 (TTL)
    if ipv4_checksum(bytes(tampered)) != 0:
        print("Tampered packet ✗  (checksum no longer zero — corruption detected)")


if __name__ == "__main__":
    main()
