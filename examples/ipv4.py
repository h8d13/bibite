"""ipv4.py

Encode and decode an IPv4 packet header with Bits.

The IPv4 header is the classic case where bit-level access pays off: several
fields are not byte-aligned.

    Version          4 bits   packed into the first byte with IHL
    IHL              4 bits
    DSCP/ECN (ToS)   8 bits
    Total Length    16 bits
    Identification  16 bits
    Flags            3 bits   Flags + Fragment Offset cross a byte boundary
    Fragment Offset 13 bits
    TTL              8 bits
    Protocol         8 bits
    Header Checksum 16 bits
    Source IP       32 bits
    Destination IP  32 bits

With raw bytes you would hand-write shifts and masks (byte0 = (version << 4) | ihl,
flags = (word >> 13) & 0x7, ...). With Bits you write the fields in order at
their declared widths, and decoding is the mirror image.

Run from the repo root: python -m examples.ipv4
"""
from dataclasses import dataclass

from bite import Bits


@dataclass
class IPv4Header:
    version: int = 4
    ihl: int = 5                  # 5 * 32-bit words = 20-byte header, no options
    tos: int = 0                  # combined DSCP + ECN
    total_length: int = 20
    identification: int = 0
    flags: int = 0b010            # Reserved=0, DF=1, MF=0
    fragment_offset: int = 0
    ttl: int = 64
    protocol: int = 6             # TCP
    checksum: int = 0
    src_ip: int = 0
    dst_ip: int = 0

    def encode(self, fill_checksum: bool = True) -> bytes:
        """Pack the header into 20 bytes. With fill_checksum, compute and embed
        the IPv4 header checksum."""
        if fill_checksum:
            self.checksum = 0  # the checksum field must read as zero while being summed
            self.checksum = ipv4_checksum(self.encode(fill_checksum=False))

        b = Bits()
        b.write(self.version, 4)            # fills byte 0 together with ihl
        b.write(self.ihl, 4)
        b.write(self.tos, 8)
        b.write(self.total_length, 16)
        b.write(self.identification, 16)
        b.write(self.flags, 3)              # 3 + 13 bits straddle a byte boundary
        b.write(self.fragment_offset, 13)
        b.write(self.ttl, 8)
        b.write(self.protocol, 8)
        b.write(self.checksum, 16)
        b.write(self.src_ip, 32)
        b.write(self.dst_ip, 32)
        return bytes(b)

    @classmethod
    def decode(cls, data: bytes) -> "IPv4Header":
        b = Bits(data)
        return cls(
            version=b.read(4),
            ihl=b.read(4),
            tos=b.read(8),
            total_length=b.read(16),
            identification=b.read(16),
            flags=b.read(3),                # same widths and order as encode
            fragment_offset=b.read(13),
            ttl=b.read(8),
            protocol=b.read(8),
            checksum=b.read(16),
            src_ip=b.read(32),
            dst_ip=b.read(32),
        )


def ipv4_checksum(header: bytes) -> int:
    """IPv4 header checksum (RFC 791): 16-bit one's-complement of the
    one's-complement sum of every 16-bit word. Run over a header whose checksum
    field is already correct, it returns 0, which is how verification works."""
    if len(header) % 2:                             # the sum reads 16-bit words
        header += b"\x00"
    total = sum(Bits(header).chunks(16))
    while total >> 16:                              # fold the carry back in (one's-complement add)
        total = (total & 0xFFFF) + (total >> 16)
    return total ^ 0xFFFF                           # final one's complement


def ip_to_int(s: str) -> int:
    b = Bits()
    for octet in s.split("."):
        b.write(int(octet), 8)
    return b.to_int()


def int_to_ip(n: int) -> str:
    return ".".join(str(c) for c in Bits(n, nbits=32).chunks(8))


def main() -> None:
    # TCP packet, 192.168.1.10 -> 8.8.8.8
    hdr = IPv4Header(
        total_length=40,             # 20 IP + 20 TCP
        identification=0x1C46,
        flags=0b010,                 # Don't Fragment
        ttl=64,
        protocol=6,                  # TCP
        src_ip=ip_to_int("192.168.1.10"),
        dst_ip=ip_to_int("8.8.8.8"),
    )

    raw = hdr.encode()
    print(f"Encoded ({len(raw)} bytes):")
    print(f"  {raw.hex(' ')}\n")

    decoded = IPv4Header.decode(raw)
    df = bool(decoded.flags & 0b010)
    mf = bool(decoded.flags & 0b001)
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

    # A correct header, re-summed including its checksum field, folds to zero.
    assert ipv4_checksum(raw) == 0, "checksum mismatch"
    print("Checksum verifies  (the header is intact)\n")

    # Flip one bit and the checksum stops folding to zero.
    tampered = bytearray(raw)
    tampered[8] ^= 0x01          # lowest bit of byte 8 (TTL)
    if ipv4_checksum(bytes(tampered)) != 0:
        print("Tampered packet  (checksum no longer zero, corruption detected)")


if __name__ == "__main__":
    main()
