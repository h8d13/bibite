"""bite
 a small, all-round toolkit for raw bit/byte manipulation.

Convention: bit 0 is the most significant bit. Bits are packed MSB-first within
each byte (network order). Endianness, where it appears, refers to byte order
in int<->Bits conversions, not bit order within bytes.

Bits is also a read/write stream: `write(value, nbits)` appends, while
`read(n)` consumes from an internal cursor (`pos`).
"""

from typing import Iterable, Iterator, Literal, Union, overload

BitsLike = Union["Bits", bytes, bytearray, memoryview, int, str, Iterable[int]]
Endian = Literal["big", "little"]


class Bits:
    """A mutable bit-string with built-in read/write cursor.

    Examples:
        Bits()                        # empty
        Bits(b"\\xab\\xcd")           # from bytes
        Bits(0xABC, nbits=12)         # from int, left-aligned in storage
        Bits("0b10101010")            # from binary literal
        Bits("0xdeadbeef")            # from hex literal
        Bits.from_bits([1, 0, 1, 1])  # from iterable of 0/1
        Bits.zeros(20) / Bits.ones(20)

        # Streaming use:
        b = Bits()
        b.write(0xA, 4); b.write(0xFF, 8)
        b.read(4)  # -> 0xA
        b.read(8)  # -> 0xFF
    """

    __slots__ = ("_data", "_nbits", "pos")

    # ---- construction --------------------------------------------------

    def __init__(self, source: BitsLike = b"", nbits: int | None = None) -> None:
        self.pos: int = 0
        if isinstance(source, Bits):
            self._data = bytearray(source._data)
            self._nbits = source._nbits if nbits is None else nbits
            return
        if isinstance(source, bool):  # bool is int — guard before int branch
            source = int(source)
        if isinstance(source, int):
            n = nbits if nbits is not None else max(source.bit_length(), 1)
            self._data = bytearray((source << ((-n) % 8)).to_bytes((n + 7) // 8, "big"))
            self._nbits = n
            return
        if isinstance(source, str):
            self._data, self._nbits = self._parse_str(source, nbits)
            return
        # bytes-like or iterable of byte values
        self._data = bytearray(source)
        self._nbits = nbits if nbits is not None else len(self._data) * 8

    @staticmethod
    def _parse_str(s: str, nbits: int | None) -> tuple[bytearray, int]:
        s = s.strip().replace("_", "").replace(" ", "").lower()
        if s.startswith("0b"):
            s = s[2:]
            if not s:
                return bytearray(), 0
            if set(s) - {"0", "1"}:
                raise ValueError(f"invalid binary literal: {s!r}")
            n = len(s)
            v = int(s, 2) << ((-n) % 8)
            return bytearray(v.to_bytes((n + 7) // 8, "big")), nbits if nbits is not None else n
        if s.startswith("0x"):
            s = s[2:]
        if not s:
            return bytearray(), 0
        if set(s) <= {"0", "1"} and (nbits is not None or len(s) % 2):
            n = len(s)
            v = int(s, 2) << ((-n) % 8)
            return bytearray(v.to_bytes((n + 7) // 8, "big")), nbits if nbits is not None else n
        orig_len = len(s)
        if orig_len % 2:
            s += "0"
        return bytearray.fromhex(s), nbits if nbits is not None else orig_len * 4

    @classmethod
    def from_bits(cls, bits: Iterable[int]) -> "Bits":
        out = cls()
        for b in bits:
            out.append(b)
        return out

    @classmethod
    def from_int(cls, value: int, nbits: int, endian: Endian = "big") -> "Bits":
        b = cls(value, nbits=nbits)
        return b.byteswap() if endian == "little" else b

    @classmethod
    def zeros(cls, nbits: int) -> "Bits":
        return cls(b"\x00" * ((nbits + 7) // 8), nbits=nbits)

    @classmethod
    def ones(cls, nbits: int) -> "Bits":
        b = cls(b"\xff" * ((nbits + 7) // 8), nbits=nbits)
        b._mask_tail()
        return b

    # ---- internals -----------------------------------------------------

    def _bitpos(self, i: int) -> tuple[int, int]:
        if i < 0:
            i += self._nbits
        if not 0 <= i < self._nbits:
            raise IndexError(i)
        return i >> 3, 7 - (i & 7)

    def _mask_tail(self) -> None:
        tail = (-self._nbits) % 8
        if tail and self._data:
            self._data[-1] &= (0xFF << tail) & 0xFF

    # ---- bit access ----------------------------------------------------

    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, i: slice) -> "Bits": ...
    def __getitem__(self, i):
        if isinstance(i, slice):
            return Bits.from_bits(self[j] for j in range(*i.indices(self._nbits)))
        b, s = self._bitpos(i)
        return (self._data[b] >> s) & 1

    def __setitem__(self, i: int, v: int) -> None:
        b, s = self._bitpos(i)
        m = 1 << s
        self._data[b] = (self._data[b] | m) if v else (self._data[b] & ~m)

    # ---- mutation ------------------------------------------------------

    def append(self, bit: int) -> None:
        if self._nbits & 7 == 0:
            self._data.append(0)
        self._nbits += 1
        self[self._nbits - 1] = bit

    def extend(self, other: Iterable[int]) -> None:
        for b in other:
            self.append(b)

    def clear(self) -> None:
        self._data.clear()
        self._nbits = 0
        self.pos = 0

    # ---- streaming I/O -------------------------------------------------

    def write(self, value: int, nbits: int) -> None:
        """Append the low `nbits` of `value`, MSB-first."""
        for i in range(nbits - 1, -1, -1):
            self.append((value >> i) & 1)

    def write_bits(self, b: BitsLike) -> None:
        """Append another Bits (or anything BitsLike) at the end."""
        self.extend(b if isinstance(b, Bits) else Bits(b))

    def pad_to_byte(self, bit: int = 0) -> None:
        while self._nbits & 7:
            self.append(bit)

    def read(self, n: int) -> int:
        """Consume n bits from the cursor and return them as an unsigned int."""
        if self.pos + n > self._nbits:
            raise EOFError(f"requested {n} bits, only {self._nbits - self.pos} remain")
        v = 0
        for _ in range(n):
            v = (v << 1) | self[self.pos]
            self.pos += 1
        return v

    def read_bits(self, n: int) -> "Bits":
        out = self[self.pos : self.pos + n]
        self.pos += n
        return out

    def read_bytes(self, n: int) -> bytes:
        return bytes(self.read_bits(n * 8))

    def chunks(self, n: int) -> Iterator[int]:
        """Yield successive `n`-bit ints from the cursor; stop when fewer than `n` bits remain."""
        while self.remaining() >= n:
            yield self.read(n)

    def skip(self, n: int) -> None:
        self.pos += n

    def seek(self, pos: int) -> None:
        self.pos = pos

    def rewind(self) -> None:
        self.pos = 0

    def remaining(self) -> int:
        return self._nbits - self.pos

    # ---- conversions ---------------------------------------------------

    def to_int(self, endian: Endian = "big") -> int:
        if self._nbits == 0:
            return 0
        v = int.from_bytes(self._data, "big") >> ((-self._nbits) % 8)
        if endian == "big":
            return v
        nbytes = (self._nbits + 7) // 8
        return int.from_bytes(v.to_bytes(nbytes, "big"), "little")

    def to_bytes(self) -> bytes: return bytes(self._data)
    def to_hex(self) -> str: return self._data.hex()
    def to_bin(self) -> str: return "".join(str(self[i]) for i in range(self._nbits))

    def byteswap(self) -> "Bits":
        if self._nbits % 8:
            raise ValueError("byteswap requires byte-aligned length")
        return Bits(bytes(reversed(self._data)), nbits=self._nbits)

    # ---- bitwise -------------------------------------------------------

    def _binop(self, other: "Bits", op) -> "Bits":
        if not isinstance(other, Bits):
            other = Bits(other)
        if self._nbits != other._nbits:
            raise ValueError(f"length mismatch: {self._nbits} vs {other._nbits}")
        out = Bits(bytes(op(a, b) for a, b in zip(self._data, other._data)), nbits=self._nbits)
        out._mask_tail()
        return out

    def __and__(self, o: "Bits") -> "Bits": return self._binop(o, lambda a, b: a & b)
    def __or__(self, o: "Bits") -> "Bits":  return self._binop(o, lambda a, b: a | b)
    def __xor__(self, o: "Bits") -> "Bits": return self._binop(o, lambda a, b: a ^ b)

    def __invert__(self) -> "Bits":
        out = Bits(bytes(~b & 0xFF for b in self._data), nbits=self._nbits)
        out._mask_tail()
        return out

    def __lshift__(self, n: int) -> "Bits":
        if n <= 0:           return Bits(self)
        if n >= self._nbits: return Bits.zeros(self._nbits)
        return self[n:] + Bits.zeros(n)

    def __rshift__(self, n: int) -> "Bits":
        if n <= 0:           return Bits(self)
        if n >= self._nbits: return Bits.zeros(self._nbits)
        return Bits.zeros(n) + self[: self._nbits - n]

    def __add__(self, other: "Bits") -> "Bits":
        """Bit-level concatenation (NOT arithmetic)."""
        out = Bits(self)
        out.extend(other if isinstance(other, Bits) else Bits(other))
        return out

    # ---- analysis ------------------------------------------------------

    def count(self, bit: int = 1) -> int:
        ones = sum(b.bit_count() for b in self._data)
        tail = (-self._nbits) % 8
        if tail and self._data:
            ones -= (self._data[-1] & ((1 << tail) - 1)).bit_count()
        return ones if bit else self._nbits - ones

    def reversed_bits(self) -> "Bits":
        return Bits.from_bits(self[i] for i in range(self._nbits - 1, -1, -1))

    def find(self, bit: int, start: int = 0) -> int:
        for i in range(start, self._nbits):
            if self[i] == bit:
                return i
        return -1

    # ---- dunders -------------------------------------------------------

    def __len__(self) -> int: return self._nbits
    def __iter__(self) -> Iterator[int]: return (self[i] for i in range(self._nbits))
    def __bytes__(self) -> bytes: return bytes(self._data)
    def __int__(self) -> int: return self.to_int()
    def __bool__(self) -> bool: return self._nbits > 0

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Bits): return NotImplemented
        return self._nbits == o._nbits and bytes(self._data) == bytes(o._data)

    def __hash__(self) -> int:
        return hash((bytes(self._data), self._nbits))

    def __repr__(self) -> str:
        if self._nbits <= 64:
            return f"Bits('0b{self.to_bin()}')"
        return f"Bits(<{self._nbits} bits, 0x{self.to_hex()[:16]}…>)"
