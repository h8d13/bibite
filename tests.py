"""test.py 
coverage-style tests for bite.
"""

import unittest

from bite import Bits


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction(unittest.TestCase):
    def test_empty(self):
        b = Bits()
        self.assertEqual(len(b), 0)
        self.assertEqual(bytes(b), b"")
        self.assertEqual(b.to_bin(), "")
        self.assertEqual(b.pos, 0)

    def test_from_bytes(self):
        b = Bits(b"\xab\xcd")
        self.assertEqual(len(b), 16)
        self.assertEqual(b.to_hex(), "abcd")

    def test_from_bytearray(self):
        b = Bits(bytearray([0xAB, 0xCD]))
        self.assertEqual(b.to_hex(), "abcd")

    def test_from_memoryview(self):
        b = Bits(memoryview(b"\xab\xcd"))
        self.assertEqual(b.to_hex(), "abcd")

    def test_from_int_with_nbits(self):
        b = Bits(0xABC, nbits=12)
        self.assertEqual(len(b), 12)
        self.assertEqual(b.to_int(), 0xABC)
        self.assertEqual(b.to_bin(), "101010111100")

    def test_from_int_without_nbits_uses_bit_length(self):
        b = Bits(0xABC)
        self.assertEqual(len(b), 12)
        self.assertEqual(b.to_int(), 0xABC)

    def test_from_int_zero(self):
        b = Bits(0, nbits=8)
        self.assertEqual(b.to_int(), 0)
        self.assertEqual(b.to_bin(), "00000000")

    def test_from_binary_string_prefixed(self):
        b = Bits("0b10101010")
        self.assertEqual(b.to_bin(), "10101010")

    def test_from_binary_string_with_underscores_and_spaces(self):
        b = Bits("0b1010_1010")
        self.assertEqual(b.to_bin(), "10101010")
        b = Bits("0b 1010 1010")
        self.assertEqual(b.to_bin(), "10101010")

    def test_from_hex_string_prefixed(self):
        b = Bits("0xDEADBEEF")
        self.assertEqual(b.to_hex(), "deadbeef")
        self.assertEqual(len(b), 32)

    def test_from_bare_hex(self):
        b = Bits("deadbeef")
        self.assertEqual(b.to_hex(), "deadbeef")

    def test_invalid_binary_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b10201")

    def test_copy_constructor(self):
        a = Bits(0xABC, nbits=12)
        b = Bits(a)
        self.assertEqual(a, b)
        b.append(1)  # mutating copy must not affect source
        self.assertNotEqual(a, b)

    def test_bool_treated_as_int(self):
        # True is 1; without nbits, bit_length is 1
        self.assertEqual(Bits(True).to_bin(), "1")
        self.assertEqual(Bits(False, nbits=4).to_bin(), "0000")

    def test_empty_binary_prefix(self):
        self.assertEqual(len(Bits("0b")), 0)

    def test_empty_hex_prefix(self):
        self.assertEqual(len(Bits("0x")), 0)

    def test_empty_string(self):
        self.assertEqual(len(Bits("")), 0)

    def test_bare_binary_odd_length(self):
        # all 0/1 chars + odd length → parsed as binary
        b = Bits("101")
        self.assertEqual(b.to_bin(), "101")

    def test_bare_binary_with_explicit_nbits(self):
        # all 0/1 chars + nbits given → parsed as binary even when even length
        b = Bits("1010", nbits=4)
        self.assertEqual(b.to_bin(), "1010")

    def test_hex_odd_length_pads_right(self):
        # "abc" -> "abc0", 3 nibbles = 12 bits
        b = Bits("0xabc")
        self.assertEqual(b.to_hex(), "abc0")
        self.assertEqual(len(b), 12)


class TestFactories(unittest.TestCase):
    def test_from_bits(self):
        b = Bits.from_bits([1, 0, 1, 1, 0])
        self.assertEqual(b.to_bin(), "10110")

    def test_from_int_big(self):
        b = Bits.from_int(0xABCD, nbits=16, endian="big")
        self.assertEqual(b.to_bytes(), b"\xab\xcd")

    def test_from_int_little(self):
        b = Bits.from_int(0xABCD, nbits=16, endian="little")
        self.assertEqual(b.to_bytes(), b"\xcd\xab")

    def test_zeros(self):
        b = Bits.zeros(20)
        self.assertEqual(len(b), 20)
        self.assertEqual(b.count(1), 0)
        self.assertEqual(b.count(0), 20)

    def test_ones(self):
        b = Bits.ones(20)
        self.assertEqual(len(b), 20)
        self.assertEqual(b.count(1), 20)
        # tail bits must be masked so to_int matches conceptual value
        self.assertEqual(b.to_int(), (1 << 20) - 1)


# ---------------------------------------------------------------------------
# Indexing / slicing
# ---------------------------------------------------------------------------


class TestIndexing(unittest.TestCase):
    def setUp(self):
        self.b = Bits("0b10110100")

    def test_positive_index(self):
        self.assertEqual(self.b[0], 1)
        self.assertEqual(self.b[1], 0)
        self.assertEqual(self.b[7], 0)

    def test_negative_index(self):
        self.assertEqual(self.b[-1], 0)
        self.assertEqual(self.b[-8], 1)

    def test_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            _ = self.b[8]
        with self.assertRaises(IndexError):
            _ = self.b[-9]

    def test_slice(self):
        self.assertEqual(self.b[2:6].to_bin(), "1101")
        self.assertEqual(self.b[:4].to_bin(), "1011")
        self.assertEqual(self.b[4:].to_bin(), "0100")

    def test_slice_step(self):
        self.assertEqual(self.b[::2].to_bin(), "1100")

    def test_slice_returns_independent_copy(self):
        s = self.b[2:6]
        s[0] = 0
        self.assertEqual(self.b[2], 1)  # original unaffected

    def test_setitem(self):
        b = Bits.zeros(8)
        b[0] = 1
        b[3] = 1
        b[-1] = 1
        self.assertEqual(b.to_bin(), "10010001")


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


class TestMutation(unittest.TestCase):
    def test_append_grows_storage(self):
        b = Bits()
        for bit in [1, 0, 1, 1, 0, 1, 0, 1, 1]:
            b.append(bit)
        self.assertEqual(len(b), 9)
        self.assertEqual(b.to_bin(), "101101011")

    def test_extend(self):
        b = Bits("0b101")
        b.extend([1, 0, 0])
        self.assertEqual(b.to_bin(), "101100")

    def test_clear_resets_pos(self):
        b = Bits("0b1010")
        b.read(2)
        b.clear()
        self.assertEqual(len(b), 0)
        self.assertEqual(b.pos, 0)


# ---------------------------------------------------------------------------
# Streaming read/write
# ---------------------------------------------------------------------------


class TestStreaming(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        b = Bits()
        b.write(0xA, 4)
        b.write(0x3, 3)
        b.write(0xFF, 8)
        self.assertEqual(b.read(4), 0xA)
        self.assertEqual(b.read(3), 0x3)
        self.assertEqual(b.read(8), 0xFF)

    def test_write_bits(self):
        b = Bits()
        b.write_bits(Bits("0b1101"))
        b.write_bits(b"\xff")
        self.assertEqual(b.to_bin(), "1101" + "11111111")

    def test_pad_to_byte(self):
        b = Bits("0b10101")
        b.pad_to_byte()
        self.assertEqual(len(b), 8)
        self.assertEqual(b.to_bin(), "10101000")

    def test_pad_to_byte_with_one(self):
        b = Bits("0b101")
        b.pad_to_byte(1)
        self.assertEqual(b.to_bin(), "10111111")

    def test_pad_to_byte_already_aligned_is_noop(self):
        b = Bits(b"\xab")
        b.pad_to_byte()
        self.assertEqual(len(b), 8)

    def test_read_bits_returns_bits(self):
        b = Bits("0b11001010")
        out = b.read_bits(4)
        self.assertIsInstance(out, Bits)
        self.assertEqual(out.to_bin(), "1100")
        self.assertEqual(b.pos, 4)

    def test_read_bytes(self):
        b = Bits(b"\xab\xcd\xef")
        self.assertEqual(b.read_bytes(2), b"\xab\xcd")
        self.assertEqual(b.pos, 16)

    def test_skip_seek_rewind(self):
        b = Bits("0b11110000")
        b.skip(4)
        self.assertEqual(b.read(4), 0x0)
        b.rewind()
        self.assertEqual(b.read(4), 0xF)
        b.seek(2)
        self.assertEqual(b.pos, 2)

    def test_remaining(self):
        b = Bits(b"\xab")
        self.assertEqual(b.remaining(), 8)
        b.read(3)
        self.assertEqual(b.remaining(), 5)

    def test_read_past_end_raises(self):
        b = Bits("0b101")
        with self.assertRaises(EOFError):
            b.read(4)

    def test_chunks_consumes_from_cursor_and_drops_short_tail(self):
        b = Bits("0xdeadbeef")           # 32 bits
        b.read(4)                        # cursor at 4, 28 bits remain
        got = list(b.chunks(8))          # yields 3x 8-bit, 4-bit tail dropped
        self.assertEqual(got, [0xEA, 0xDB, 0xEE])
        self.assertEqual(b.remaining(), 4)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


class TestConversions(unittest.TestCase):
    def test_to_int_big(self):
        self.assertEqual(Bits(0xABC, nbits=12).to_int("big"), 0xABC)

    def test_to_int_little(self):
        b = Bits(b"\xab\xcd")
        # big: 0xABCD; little: byte-swap -> 0xCDAB
        self.assertEqual(b.to_int("big"), 0xABCD)
        self.assertEqual(b.to_int("little"), 0xCDAB)

    def test_to_int_empty(self):
        self.assertEqual(Bits().to_int(), 0)

    def test_to_bytes(self):
        self.assertEqual(Bits(0xABC, nbits=12).to_bytes(), b"\xab\xc0")

    def test_to_hex(self):
        self.assertEqual(Bits(b"\xde\xad\xbe\xef").to_hex(), "deadbeef")

    def test_to_bin(self):
        self.assertEqual(Bits(b"\xa5").to_bin(), "10100101")

    def test_byteswap(self):
        b = Bits(b"\xab\xcd\xef")
        self.assertEqual(b.byteswap().to_bytes(), b"\xef\xcd\xab")

    def test_byteswap_unaligned_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b101").byteswap()

    def test_roundtrip_int_through_bits(self):
        for v, n in [(0, 1), (1, 1), (0xABC, 12), (0xFFFFFFFF, 32), (0x123456789ABCDEF0, 64)]:
            self.assertEqual(Bits(v, nbits=n).to_int(), v)


# ---------------------------------------------------------------------------
# Bitwise
# ---------------------------------------------------------------------------


class TestBitwise(unittest.TestCase):
    def setUp(self):
        self.a = Bits("0b1100")
        self.c = Bits("0b1010")

    def test_and(self):
        self.assertEqual((self.a & self.c).to_bin(), "1000")

    def test_binop_coerces_non_bits(self):
        # Right-hand operand is bytes; should be wrapped in Bits()
        a = Bits(b"\xff")
        result = a & b"\x0f"
        self.assertEqual(result.to_hex(), "0f")

    def test_or(self):
        self.assertEqual((self.a | self.c).to_bin(), "1110")

    def test_xor(self):
        self.assertEqual((self.a ^ self.c).to_bin(), "0110")

    def test_invert_masks_tail(self):
        b = ~Bits("0b1100")
        self.assertEqual(b.to_bin(), "0011")  # not 00111111
        self.assertEqual(b.to_int(), 0b0011)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _ = Bits("0b1100") & Bits("0b101")

    def test_lshift(self):
        self.assertEqual((Bits("0b1100") << 1).to_bin(), "1000")
        self.assertEqual((Bits("0b1100") << 2).to_bin(), "0000")

    def test_rshift(self):
        self.assertEqual((Bits("0b1100") >> 1).to_bin(), "0110")
        self.assertEqual((Bits("0b1100") >> 2).to_bin(), "0011")

    def test_shift_zero_and_over(self):
        b = Bits("0b1100")
        self.assertEqual((b << 0).to_bin(), "1100")
        self.assertEqual((b >> 0).to_bin(), "1100")
        self.assertEqual((b << 10).to_bin(), "0000")
        self.assertEqual((b >> 10).to_bin(), "0000")

    def test_concat(self):
        self.assertEqual((Bits("0b101") + Bits("0b11")).to_bin(), "10111")

    def test_concat_byte_boundary(self):
        # tests that concatenation works across non-aligned boundaries
        a = Bits("0b10101")          # 5 bits
        b = Bits("0b110011")         # 6 bits
        self.assertEqual((a + b).to_bin(), "10101110011")
        self.assertEqual(len(a + b), 11)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class TestAnalysis(unittest.TestCase):
    def test_count_ones(self):
        self.assertEqual(Bits("0b10110100").count(1), 4)

    def test_count_zeros(self):
        self.assertEqual(Bits("0b10110100").count(0), 4)

    def test_count_excludes_tail_bits(self):
        # ones() must mask its tail; counting must not see phantom bits
        self.assertEqual(Bits.ones(12).count(1), 12)
        self.assertEqual(Bits.ones(12).count(0), 0)

    def test_count_empty(self):
        self.assertEqual(Bits().count(1), 0)
        self.assertEqual(Bits().count(0), 0)

    def test_reversed_bits(self):
        self.assertEqual(Bits("0b10110").reversed_bits().to_bin(), "01101")

    def test_find(self):
        b = Bits("0b00010100")
        self.assertEqual(b.find(1), 3)
        self.assertEqual(b.find(1, start=4), 5)
        self.assertEqual(b.find(0), 0)

    def test_find_not_found(self):
        self.assertEqual(Bits.zeros(8).find(1), -1)
        self.assertEqual(Bits.ones(8).find(0), -1)


# ---------------------------------------------------------------------------
# Dunders
# ---------------------------------------------------------------------------


class TestDunders(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(Bits(0xABC, nbits=12)), 12)
        self.assertEqual(len(Bits()), 0)

    def test_iter(self):
        self.assertEqual(list(Bits("0b1010")), [1, 0, 1, 0])

    def test_bytes_dunder(self):
        self.assertEqual(bytes(Bits(b"\xab\xcd")), b"\xab\xcd")

    def test_int_dunder(self):
        self.assertEqual(int(Bits(0xABC, nbits=12)), 0xABC)

    def test_bool(self):
        self.assertFalse(bool(Bits()))
        self.assertTrue(bool(Bits.zeros(1)))  # length > 0
        self.assertTrue(bool(Bits(0xAB, nbits=8)))

    def test_eq_and_ne(self):
        self.assertEqual(Bits(0xAB, nbits=8), Bits(0xAB, nbits=8))
        self.assertNotEqual(Bits(0xAB, nbits=8), Bits(0xAB, nbits=12))
        self.assertNotEqual(Bits(0xAB, nbits=8), Bits(0xAC, nbits=8))
        self.assertNotEqual(Bits(0xAB, nbits=8), "0xAB")  # not Bits -> not equal

    def test_hash_consistent_with_eq(self):
        a = Bits(0xAB, nbits=8)
        b = Bits(0xAB, nbits=8)
        self.assertEqual(hash(a), hash(b))
        s = {a, b}
        self.assertEqual(len(s), 1)

    def test_hash_distinguishes_length(self):
        # Same byte payload, different nbits → different hash
        self.assertNotEqual(
            hash(Bits(b"\xab", nbits=8)),
            hash(Bits(b"\xab", nbits=4)),
        )

    def test_repr_short(self):
        self.assertEqual(repr(Bits("0b1010")), "Bits('0b1010')")

    def test_repr_long(self):
        b = Bits.zeros(128)
        r = repr(b)
        self.assertIn("128 bits", r)


# ---------------------------------------------------------------------------
# Integration: a realistic protocol-style roundtrip
# ---------------------------------------------------------------------------


class TestProtocolRoundtrip(unittest.TestCase):
    """Encode a tiny packet, decode it back, byte-align included."""

    def test_packet_roundtrip(self):
        # 3-bit version, 5-bit type, 16-bit payload length, 4-bit flags, then pad
        b = Bits()
        b.write(0b101, 3)
        b.write(0b10110, 5)
        b.write(0xBEEF, 16)
        b.write(0b1001, 4)
        b.pad_to_byte()
        self.assertEqual(len(b) % 8, 0)

        # Decode
        b.rewind()
        self.assertEqual(b.read(3), 0b101)
        self.assertEqual(b.read(5), 0b10110)
        self.assertEqual(b.read(16), 0xBEEF)
        self.assertEqual(b.read(4), 0b1001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
