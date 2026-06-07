"""Coverage-gap tests for bite.

Targets behaviors/branches not exercised by the existing suite: zero-width
edge cases, cursor-bounds raises, input coercion/normalization, slicing and
shift boundaries, equality/hash protocol corners, repr at the 64-bit boundary,
and masked-tail conversions on non-byte-aligned Bits. Expected values are
hand-traced MSB-first (bit 0 = MSB).
"""

import unittest

from bite import Bits


class TestZeroWidthEdges(unittest.TestCase):
    def test_read_zero_is_noop(self):
        b = Bits("0b101")
        self.assertEqual(b.read(0), 0)
        self.assertEqual(b.pos, 0)

    def test_read_zero_at_end_of_stream(self):
        b = Bits("0b101")
        b.read(3)
        self.assertEqual(b.read(0), 0)  # pos == nbits, 0 bits still legal
        self.assertEqual(b.pos, 3)

    def test_read_bits_zero_returns_empty(self):
        b = Bits("0b101")
        out = b.read_bits(0)
        self.assertEqual(len(out), 0)
        self.assertEqual(b.pos, 0)

    def test_read_bytes_zero_returns_empty(self):
        self.assertEqual(Bits(b"\xab").read_bytes(0), b"")

    def test_write_zero_width_is_noop(self):
        b = Bits("0b1")
        b.write(0xFF, 0)
        self.assertEqual(b.to_bin(), "1")
        self.assertEqual(len(b), 1)

    def test_pad_to_byte_on_empty_is_noop(self):
        b = Bits()
        b.pad_to_byte()
        self.assertEqual(len(b), 0)

    def test_byteswap_empty(self):
        b = Bits().byteswap()  # 0 % 8 == 0, allowed
        self.assertEqual(len(b), 0)
        self.assertEqual(b.to_bin(), "")

    def test_reversed_bits_empty(self):
        self.assertEqual(Bits().reversed_bits().to_bin(), "")

    def test_find_on_empty(self):
        self.assertEqual(Bits().find(1), -1)
        self.assertEqual(Bits().find(0), -1)

    def test_chunks_when_remaining_less_than_n(self):
        # 3 bits, chunk size 8 -> nothing yielded, cursor untouched
        b = Bits("0b101")
        self.assertEqual(list(b.chunks(8)), [])
        self.assertEqual(b.pos, 0)

    def test_empty_slice(self):
        self.assertEqual(len(Bits()[:]), 0)


class TestCursorBounds(unittest.TestCase):
    def test_seek_past_end_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b1010").seek(99)

    def test_seek_negative_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b1010").seek(-1)

    def test_seek_to_exact_end_allowed(self):
        b = Bits("0b101")
        b.seek(3)  # == nbits, valid
        self.assertEqual(b.remaining(), 0)

    def test_skip_negative_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b1010").skip(-1)

    def test_skip_past_end_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b1010").skip(99)

    def test_skip_to_exact_end_allowed(self):
        b = Bits("0b1010")
        b.skip(4)
        self.assertEqual(b.pos, 4)

    def test_read_bits_past_end_raises(self):
        b = Bits("0b101")
        with self.assertRaises(EOFError):
            b.read_bits(4)

    def test_read_bytes_past_end_raises(self):
        b = Bits(b"\xab")  # 8 bits, ask for 16
        with self.assertRaises(EOFError):
            b.read_bytes(2)


class TestInputCoercion(unittest.TestCase):
    def test_setitem_truthy_normalizes_to_one(self):
        b = Bits.zeros(4)
        b[0] = 2  # any truthy -> 1
        self.assertEqual(b[0], 1)
        self.assertEqual(b.to_bin(), "1000")

    def test_from_bits_truthy_values_normalize(self):
        self.assertEqual(Bits.from_bits([2, 0, 5, 0]).to_bin(), "1010")

    def test_extend_truthy_values_normalize(self):
        b = Bits()
        b.extend([3, 0, 9])
        self.assertEqual(b.to_bin(), "101")

    def test_append_truthy_normalizes(self):
        b = Bits()
        b.append(7)
        self.assertEqual(b.to_bin(), "1")

    def test_add_coerces_bytes_operand(self):
        # 0b1010 (4 bits) ++ Bits(b"\xff") (8 bits) -> 12 bits
        self.assertEqual((Bits("0b1010") + b"\xff").to_bin(), "101011111111")

    def test_write_negative_value_uses_low_bits(self):
        # (-1 >> i) & 1 == 1 for all i -> all ones
        b = Bits()
        b.write(-1, 4)
        self.assertEqual(b.to_bin(), "1111")
        # -2 == ...11110 -> low 4 bits 1110
        b = Bits()
        b.write(-2, 4)
        self.assertEqual(b.to_bin(), "1110")

    def test_from_list_of_byte_values(self):
        # iterable (list) branch, distinct from bytes/bytearray/memoryview
        self.assertEqual(Bits([0xAB, 0xCD]).to_hex(), "abcd")
        self.assertEqual(len(Bits([0xAB, 0xCD])), 16)

    def test_from_tuple_of_byte_values(self):
        b = Bits((0xFF, 0x00))
        self.assertEqual(b.to_hex(), "ff00")
        self.assertEqual(len(b), 16)

    def test_from_generator_of_byte_values(self):
        self.assertEqual(Bits(iter([0x01, 0x02])).to_hex(), "0102")

    def test_write_bits_coerces_int(self):
        b = Bits()
        b.write_bits(0xAB)  # Bits(0xAB) -> 8 bits
        self.assertEqual(b.to_bin(), "10101011")

    def test_binop_int_operand_length_mismatch_raises(self):
        # Bits(0x0F) is 4 bits (bit_length), 8-bit lhs -> mismatch
        with self.assertRaises(ValueError):
            _ = Bits(b"\xff") & 0x0F


class TestBoundaries(unittest.TestCase):
    def test_lshift_by_exactly_nbits(self):
        self.assertEqual((Bits("0b1100") << 4).to_bin(), "0000")

    def test_rshift_by_exactly_nbits(self):
        self.assertEqual((Bits("0b1100") >> 4).to_bin(), "0000")

    def test_slice_negative_step_full_reverse(self):
        b = Bits("0b10110100")
        self.assertEqual(b[::-1].to_bin(), "00101101")

    def test_slice_negative_step_range(self):
        # indices(8) for 6:2:-1 -> 6,5,4,3 -> bits at those positions
        # 10110100: idx6=0,5=1,4=0,3=1 -> "0101"
        b = Bits("0b10110100")
        self.assertEqual(b[6:2:-1].to_bin(), "0101")

    def test_slice_indices_clamped_high(self):
        b = Bits("0b10110100")
        self.assertEqual(b[2:99].to_bin(), "110100")

    def test_slice_indices_clamped_low(self):
        b = Bits("0b10110100")
        self.assertEqual(b[-99:4].to_bin(), "1011")

    def test_slice_reversed_bounds_empty(self):
        self.assertEqual(len(Bits("0b10110100")[5:2]), 0)

    def test_find_start_beyond_length(self):
        self.assertEqual(Bits("0b1010").find(1, start=99), -1)

    def test_count_byte_aligned(self):
        b = Bits(b"\xff\x0f")
        self.assertEqual(b.count(1), 12)
        self.assertEqual(b.count(0), 4)

    def test_count_non_aligned_excludes_tail(self):
        # 11 bits: "10110100101" -> six 1s, five 0s; phantom tail bits ignored
        b = Bits("0b10110100101")
        self.assertEqual(b.count(1), 6)
        self.assertEqual(b.count(0), 5)


class TestTailZeroOnNonAlignedOps(unittest.TestCase):
    """Invert/AND on non-byte-aligned lengths must leave the tail masked;
    verify via to_hex/to_int, which read the raw storage byte."""

    def test_invert_3bit_masks_tail(self):
        b = ~Bits("0b101")  # -> 010, storage byte 0x40
        self.assertEqual(b.to_bin(), "010")
        self.assertEqual(b.to_hex(), "40")
        self.assertEqual(b.to_int(), 0b010)

    def test_invert_11bit_masks_tail(self):
        # ~10110100101 = 01001011010 ; stored as 0x4b40 (last 5 bits zero)
        b = ~Bits("0b10110100101")
        self.assertEqual(b.to_bin(), "01001011010")
        self.assertEqual(b.to_hex(), "4b40")
        self.assertEqual(b.to_int(), 0b01001011010)

    def test_and_11bit_masks_tail(self):
        a = Bits("0b10110100101")
        c = Bits("0b11111111111")
        r = a & c  # == a, tail must stay zero
        self.assertEqual(r.to_hex(), "b4a0")
        self.assertEqual(r.to_int(), 0b10110100101)


class TestEqualityHashProtocol(unittest.TestCase):
    def test_eq_non_bits_is_false(self):
        # __eq__ returns NotImplemented -> Python falls back to identity -> False
        self.assertFalse(Bits("0b1010") == 5)
        self.assertFalse(Bits("0b1010") == "x")

    def test_ne_non_bits_is_true(self):
        self.assertTrue(Bits("0b1010") != 5)

    def test_usable_as_dict_key(self):
        d = {Bits("0b1010"): "v"}
        self.assertEqual(d[Bits("0b1010")], "v")

    def test_differently_constructed_equal_hash_equal(self):
        # bytes+nbits shrink vs ones() factory -> same value, same hash
        a = Bits(b"\xff\xff\xff", nbits=9)
        b = Bits.ones(9)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class TestRepr(unittest.TestCase):
    def test_repr_at_64_bit_boundary_is_binary_form(self):
        # nbits <= 64 -> full binary form
        b = Bits.zeros(64)
        self.assertEqual(repr(b), "Bits('0b" + "0" * 64 + "')")

    def test_repr_above_64_is_ellipsis_form(self):
        # nbits > 64 -> "<N bits, 0x<first 16 hex chars>…>"
        b = Bits(b"\xde\xad\xbe\xef" * 3)  # 96 bits
        self.assertEqual(repr(b), "Bits(<96 bits, 0xdeadbeefdeadbeef…>)")

    def test_repr_just_over_boundary(self):
        b = Bits.zeros(65)
        self.assertEqual(repr(b), "Bits(<65 bits, 0x0000000000000000…>)")


class TestNonAlignedConversions(unittest.TestCase):
    def test_to_hex_partial_last_byte_masked(self):
        # 12-bit 0xABC stored left-aligned -> 0xab 0xc0
        self.assertEqual(Bits(0xABC, nbits=12).to_hex(), "abc0")

    def test_to_bytes_partial_last_byte_masked(self):
        self.assertEqual(Bits(0xABC, nbits=12).to_bytes(), b"\xab\xc0")

    def test_to_hex_3bit_shows_masked_tail(self):
        # "0b101" -> storage byte 0xa0
        self.assertEqual(Bits("0b101").to_hex(), "a0")
        self.assertEqual(Bits("0b101").to_bytes(), b"\xa0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
