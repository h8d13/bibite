"""Construction and factory tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
