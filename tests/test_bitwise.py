"""Bitwise operator tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
