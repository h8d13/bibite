"""Conversion tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
