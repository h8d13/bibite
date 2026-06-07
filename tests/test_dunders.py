"""Dunder / protocol-method tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
