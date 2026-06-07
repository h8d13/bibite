"""Regression tests for defects found in the /improve critic-gated review.

One test class per fixed case. Run from repo root:
    python -m unittest tests.test_fixes
    python -m unittest discover -s tests
"""

import unittest

from bite import Bits


class TestIntOverflowTruncates(unittest.TestCase):
    """int source wider than nbits truncated to low n bits, not OverflowError."""

    def test_value_exceeds_nbits(self):
        self.assertEqual(Bits(0xFF, nbits=4), Bits("0b1111"))
        self.assertEqual(Bits(0xFF, nbits=4).to_int(), 0xF)

    def test_negative_int_with_nbits_is_twos_complement(self):
        self.assertEqual(Bits(-1, nbits=8).to_bin(), "11111111")
        self.assertEqual(Bits(-5, nbits=4), Bits("0b1011"))

    def test_n_zero(self):
        self.assertEqual(len(Bits(5, nbits=0)), 0)


class TestNbitsShrinkDropsTrailingBytes(unittest.TestCase):
    """nbits smaller than data: excess whole bytes dropped, tail masked, so
    value-equal Bits compare == and hash equal regardless of construction."""

    def test_bytes_branch(self):
        b = Bits(b"\xff\xff\xff", nbits=9)
        self.assertEqual(b.count(1), 9)
        self.assertEqual(b, Bits.ones(9))
        self.assertEqual(hash(b), hash(Bits.ones(9)))
        self.assertEqual(b.to_int(), 0x1FF)

    def test_copy_branch(self):
        b = Bits(Bits(b"\xff\xff\xff"), nbits=4)
        self.assertEqual(b.count(1), 4)
        self.assertEqual(b, Bits("0b1111"))

    def test_str_branch(self):
        b = Bits("0xff", nbits=4)
        self.assertEqual(b.count(1), 4)
        self.assertEqual(b, Bits("0b1111"))


class TestNbitsExceedsDataRejected(unittest.TestCase):
    """nbits larger than supplied data raises instead of building a corrupt
    object that crashes on read."""

    def test_bytes_branch(self):
        with self.assertRaises(ValueError):
            Bits(b"\xff", nbits=12)

    def test_copy_branch(self):
        with self.assertRaises(ValueError):
            Bits(Bits(b"\xff"), nbits=12)

    def test_str_branch(self):
        with self.assertRaises(ValueError):
            Bits("0b1010", nbits=12)


class TestNegativeIntNeedsWidth(unittest.TestCase):
    """negative int with no nbits raises rather than silently mangling."""

    def test_raises(self):
        with self.assertRaises(ValueError):
            Bits(-5)
        with self.assertRaises(ValueError):
            Bits(-1)


class TestLittleEndianToIntSymmetry(unittest.TestCase):
    """to_int('little') rejects non-byte-aligned widths, matching from_int."""

    def test_unaligned_raises(self):
        with self.assertRaises(ValueError):
            Bits(0xABC, nbits=12).to_int("little")

    def test_aligned_roundtrips(self):
        b = Bits.from_int(0x1234, nbits=16, endian="little")
        self.assertEqual(b.to_int("little"), 0x1234)


class TestReadBytesNegativeMessage(unittest.TestCase):
    """read_bytes rejects negative n with the unscaled argument in the message."""

    def test_unscaled_value(self):
        with self.assertRaises(ValueError) as ctx:
            Bits(b"\xff").read_bytes(-1)
        self.assertIn("-1", str(ctx.exception))
        self.assertNotIn("-8", str(ctx.exception))


class TestParseStrNbitsRoutedThroughNormalize(unittest.TestCase):
    """The str branch now reconciles parsed data against an explicit nbits via
    _normalize (shrink/reject/mask), instead of bypassing it.

    Base-selection contract is unchanged: bare all-0/1 is BINARY when nbits is
    given or length is odd, else HEX (see tests.py TestConstruction).
    """

    def test_str_nbits_shrinks_and_masks(self):
        # bare even binary with nbits -> binary (contract), then width applied
        self.assertEqual(Bits("1010", nbits=4), Bits.from_bits([1, 0, 1, 0]))
        # hex literal narrowed: storage truncated, tail masked, eq/hash hold
        b = Bits("0xff", nbits=4)
        self.assertEqual(b.count(1), 4)
        self.assertEqual(b, Bits("0b1111"))
        self.assertEqual(hash(b), hash(Bits("0b1111")))

    def test_str_nbits_exceeds_parsed_data_raises(self):
        with self.assertRaises(ValueError):
            Bits("0b1010", nbits=12)
        with self.assertRaises(ValueError):
            Bits("0xf", nbits=20)

    def test_bare_odd_binary_unaffected(self):
        self.assertEqual(Bits("101"), Bits("0b101"))
        self.assertEqual(len(Bits("101")), 3)


if __name__ == "__main__":
    unittest.main()
