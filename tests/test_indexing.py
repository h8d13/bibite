"""Indexing and slicing tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
