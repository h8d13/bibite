"""Analysis tests for bite (count / reversed_bits / find)."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
