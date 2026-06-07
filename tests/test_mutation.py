"""Mutation tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
