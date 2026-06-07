"""Streaming read/write tests for bite."""

import unittest

from bite import Bits


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
