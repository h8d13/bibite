"""Integration: a realistic protocol-style roundtrip for bite."""

import unittest

from bite import Bits


class TestProtocolRoundtrip(unittest.TestCase):
    """Encode a tiny packet, decode it back, byte-align included."""

    def test_packet_roundtrip(self):
        # 3-bit version, 5-bit type, 16-bit payload length, 4-bit flags, then pad
        b = Bits()
        b.write(0b101, 3)
        b.write(0b10110, 5)
        b.write(0xBEEF, 16)
        b.write(0b1001, 4)
        b.pad_to_byte()
        self.assertEqual(len(b) % 8, 0)

        # Decode
        b.rewind()
        self.assertEqual(b.read(3), 0b101)
        self.assertEqual(b.read(5), 0b10110)
        self.assertEqual(b.read(16), 0xBEEF)
        self.assertEqual(b.read(4), 0b1001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
