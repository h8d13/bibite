"""float754.py

Take apart an IEEE-754 single-precision float and put it back together.

A 32-bit float is three packed fields, MSB-first:

    sign      1 bit
    exponent  8 bits   biased by 127
    mantissa 23 bits   the fraction after an implicit leading 1 (when normalized)

struct.pack(">f", x) gives the 4 raw big-endian bytes. Wrapping them in Bits
lets us slice the three fields out by position and read each as an int, instead
of juggling shifts and masks against a 32-bit word.

Run from the repo root: python -m examples.float754
"""
import math
import struct

from bite import Bits


def dissect(x: float) -> tuple[int, int, int]:
    """Return (sign, exponent, mantissa) for a single-precision float."""
    b = Bits(struct.pack(">f", x))   # 32 bits, MSB-first
    sign = b[0:1].to_int()
    exponent = b[1:9].to_int()
    mantissa = b[9:32].to_int()
    return sign, exponent, mantissa


def reconstruct(sign: int, exponent: int, mantissa: int) -> float:
    """Rebuild the float from its fields. Handles zero, normal, denormal, inf
    and nan."""
    s = -1.0 if sign else 1.0
    if exponent == 0xFF:
        return float("nan") if mantissa else s * float("inf")
    if exponent == 0:
        # denormal (and zero): no implicit leading 1, fixed exponent of -126
        return s * 2.0 ** -126 * (mantissa / 2 ** 23)
    # normal: implicit leading 1, exponent unbiased by 127
    return s * 2.0 ** (exponent - 127) * (1 + mantissa / 2 ** 23)


def main() -> None:
    samples = [1.0, 0.15625, -3.5, 0.1, -0.0]
    for x in samples:
        sign, exponent, mantissa = dissect(x)
        rebuilt = reconstruct(sign, exponent, mantissa)

        print(f"{x!r}")
        print(f"  sign     : {sign:01b}  ({sign})")
        print(f"  exponent : {exponent:08b}  ({exponent}, unbiased {exponent - 127})")
        print(f"  mantissa : {mantissa:023b}  ({mantissa})")
        print(f"  rebuilt  : {rebuilt!r}")

        # The rebuilt value must match what struct decodes from the same bits.
        ref = struct.unpack(">f", struct.pack(">f", x))[0]
        if rebuilt != rebuilt:           # nan never equals itself
            assert ref != ref, "expected nan"
        else:
            # copysign also pins the sign, since 0.0 == -0.0 would otherwise pass
            assert rebuilt == ref and math.copysign(1, rebuilt) == math.copysign(1, ref), \
                f"mismatch: {rebuilt!r} != {ref!r}"
        print("  round-trips against struct.unpack\n")


if __name__ == "__main__":
    main()
