"""Test package for bite.

Single path shim: add the repo root to sys.path so `from bite import Bits`
resolves in every test module. Runs once on package import. Invoke tests with
`python -m unittest` (e.g. `python -m unittest discover -s tests`) from the repo
root, or `python -m unittest tests.test_streaming` for one module.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
