#!/usr/bin/env python3
"""Run the supported test suite with a clear dependency error."""

from __future__ import annotations

import sys

try:
    import pytest
except ImportError:
    print("pytest is required. Install development dependencies with: pip install -e '.[dev,server]'", file=sys.stderr)
    raise SystemExit(2)

raise SystemExit(pytest.main(["-q", *sys.argv[1:]]))
