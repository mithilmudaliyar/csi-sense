"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A real ESP32-CSI-Tool line taken verbatim from the vendored example
# (firmware/esp32-csi-tool/python_utils/example_csi.csv). len=384 here.
REAL_EXAMPLE_LINE = (
    "CSI_DATA,AP,3C:71:BF:6D:2A:78,-73,11,1,0,1,1,1,0,0,0,0,-93,0,1,1,"
    "80272146,0,101,0,0,80.363225,384,[101 -48 5 0 0 0 0 0 0 0 5 2 23 12 "
    "25 13 27 16 28 19 27 20 24 22 22 23 20 24 19 25 18 25 20 27 20 27 18 "
    "26 16 26 16 25 16 25 14 23 12 21 12 21 12 20 14 19 15 18 14 17 16 17 "
    "18 16 18 14 10 6 20 11 20 10 22 10 22 10 23 10 25 11 25 10 24 8 25 7 "
    "27 5 27 5 26 6 26 7 27 8 27 7 28 6 29 5 27 4 25 3 25 3 26 4 26 4 26 3 "
    "26 3 25 3 24 1 5 0 0 0 0 0 0 0 0 0 ]"
)


@pytest.fixture
def real_example_line() -> str:
    return REAL_EXAMPLE_LINE


@pytest.fixture
def default_config():
    from pipeline.config import DEFAULT_CONFIG

    return DEFAULT_CONFIG
