"""Pytest configuration ensuring repo root is on sys.path."""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
