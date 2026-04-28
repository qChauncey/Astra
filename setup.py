"""Minimal setup.py shim for PEP 660 editable install compatibility.

All configuration lives in pyproject.toml. This file exists only so that
older pip versions or environments that lack full PEP 660 support can
still perform editable installs via the legacy fallback path.
"""

from setuptools import setup

setup()