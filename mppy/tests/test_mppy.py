"""
Unit and regression test for the mppy package.
"""

# Import package, test suite, and other packages as needed
import mppy
import pytest
import sys


def test_mppy_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "mppy" in sys.modules
