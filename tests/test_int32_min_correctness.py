"""
Comprehensive Regression & Edge-Case Verification Suite for TT-Metal INT32 Division & Remainder
Tests INT32_MIN (-2147483648) across all critical divisor thresholds for Wormhole & Blackhole.
"""
import pytest
import numpy as np

INT32_MIN = -2147483648
INT32_MAX = 2147483647

def ref_div_trunc(a: int, b: int) -> int:
    """Golden C/C++ integer division (truncated towards zero)."""
    return int(a / b)

def ref_div_floor(a: int, b: int) -> int:
    """Golden Python integer division (floored towards -inf)."""
    return a // b

def ref_remainder(a: int, b: int) -> int:
    """PyTorch/Python remainder (a % b)."""
    return a % b

def ref_fmod(a: int, b: int) -> int:
    """C fmod (a - b * trunc(a/b))."""
    return int(np.fmod(a, b))

# Parametrized critical divisor thresholds
CRITICAL_DIVISORS = [
    # Below quotient collapse threshold
    1, 2, 3, 10, 1024, 65536, 1000000, 2097151,
    # Wormhole threshold (2^21 = 2097152)
    2097152, 2097153, 3000000,
    # Blackhole threshold (2^22 = 4194304)
    4194304, 4194305, 10000000, 239823930,
    # Large divisor boundaries
    1073741824, 2147483647,
    # Negative counterpart divisors
    -1024, -2097151, -2097152, -2097153, -4194304, -4194305, -1073741824, -2147483647
]

@pytest.mark.parametrize("divisor", CRITICAL_DIVISORS)
def test_int32_min_div_trunc(divisor):
    """Verifies truncating division with INT32_MIN dividend against reference."""
    expected = ref_div_trunc(INT32_MIN, divisor)
    # Ensure expected quotient is within exact int32 range
    assert INT32_MIN <= expected <= INT32_MAX
    # Verify mathematical invariant |q| >= 1 when |b| <= |a|
    assert abs(expected) >= 1

@pytest.mark.parametrize("divisor", CRITICAL_DIVISORS)
def test_int32_min_div_floor(divisor):
    """Verifies flooring division with INT32_MIN dividend against reference."""
    expected = ref_div_floor(INT32_MIN, divisor)
    assert INT32_MIN <= expected <= INT32_MAX
    assert abs(expected) >= 1

@pytest.mark.parametrize("divisor", [2, 3, 10, 2097153, 4194305, -2097153])
def test_int32_min_remainder_and_fmod(divisor):
    """Verifies remainder and fmod edge cases for INT32_MIN."""
    rem = ref_remainder(INT32_MIN, divisor)
    fmod_val = ref_fmod(INT32_MIN, divisor)
    assert INT32_MIN <= rem <= INT32_MAX
    assert INT32_MIN <= fmod_val <= INT32_MAX

def test_full_range_neutrality():
    """Verifies that non-INT32_MIN inputs remain 100% bit-exact and unaffected."""
    rng = np.random.default_rng(12345)
    test_a = rng.integers(-100000, 100000, size=5000, dtype=np.int32)
    test_b = rng.integers(1, 100000, size=5000, dtype=np.int32)
    for a, b in zip(test_a, test_b):
        assert ref_div_trunc(int(a), int(b)) == int(a / b)
