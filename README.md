# TT-Metal INT32_MIN Correctness Fix (Issue #55502 / #51476)

## Summary
Resolves silent numerical collapse for `INT32_MIN` (`-2147483648`) in `ttnn.div` (`trunc`/`floor`), `ttnn.remainder`, `ttnn.fmod`, and scalar promotion on both **Wormhole B0** and **Blackhole** architectures.

### Root Cause Analysis
During quotient estimation and residual correction (`r = |a| - q*|b|`), when `a == INT32_MIN`, `r` takes the bit pattern `0x80000000` (representing `2^31`).
1. Converting `0x80000000` via `sfpi::convert<sfpi::vFloat>(sfpi::abs(r))` treated the value as signed magnitude, collapsing it to `-0.0f`.
2. The sign condition `v_if(r >= 0)` on Wormhole (or `v_if(r < 0)` on Blackhole) interpreted `0x80000000` as negative, applying the residual correction in the wrong direction.
3. For divisors `|b| >= 2^21` (Wormhole) and `|b| >= 2^22` (Blackhole), this caused the entire quotient to collapse to `±1` (up to 99.90% relative error).

### The Fix
1. **Magnitude Guard:** Predicate `v_if(r_f < 0.0f) { r_f = 0x1.0p31f; }` restores the true `2^31` residual magnitude.
2. **Sign Guard:** `sfpi::vInt r_sign = r;` with `v_if(r_f < 0.0f) { r_sign = 0; }` ensures the positive `2^31` remainder is correctly treated as non-negative in direction branches.
3. **Regression Neutrality:** Validated across all threshold boundaries with zero regression on non-`INT32_MIN` inputs.

### Verification
Run the verification suite:
```bash
pytest tests/test_int32_min_correctness.py -v
```
