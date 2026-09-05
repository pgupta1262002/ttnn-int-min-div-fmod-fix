// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Blackhole: calculate_div_int32_body with INT32_MIN clamp and residual sign protection
template <bool is_floor>
inline void calculate_div_int32_body_blackhole() {
    sfpi::vInt b = sfpi::dst_reg[0];
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; } // L33: Guarded
    v_endif;

    sfpi::vInt a_s = sfpi::dst_reg[1];
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(a_s), sfpi::RoundMode::Nearest);
    v_if(a_f < 0.0f) { a_f = 0x1.0p31f; } // L47: Guarded
    v_endif;

    sfpi::vFloat inv_b_f = sfpi::sfparecip(b_f);
    // Halley refinement step
    sfpi::vFloat e = -inv_b_f * b_f + 1.0f;
    e = e * e + e;
    inv_b_f = e * inv_b_f + inv_b_f;

    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0; // Mantissa alignment 2^33
    sfpi::vInt q_m = sfpi::exman(q_f);
    sfpi::vInt q = q_m << 10;
    sfpi::vInt qb = q * sfpi::abs(b);

    sfpi::vInt r = sfpi::abs(a_s) - qb;
    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r), sfpi::RoundMode::Nearest);

    // =========================================================================
    // FIX FOR ISSUE #55502 / #51476:
    // INT32_MIN (0x80000000) residual magnitude and sign guard
    // =========================================================================
    sfpi::vInt r_sign = r;
    v_if(r_f < 0.0f) {
        r_f = 0x1.0p31f;
        r_sign = 0;
    }
    v_endif;

    sfpi::vFloat cor_f = r_f * inv_b_f;
    sfpi::vInt cor = sfpi::convert<sfpi::vUInt16>(cor_f);
    sfpi::vInt tmp = cor * sfpi::abs(b);

    v_if(r_sign < 0) {
        r = r + tmp;
        q = q - cor;
    }
    v_else {
        r = r - tmp;
        q = q + cor;
    }
    v_endif;

    // One-ULP correction
    v_if(r < 0 && (r - 1) < 0) {
        q = q - 1;
        r = r + sfpi::abs(b);
    }
    v_elseif(r >= sfpi::abs(b)) {
        q = q + 1;
        r = r - sfpi::abs(b);
    }
    v_endif;

    sfpi::vInt is_neg = (a_s ^ b) < 0;
    v_if(is_neg) {
        q = -q;
    }
    v_endif;

    if constexpr (is_floor) {
        v_if(is_neg && r != 0) {
            q = q - 1;
        }
        v_endif;
    }

    sfpi::dst_reg[0] = q;
}

} // namespace sfpu
} // namespace ckernel
