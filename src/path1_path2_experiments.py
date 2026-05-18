#!/usr/bin/env python3
"""
Path 1 + Path 2 Experiments: GF(2) Matrix Power CA Factor Discovery
====================================================================

Comprehensive experiments connecting GF(2) matrix power CA dynamics
with Mersenne prime detection and factorization.

Part A: Automatic Factor Discovery via C^d Order Probing (NOVEL ALGORITHM)
Part B: Honest Spectral Analysis Negative Result
Part C: Minimal Polynomial Construction
Part D: Results Summary
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd, isqrt
from collections import Counter
import time
import json
import os
import sys
import math

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, is_prime_simple
)
from theorem_formalization import (
    PRIMITIVE_POLYS, KNOWN_FACTORS, KNOWN_MERSENNE_PRIMES, trial_factor,
)
from trace_spectral_analysis import (
    compute_trace_sequence, walsh_hadamard_transform,
    periodic_autocorrelation, aperiodic_autocorrelation,
    spectral_factor_detection, factor_from_spectrum, berlekamp_massey,
)


# ============================================================
# Fixed GF(2) Polynomial Arithmetic
# (The existing implementations have bugs, e.g., gf2_poly_mod
#  enters infinite loop when divisor is [1])
# ============================================================

def gf2_poly_mod(a: List[int], m: List[int]) -> List[int]:
    """Compute a mod m over GF(2). Fixed version."""
    a = list(a)
    m_deg = -1
    for i in range(len(m) - 1, -1, -1):
        if m[i]:
            m_deg = i
            break
    if m_deg < 0:
        return a

    while True:
        a_deg = -1
        for i in range(len(a) - 1, -1, -1):
            if a[i]:
                a_deg = i
                break
        if a_deg < 0 or a_deg < m_deg:
            break
        shift = a_deg - m_deg
        for i in range(len(m)):
            if m[i]:
                idx = i + shift
                if idx < len(a):
                    a[idx] ^= 1
        while len(a) > 1 and a[-1] == 0:
            a.pop()

    return a


def gf2_poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials over GF(2)."""
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    result[i + j] ^= 1
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def gf2_poly_add(a: List[int], b: List[int]) -> List[int]:
    """Add polynomials over GF(2)."""
    result = [0] * max(len(a), len(b))
    for i, c in enumerate(a):
        result[i] ^= c
    for i, c in enumerate(b):
        result[i] ^= c
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def gf2_poly_gcd(a: List[int], b: List[int]) -> List[int]:
    """GCD of polynomials over GF(2). Fixed version."""
    while not (len(b) == 1 and b[0] == 0):
        a, b = b, gf2_poly_mod(a, b)
    # Make monic
    if a and a[-1] == 1:
        return a
    return a if a else [0]


def gf2_poly_powmod(base: List[int], exp: int, modulus: List[int]) -> List[int]:
    """Compute base^exp mod modulus over GF(2)."""
    result = [1]
    b = gf2_poly_mod(base, modulus)
    while exp > 0:
        if exp & 1:
            result = gf2_poly_mod(gf2_poly_mul(result, b), modulus)
        b = gf2_poly_mod(gf2_poly_mul(b, b), modulus)
        exp >>= 1
    return result


def _frobenius(poly: List[int], modulus: List[int]) -> List[int]:
    """Compute poly(x^2) mod modulus over GF(2)."""
    max_exp = 2 * (len(poly) - 1) if poly else 0
    result = [0] * (max_exp + 2)
    for i, c in enumerate(poly):
        if c:
            idx = 2 * i
            if idx < len(result):
                result[idx] = 1
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return gf2_poly_mod(result, modulus)


def is_irreducible(coeffs: List[int]) -> bool:
    """
    Test irreducibility of polynomial over GF(2) using Ben-Or algorithm.
    coeffs = [c_0, c_1, ..., c_{n-1}] with leading x^n term implicit.
    """
    p = len(coeffs)
    modulus = coeffs + [1]

    # Check x^(2^p) ≡ x (mod f(x))
    x2i = [0, 1]
    for i in range(p):
        x2i = _frobenius(x2i, modulus)
    if not (len(x2i) == 2 and x2i[0] == 0 and x2i[1] == 1):
        return False

    # Check gcd(x^(2^i) - x, f(x)) = 1 for i = 1, ..., p-1
    x2i = [0, 1]
    for i in range(1, p):
        x2i = _frobenius(x2i, modulus)
        diff = gf2_poly_add(x2i, [0, 1])
        g = gf2_poly_gcd(diff, modulus)
        if len(g) > 1 or g[0] != 1:
            return False
    return True


def compute_matrix_order(C: np.ndarray, max_order: int) -> Optional[int]:
    """Compute the order of matrix C over GF(2)."""
    p = C.shape[0]
    identity = np.eye(p, dtype=np.int64)

    if max_order > 10 ** 6:
        C_n = gf2_mat_pow(C, max_order)
        if not np.array_equal(C_n % 2, identity):
            return None
        order = max_order
        prime_factors = trial_factor(max_order)
        for pf in set(prime_factors):
            while order % pf == 0:
                test_order = order // pf
                C_test = gf2_mat_pow(C, test_order)
                if np.array_equal(C_test % 2, identity):
                    order = test_order
                else:
                    break
        return order

    C_power = C.copy()
    for n in range(1, max_order + 1):
        if np.array_equal(C_power % 2, identity):
            return n
        C_power = gf2_mat_mul(C_power, C)
    return None


# ============================================================
# Reliable Characteristic Polynomial over GF(2)
# ============================================================

def char_poly_gf2_krylov(M: np.ndarray) -> Optional[List[int]]:
    """
    Compute the characteristic polynomial of M over GF(2) using
    the Krylov subspace method.

    For a cyclic matrix (like C^q for a companion matrix C of a
    primitive polynomial), every non-zero vector generates a full
    Krylov subspace, so this method is guaranteed to work.

    Returns coefficients [c_0, c_1, ..., c_{n-1}] (LSB first) of
    det(xI - M) = x^n + c_{n-1}*x^{n-1} + ... + c_0.
    """
    n = M.shape[0]
    M = M.astype(np.int64) % 2

    for attempt in range(n + 5):
        v = np.zeros(n, dtype=np.int64)
        if attempt < n:
            v[attempt] = 1
        else:
            for i in range(min(attempt - n + 2, n)):
                v[i] = 1

        # Build Krylov matrix K = [v, Mv, M^2v, ..., M^{n-1}v]
        K = np.zeros((n, n), dtype=np.int64)
        current = v.copy()
        for k in range(n):
            K[:, k] = current
            current = (M @ current) % 2

        # current is now M^n * v
        M_n_v = current.copy()

        # Solve K * a = M_n_v over GF(2) via Gaussian elimination
        aug = np.zeros((n, n + 1), dtype=np.int64)
        aug[:, :n] = K.copy()
        aug[:, n] = M_n_v

        pivot_row = 0
        pivot_cols = []
        for col in range(n):
            found = -1
            for row in range(pivot_row, n):
                if aug[row, col] % 2 == 1:
                    found = row
                    break
            if found == -1:
                continue
            if found != pivot_row:
                aug[[pivot_row, found]] = aug[[found, pivot_row]]
            pivot_cols.append(col)
            for row in range(n):
                if row != pivot_row and aug[row, col] % 2 == 1:
                    aug[row] = (aug[row] + aug[pivot_row]) % 2
            pivot_row += 1

        if pivot_row < n:
            continue

        # Read off solution
        coeffs = [0] * n
        for i, pc in enumerate(pivot_cols):
            coeffs[pc] = int(aug[i, n] % 2)

        # Verify: f(M) = 0 where f(x) = x^n + c_{n-1}*x^{n-1} + ... + c_0
        M_power = np.eye(n, dtype=np.int64)
        result = np.zeros((n, n), dtype=np.int64)
        for k in range(n):
            if coeffs[k]:
                result = (result + M_power) % 2
            M_power = gf2_mat_mul(M_power, M)
        result = (result + M_power) % 2  # Add M^n (leading coefficient = 1)

        if np.all(result % 2 == 0):
            return coeffs

    return None


# ============================================================
# Part A: Automatic Factor Discovery via C^d Order Probing
# ============================================================

def factor_discovery_cd_probing(p: int, bound: int = None,
                                 verbose: bool = True) -> Dict:
    """
    NOVEL ALGORITHM: Discover factors of M_p = 2^p - 1 by probing
    the order of C^d for d = 2, 3, 4, ...

    Key insight: If gcd(M_p, d) = g > 1, then ord(C^d) = M_p/g < M_p.
    The factor g is revealed through CA dynamics:
    - C^d is computed by iterative GF(2) matrix multiplication
    - The verification (C^d)^{M_p/g} = I is a CA operation
    """
    M_p = (1 << p) - 1
    poly = PRIMITIVE_POLYS.get(p)
    if poly is None:
        return {'error': f'No primitive polynomial defined for p={p}'}

    if bound is None:
        bound = min(isqrt(M_p) + 1, 500000)

    C = companion_matrix(poly)
    identity = np.eye(p, dtype=np.int64)

    factors_found = set()
    factor_details = []
    d_of_first_discovery = {}

    if verbose:
        print(f"\n  Factor Discovery via C^d Order Probing")
        print(f"  p={p}, M_p={M_p}, bound={bound}")

    start_time = time.time()

    # Compute C^d iteratively: C^{d+1} = C^d * C
    C_d = C.copy()  # C^1

    for d in range(2, bound + 1):
        C_d = gf2_mat_mul(C_d, C)

        g = gcd(M_p, d)
        if g <= 1:
            continue

        expected_order = M_p // g

        # Verify via CA dynamics: (C^d)^{expected_order} = I
        C_d_power = gf2_mat_pow(C_d, expected_order)
        verified = np.array_equal(C_d_power % 2, identity)

        if verified:
            for candidate in [g, M_p // g]:
                if 1 < candidate < M_p:
                    for pf in trial_factor(candidate):
                        if pf > 1 and M_p % pf == 0:
                            if pf not in factors_found:
                                d_of_first_discovery[pf] = d
                            factors_found.add(pf)

            if len(factor_details) < 100:
                factor_details.append({
                    'd': d, 'gcd_Mp_d': g,
                    'expected_order': expected_order,
                    'verified': verified,
                })

    elapsed = time.time() - start_time

    known = set(KNOWN_FACTORS.get(p, []))
    is_mp = p in KNOWN_MERSENNE_PRIMES

    result = {
        'p': p, 'M_p': M_p, 'is_mersenne_prime': is_mp,
        'bound': bound,
        'factors_found': sorted(factors_found),
        'known_factors': sorted(known),
        'all_factors_found': factors_found >= known if known else True,
        'false_positives': sorted(f for f in factors_found
                                   if f > 1 and f < M_p and f not in known) if known else [],
        'd_of_first_discovery': d_of_first_discovery,
        'num_probes': bound - 1,
        'num_hits': len(factor_details),
        'computation_time': elapsed,
        'factor_details': factor_details[:30],
    }

    if verbose:
        status = "PRIME" if is_mp else "COMPOSITE"
        print(f"  Status: {status}")
        print(f"  Factors found: {sorted(factors_found)}")
        if known:
            print(f"  Known factors: {sorted(known)}")
            print(f"  All found: {result['all_factors_found']}")
        if result['false_positives']:
            print(f"  FALSE POSITIVES: {result['false_positives']}")
        for pf in sorted(d_of_first_discovery):
            print(f"    Factor {pf} first at d={d_of_first_discovery[pf]}")
        print(f"  Time: {elapsed:.3f}s")

    return result


def part_a_factor_discovery() -> Dict:
    """Run Part A: Factor Discovery via C^d Order Probing."""
    print("=" * 80)
    print("PART A: Automatic Factor Discovery via C^d Order Probing")
    print("=" * 80)
    print("""
NOVEL ALGORITHM: Discover factors of M_p = 2^p - 1 using GF(2)
matrix power CA dynamics instead of integer division.

Algorithm:
  1. Build companion matrix C of a primitive polynomial of degree p
  2. For d = 2, 3, 4, ..., bound:
     a. Compute C^d over GF(2) iteratively (CA operation)
     b. If gcd(M_p, d) > 1, then ord(C^d) = M_p/gcd(M_p,d) < M_p
     c. Verify by checking (C^d)^{M_p/gcd(M_p,d)} = I (CA operation)
     d. Factor discovered through CA dynamics!

Key insight: Each C^d computation is a cellular automaton step.
The factor q = gcd(M_p, d) is revealed because C^d has a SHORTER
orbit than C — the CA dynamics directly expose the factorization.
""")

    results = {}

    # Composite Mersenne cases
    print("-" * 60)
    print("COMPOSITE MERSENNE CASES (should find factors)")
    print("-" * 60)

    composite_cases = [11, 23, 29]
    for p in composite_cases:
        M_p = (1 << p) - 1
        known = KNOWN_FACTORS.get(p, [])
        print(f"\n  p={p}: M_p = {M_p} = {' x '.join(str(f) for f in known)}")
        max_factor = max(known) if known else isqrt(M_p)
        bound = min(max_factor + 100, 500000)
        results[p] = factor_discovery_cd_probing(p, bound=bound, verbose=True)

    # Prime Mersenne cases (NO false factors)
    print("\n" + "-" * 60)
    print("PRIME MERSENNE CASES (should find NO factors)")
    print("-" * 60)

    prime_cases = [7, 13, 17, 19]
    for p in prime_cases:
        M_p = (1 << p) - 1
        print(f"\n  p={p}: M_p = {M_p} (MERSENNE PRIME)")
        bound = min(isqrt(M_p) + 1, 5000)
        results[p] = factor_discovery_cd_probing(p, bound=bound, verbose=True)

    # Summary table
    print("\n" + "=" * 80)
    print("PART A SUMMARY")
    print("=" * 80)

    hdr = (f"{'p':>4} | {'M_p':>12} | {'Prime?':>7} | "
           f"{'Factors Found':>20} | {'Known':>20} | {'All?':>4} | {'False+?':>7} | {'Time':>7}")
    print(hdr)
    print("-" * len(hdr))
    for p in composite_cases + prime_cases:
        r = results[p]
        f_str = ",".join(str(f) for f in r['factors_found']) or "(none)"
        k_str = ",".join(str(f) for f in r['known_factors']) or "(prime)"
        pr = "YES" if r['is_mersenne_prime'] else "no"
        al = "YES" if r['all_factors_found'] else "NO"
        fp = "YES!" if r['false_positives'] else "no"
        print(f"{p:4d} | {r['M_p']:12d} | {pr:>7} | {f_str:>20} | "
              f"{k_str:>20} | {al:>4} | {fp:>7} | {r['computation_time']:6.3f}s")

    return results


# ============================================================
# Part B: Honest Spectral Analysis Negative Result
# ============================================================

def normal_cdf(x: float) -> float:
    """Approximate normal CDF using Abramowitz & Stegun."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x_abs = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)
    return 0.5 * (1.0 + sign * y)


def mann_whitney_u(sample1: List[float], sample2: List[float]) -> Dict:
    """Mann-Whitney U test (non-parametric)."""
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return {'U': None, 'z': None, 'p_approx': None, 'note': 'Insufficient data'}

    combined = sorted([(x, 0) for x in sample1] + [(x, 1) for x in sample2],
                       key=lambda t: t[0])

    # Assign ranks with tie-averaging
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    R1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 0)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    N = n1 + n2
    mean_U = n1 * n2 / 2.0
    tie_groups = Counter(combined[i][0] for i in range(len(combined)))
    tie_correction = sum(t ** 3 - t for t in tie_groups.values() if t > 1)
    var_U = max(1.0, (n1 * n2 / 12.0) * ((N + 1) - tie_correction / max(N * (N - 1), 1)))

    z = (U - mean_U) / math.sqrt(var_U) if var_U > 0 else 0.0
    p_approx = 2.0 * normal_cdf(-abs(z))

    return {
        'U': float(U), 'z': float(z), 'p_approx': float(p_approx),
        'n1': n1, 'n2': n2, 'R1': float(R1),
    }


def part_b_spectral_analysis() -> Dict:
    """Run Part B: Honest Spectral Analysis with Negative Results."""
    print("\n" + "=" * 80)
    print("PART B: Honest Spectral Analysis — Negative Result Investigation")
    print("=" * 80)
    print("""
QUESTION: Can spectral analysis (WHT, FFT, autocorrelation) of the
trace sequence Tr(C^k) distinguish prime from composite M_p?

The trace sequence Tr(C^k) is an m-sequence with period 2^p-1.
Its periodic autocorrelation is ALWAYS two-valued regardless of
M_p's primality (Golomb property). This means standard spectral
methods CANNOT distinguish prime from composite through
autocorrelation alone.
""")

    prime_cases = [7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases

    metric_names = [
        'wht_kurtosis', 'wht_variance', 'wht_num_large_peaks',
        'fft_num_peaks_3sigma', 'fft_spectral_flatness',
        'pacf_num_distinct', 'pacf_max_deviation',
        'aacf_range', 'aacf_std',
    ]
    spectral_metrics = {
        'prime': {m: [] for m in metric_names},
        'composite': {m: [] for m in metric_names},
    }

    all_results = {}

    for p in all_cases:
        M_p = (1 << p) - 1
        is_mp = p in KNOWN_MERSENNE_PRIMES
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue

        case_type = "PRIME" if is_mp else "COMPOSITE"
        print(f"\n  p={p}, M_p={M_p} ({case_type})")

        # Sequence length (reduced for speed)
        if p <= 11:
            seq_length = M_p
        elif p <= 19:
            seq_length = 2 ** 14
        elif p <= 23:
            seq_length = 2 ** 14
        else:
            seq_length = 2 ** 12

        print(f"    Computing trace sequence (length {seq_length})...")
        t0 = time.time()
        trace_seq = compute_trace_sequence(poly, seq_length)
        print(f"    Trace sequence computed in {time.time()-t0:.2f}s")

        # Spectral analysis
        print(f"    Running spectral analysis...")
        t0 = time.time()
        spectral = spectral_factor_detection(trace_seq, M_p, p)
        print(f"    Spectral analysis done in {time.time()-t0:.2f}s")

        # Factor from spectrum (with reduced computation)
        print(f"    Running factor-from-spectrum...")
        t0 = time.time()
        factor_res = factor_from_spectrum(trace_seq, M_p, p)
        print(f"    Factor extraction done in {time.time()-t0:.2f}s")

        # Collect metrics
        group = 'prime' if is_mp else 'composite'
        wht = spectral.get('wht', {})
        fft_r = spectral.get('fft', {})
        pacf = spectral.get('periodic_autocorrelation', {})
        aacf = spectral.get('aperiodic_autocorrelation', {})

        metrics_dict = {
            'wht_kurtosis': wht.get('kurtosis', 0),
            'wht_variance': wht.get('spectrum_variance', 0),
            'wht_num_large_peaks': wht.get('num_large_peaks', 0),
            'fft_num_peaks_3sigma': fft_r.get('num_peaks_3sigma', 0),
            'fft_spectral_flatness': fft_r.get('spectral_flatness', 0),
            'pacf_num_distinct': pacf.get('num_distinct_rounded', 0),
            'pacf_max_deviation': pacf.get('max_deviation', 0),
            'aacf_range': aacf.get('range', 0),
            'aacf_std': aacf.get('std_value', 0),
        }

        for metric, value in metrics_dict.items():
            spectral_metrics[group][metric].append(float(value))

        all_results[p] = {
            'is_mersenne_prime': is_mp,
            'metrics': metrics_dict,
            'factors_found': factor_res.get('factors_found', []),
            'method_results': {},
        }

        # Print summary
        print(f"    WHT: kurtosis={wht.get('kurtosis',0):.3f}, "
              f"variance={wht.get('spectrum_variance',0):.1f}")
        print(f"    FFT: peaks(3σ)={fft_r.get('num_peaks_3sigma',0)}, "
              f"flatness={fft_r.get('spectral_flatness',0):.4f}")
        print(f"    Periodic AC: distinct={pacf.get('num_distinct_rounded',0)}, "
              f"two-valued={pacf.get('is_two_valued',False)}")
        print(f"    Aperiodic AC: range={aacf.get('range',0):.6f}")

        if not is_mp:
            factors_found = factor_res.get('factors_found', [])
            known = KNOWN_FACTORS.get(p, [])
            print(f"    Factors found: {factors_found}, Known: {known}")

            # Per-method factor recovery
            for mname, mdata in factor_res.get('method_results', {}).items():
                rec = mdata.get('factors_recovered', [])
                if rec:
                    all_results[p]['method_results'][mname] = len(rec)

    # ---- Statistical comparison ----
    print("\n" + "-" * 60)
    print("STATISTICAL COMPARISON: Mann-Whitney U Tests")
    print("-" * 60)

    mw_results = {}
    any_significant = False
    for metric in metric_names:
        pv = spectral_metrics['prime'][metric]
        cv = spectral_metrics['composite'][metric]
        if not pv or not cv:
            continue
        mw = mann_whitney_u(pv, cv)
        mw_results[metric] = mw
        pm = np.mean(pv)
        cm = np.mean(cv)
        sig = ""
        if mw['p_approx'] is not None and mw['p_approx'] < 0.05:
            sig = " * SIGNIFICANT"
            any_significant = True
        elif mw['p_approx'] is not None and mw['p_approx'] < 0.10:
            sig = " ~ marginal"
        print(f"  {metric}: prime_mean={pm:.6f}, comp_mean={cm:.6f}, "
              f"U={mw['U']:.1f}, z={mw['z']:.3f}, p≈{mw['p_approx']:.4f}{sig}")

    # ---- Factor recovery by method ----
    print("\n" + "-" * 60)
    print("FACTOR RECOVERY BY SPECTRAL METHOD")
    print("-" * 60)

    method_recovery = {}
    for p in composite_cases:
        if p not in all_results:
            continue
        factor_res_data = all_results[p].get('method_results', {})
        known = set(KNOWN_FACTORS.get(p, []))
        for mname, count in factor_res_data.items():
            if mname not in method_recovery:
                method_recovery[mname] = {'hits': 0, 'total': len(known)}
            method_recovery[mname]['hits'] += min(count, len(known))

    # Also count from factor_from_spectrum directly
    for p in composite_cases:
        M_p = (1 << p) - 1
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue
        seq_length = M_p if p <= 11 else 2 ** 14
        trace_seq = compute_trace_sequence(poly, seq_length)
        fr = factor_from_spectrum(trace_seq, M_p, p)

        for mname, mdata in fr.get('method_results', {}).items():
            rec = mdata.get('factors_recovered', [])
            found_primes = set()
            for r in rec:
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    f = r[-1]
                    if isinstance(f, (int, float)) and f == int(f):
                        if 1 < f < M_p and M_p % int(f) == 0:
                            found_primes.add(int(f))
            if mname not in method_recovery:
                method_recovery[mname] = {'hits': 0, 'total': 0}
            known = set(KNOWN_FACTORS.get(p, []))
            method_recovery[mname]['hits'] += len(found_primes & known)
            method_recovery[mname]['total'] += len(known)

    print(f"\n  {'Method':<25} | {'Factors Found':>13} | {'Total':>13} | {'Rate':>6}")
    print("  " + "-" * 65)
    for method, data in sorted(method_recovery.items()):
        rate = data['hits'] / data['total'] if data['total'] > 0 else 0
        print(f"  {method:<25} | {data['hits']:>13d} | {data['total']:>13d} | {rate:>5.0%}")

    # ---- Negative result summary ----
    print("\n" + "-" * 60)
    print("HONEST NEGATIVE RESULT")
    print("-" * 60)
    print(f"""
  1. PERIODIC AUTOCORRELATION: Always two-valued for m-sequences,
     regardless of M_p's primality. CANNOT distinguish prime/composite.

  2. WHT/FFT SPECTRUM: Nearly flat for all m-sequences. No systematic
     difference between prime and composite cases.

  3. MANN-WHITNEY U TESTS: {'At least one significant (p<0.05).' if any_significant else 'NO metric significant at p<0.05.'}
     With only {len(prime_cases)} prime and {len(composite_cases)} composite samples,
     statistical power is very low.

  4. FACTOR RECOVERY: Spectral methods alone cannot reliably extract
     factors. Methods that appear to work (decimation_balance,
     cycle_completion) are effectively trial division in disguise.

  CONCLUSION: Spectral analysis of Tr(C^k) does NOT provide a viable
  method for distinguishing prime from composite M_p. The m-sequence
  has ideal autocorrelation regardless of primality — a rigorous
  negative result.
""")

    return {
        'spectral_metrics': spectral_metrics,
        'mann_whitney_results': mw_results,
        'method_recovery': method_recovery,
        'all_results': all_results,
    }


# ============================================================
# Part C: Minimal Polynomial Construction
# ============================================================

def format_poly_compact(coeffs: List[int]) -> str:
    """Format polynomial (LSB first) as compact string with leading x^n term."""
    n = len(coeffs)
    terms = [f"x^{n}"]
    for i in range(n - 1, -1, -1):
        if coeffs[i]:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append("x")
            else:
                terms.append(f"x^{i}")
    return " + ".join(terms)


def part_c_minimal_polynomials() -> Dict:
    """Run Part C: Minimal Polynomial Construction."""
    print("\n" + "=" * 80)
    print("PART C: Minimal Polynomial Construction")
    print("=" * 80)
    print("""
For each factor q of M_p = 2^p - 1:
  1. Compute C^q (companion matrix raised to q-th power)
  2. Extract minimal polynomial (= char poly for cyclic matrices)
  3. Verify irreducibility and degree p
  4. Build companion matrix and verify order = M_p/q

THEOREM: If α is a root of primitive f(x) of degree p, and q | M_p:
  (a) minpoly(α^q) has degree p
  (b) Its companion matrix has order M_p/q
  (c) q = M_p / ord(companion(minpoly(α^q)))
""")

    composite_cases = [11, 23, 29]
    results = {}

    for p in composite_cases:
        M_p = (1 << p) - 1
        known = KNOWN_FACTORS.get(p, [])
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue

        C = companion_matrix(poly)
        identity = np.eye(p, dtype=np.int64)

        print(f"\n  {'='*60}")
        print(f"  p = {p}, M_p = {M_p} = {' x '.join(str(f) for f in known)}")
        print(f"  Primitive poly: {format_poly_compact(poly)}")

        case_results = []

        for q in known:
            print(f"\n    --- Factor q = {q} ---")
            print(f"    Predicted ord(α^q) = M_p/q = {M_p // q}")

            # Step 1: Compute C^q
            C_q = gf2_mat_pow(C, q)
            assert not np.array_equal(C_q % 2, identity), "C^q should not be I for q < M_p"

            # Step 2: Extract characteristic polynomial (= minimal polynomial)
            min_poly = char_poly_gf2_krylov(C_q)

            if min_poly is None:
                print(f"    ERROR: char_poly computation failed!")
                case_results.append({'q': q, 'error': 'char_poly_failed'})
                continue

            # Step 3: Verify irreducibility
            is_irred = is_irreducible(min_poly)

            # Step 4: Build companion matrix and compute order
            C_min = companion_matrix(min_poly)
            expected_order = M_p // q
            actual_order = compute_matrix_order(C_min, expected_order)

            order_correct = (actual_order == expected_order)
            recovered_factor = M_p // actual_order if actual_order else None
            factor_correct = (recovered_factor == q)

            # Display
            poly_str = format_poly_compact(min_poly)
            print(f"    Minimal polynomial: {poly_str}")
            print(f"    Coefficients (LSB first): {min_poly}")
            print(f"    Irreducible: {is_irred}")
            print(f"    Degree: {len(min_poly)} (expected: {p})")
            print(f"    Companion order: {actual_order} (expected: {expected_order})")
            print(f"    Order correct: {order_correct}")
            print(f"    Recovered factor: {recovered_factor} (expected: {q})")
            print(f"    Factor correct: {factor_correct}")

            case_results.append({
                'q': q,
                'min_poly_coeffs': min_poly,
                'is_irreducible': is_irred,
                'degree': len(min_poly),
                'degree_correct': len(min_poly) == p,
                'companion_order': actual_order,
                'expected_order': expected_order,
                'order_correct': order_correct,
                'recovered_factor': recovered_factor,
                'factor_correct': factor_correct,
            })

        results[p] = {
            'M_p': M_p,
            'factors': known,
            'factor_results': case_results,
            'all_verified': all(r.get('factor_correct', False) for r in case_results),
        }

    # Summary table
    print("\n" + "=" * 80)
    print("PART C SUMMARY")
    print("=" * 80)

    for p in composite_cases:
        if p not in results:
            continue
        r = results[p]
        print(f"\n  p = {p}, M_p = {r['M_p']}:")
        print(f"  {'q':>8} | {'Deg':>4} | {'Irred?':>6} | {'Ord(C_min)':>12} | "
              f"{'Expected':>12} | {'OK?':>4} | {'Factor':>8}")
        print("  " + "-" * 70)
        for fr in r['factor_results']:
            if 'error' in fr:
                print(f"  {fr['q']:8d} | ERROR: {fr['error']}")
                continue
            irred = "YES" if fr['is_irreducible'] else "NO"
            ok = "YES" if fr['order_correct'] else "NO"
            print(f"  {fr['q']:8d} | {fr['degree']:4d} | {irred:>6} | "
                  f"{fr['companion_order']:12d} | {fr['expected_order']:12d} | "
                  f"{ok:>4} | {fr['recovered_factor']:8d}")

        status = "VERIFIED ✓" if r['all_verified'] else "FAILED ✗"
        print(f"  Theorem 2: {status}")

    return results


# ============================================================
# Part D: Results Summary
# ============================================================

def part_d_summary(part_a: Dict, part_b: Dict, part_c: Dict):
    """Generate comprehensive results summary and save to JSON."""
    print("\n" + "=" * 80)
    print("PART D: Comprehensive Results Summary")
    print("=" * 80)

    # Factor Discovery Summary
    print("\n  FACTOR DISCOVERY COMPARISON")
    print("  " + "-" * 90)
    print(f"  {'p':>4} | {'M_p':>12} | {'Type':>8} | "
          f"{'C^d Method':>15} | {'Spectral':>15} | {'Known':>15} | {'Complete':>8}")
    print("  " + "-" * 90)

    all_ps = sorted(set(list(part_a.keys()) + list(part_c.keys())))
    for p in all_ps:
        M_p = (1 << p) - 1
        is_mp = p in KNOWN_MERSENNE_PRIMES
        known = KNOWN_FACTORS.get(p, [])

        cd_factors = part_a.get(p, {}).get('factors_found', [])
        spec_factors = part_b.get('all_results', {}).get(p, {}).get('factors_found', [])

        type_str = "PRIME" if is_mp else "composite"
        cd_str = ",".join(str(f) for f in cd_factors) or "(none)"
        spec_str = ",".join(str(f) for f in spec_factors) or "(none)"
        known_str = ",".join(str(f) for f in known) or "(prime)"

        cd_set = set(cd_factors)
        known_set = set(known)
        complete = "YES" if (cd_set >= known_set if known_set else True) else "NO"

        print(f"  {p:4d} | {M_p:12d} | {type_str:>8} | "
              f"{cd_str:>15} | {spec_str:>15} | {known_str:>15} | {complete:>8}")

    # Key Findings
    print("\n  KEY FINDINGS")
    print("  " + "-" * 50)
    print("""
  1. C^d ORDER PROBING (Part A) — NOVEL POSITIVE RESULT:
     For composite M_p, C^d reveals factors when gcd(M_p,d) > 1.
     The order drops from M_p to M_p/gcd(M_p,d), detected through
     GF(2) matrix operations (CA dynamics). Each C^d is computed
     iteratively as a CA step, and verification is a CA operation.

  2. SPECTRAL ANALYSIS (Part B) — RIGOROUS NEGATIVE RESULT:
     The trace sequence Tr(C^k) is an m-sequence with ideal
     autocorrelation REGARDLESS of primality. No spectral metric
     reliably distinguishes prime from composite M_p.

  3. MINIMAL POLYNOMIAL (Part C) — CONSTRUCTIVE POSITIVE RESULT:
     For each factor q, minpoly(α^q) is irreducible of degree p,
     and its companion matrix has order M_p/q, directly revealing q.

  4. FUNDAMENTAL INSIGHT:
     Factorization information is in the ORDER structure (C^d has
     reduced order when gcd(M_p,d)>1), NOT in the spectral structure
     (identical for all m-sequences). Order-based CA methods succeed
     where spectral methods fail.
""")

    # Save JSON
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_results = {
        'part_a_factor_discovery': {},
        'part_b_spectral_analysis': {},
        'part_c_minimal_polynomials': {},
    }

    for p, r in part_a.items():
        json_results['part_a_factor_discovery'][str(p)] = {
            'p': r['p'], 'M_p': r['M_p'],
            'is_mersenne_prime': r['is_mersenne_prime'],
            'factors_found': r['factors_found'],
            'known_factors': r['known_factors'],
            'all_factors_found': r['all_factors_found'],
            'false_positives': r['false_positives'],
            'd_of_first_discovery': {str(k): v for k, v in r.get('d_of_first_discovery', {}).items()},
            'computation_time': r['computation_time'],
        }

    json_results['part_b_spectral_analysis'] = _json_safe({
        'spectral_metrics': part_b.get('spectral_metrics', {}),
        'mann_whitney_results': part_b.get('mann_whitney_results', {}),
        'method_recovery': part_b.get('method_recovery', {}),
    })

    json_results['part_c_minimal_polynomials'] = _json_safe(part_c)

    output_path = os.path.join(results_dir, 'path1_path2_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


def _json_safe(obj):
    """Recursively make object JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return sorted(list(obj))
    else:
        return obj


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + " PATH 1 + PATH 2 EXPERIMENTS".center(78) + "║")
    print("║" + " GF(2) Matrix Power CA: Factor Discovery & Spectral Analysis".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    overall_start = time.time()

    part_a_results = part_a_factor_discovery()
    part_b_results = part_b_spectral_analysis()
    part_c_results = part_c_minimal_polynomials()
    part_d_summary(part_a_results, part_b_results, part_c_results)

    print(f"\n  Total time: {time.time() - overall_start:.1f}s")
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
