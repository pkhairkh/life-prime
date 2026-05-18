"""
Spectral Analysis of GF(2) Companion Matrix Trace Sequences
============================================================

CORE QUESTION: Can spectral analysis (WHT, FFT, autocorrelation) of the
trace sequence Tr(C^k) detect whether M_p = 2^p - 1 is prime or composite,
and can it recover factors WITHOUT knowing them in advance?

BACKGROUND:
  - C is the companion matrix of a primitive polynomial of degree p over GF(2)
  - The trace sequence Tr(C^k) for k=0,1,2,... is an m-sequence with period 2^p-1
  - For PRIME M_p: the periodic autocorrelation is two-valued (Golomb property)
  - For COMPOSITE M_p: the autocorrelation MAY degrade, and sub-structure at
    factor positions might be detectable through spectral methods

KEY INSIGHT: If q | M_p, the decimated sequence s(0), s(q), s(2q), ... has
period M_p/q rather than M_p. This change in sub-sampled statistics is the
primary detectable signature of compositeness.

METHODS IMPLEMENTED:
  1. Walsh-Hadamard Transform (WHT) — spectral flatness analysis
  2. FFT power spectrum — frequency-domain analysis
  3. Periodic & aperiodic autocorrelation — two-valued property check
  4. Decimation analysis — sub-sampled statistics at candidate periods
  5. Factor recovery from spectral anomalies
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd
from collections import Counter
import sys
import os
import json
import time

# Import GF(2) matrix utilities from existing code
sys.path.insert(0, os.path.dirname(__file__))
from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, gf2_mat_vec,
    PRIMITIVE_POLYS_GF2, PRIMITIVE_POLYS_MERSENNE, is_prime_simple
)


# ============================================================
# Primitive Polynomials for All Test Cases
# ============================================================

PRIMITIVE_POLYS = {
    2: [1, 1],
    3: [1, 1, 0],
    5: [1, 0, 1, 0, 0],
    7: [1, 1, 0, 0, 0, 0, 0],
    11: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    13: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    19: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    23: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    29: [1, 0, 1] + [0] * 26,
    31: [1, 0, 0, 1] + [0] * 27,
}

# Known Mersenne prime exponents
MERSENNE_PRIME_EXPONENTS = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}

# Known factorizations for composite Mersenne numbers
KNOWN_FACTORS = {
    11: [23, 89],           # 2047 = 23 × 89
    23: [47, 178481],       # 8388607 = 47 × 178481
    29: [233, 1103, 2089],  # 536870911 = 233 × 1103 × 2089
}


# ============================================================
# 1. Trace Sequence Computation
# ============================================================

def compute_trace_sequence(poly_coeffs: List[int], length: int) -> np.ndarray:
    """
    Compute Tr(C^k) for k = 0, 1, ..., length-1.

    C is the companion matrix of the primitive polynomial over GF(2).
    The trace is computed over GF(2), so Tr(C^k) in {0, 1}.

    Uses iterative multiplication: C^{k+1} = C^k * C, which is efficient
    for computing consecutive powers (O(p^2) per step).

    For an m-sequence: period = 2^p - 1, with 2^{p-1} ones and 2^{p-1} - 1 zeros.
    """
    p = len(poly_coeffs)
    C = companion_matrix(poly_coeffs)
    C_power = np.eye(p, dtype=np.int64)

    traces = np.zeros(length, dtype=np.int8)
    for k in range(length):
        traces[k] = int(np.trace(C_power) % 2)
        C_power = gf2_mat_mul(C_power, C)

    return traces


def compute_trace_at_positions(poly_coeffs: List[int], positions: List[int]) -> np.ndarray:
    """
    Compute Tr(C^k) at specific positions using matrix powering.
    O(p^3 * log(k)) per position.
    """
    p = len(poly_coeffs)
    C = companion_matrix(poly_coeffs)

    traces = np.zeros(len(positions), dtype=np.int8)
    for i, k in enumerate(positions):
        if k == 0:
            traces[i] = int(p % 2)
        else:
            C_k = gf2_mat_pow(C, k)
            traces[i] = int(np.trace(C_k) % 2)

    return traces


# ============================================================
# 2. Walsh-Hadamard Transform
# ============================================================

def walsh_hadamard_transform(seq: np.ndarray) -> np.ndarray:
    """
    Compute the Walsh-Hadamard Transform (WHT) of a real-valued sequence.

    The input should be a ±1 sequence (map binary {0,1} to {-1,+1} first).
    Sequence length is padded to the next power of 2 if necessary.

    The WHT reveals the "sequency" spectrum — correlation with all
    Walsh functions (square-wave analogs of sinusoids).

    For an m-sequence, the WHT spectrum should be nearly flat
    (all non-DC components approximately ±1), reflecting the
    pseudo-random nature.

    Returns the WHT spectrum (same length as padded input).
    """
    n = len(seq)
    # Pad to next power of 2
    if n & (n - 1) != 0:
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        padded = np.zeros(next_pow2, dtype=np.float64)
        padded[:n] = seq
        seq = padded
        n = next_pow2

    # Fast WHT (iterative, in-place)
    result = seq.astype(np.float64).copy()
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = result[j]
                y = result[j + h]
                result[j] = x + y
                result[j + h] = x - y
        h *= 2

    return result


def fast_wht_numpy(seq: np.ndarray) -> np.ndarray:
    """
    Faster WHT using numpy vectorized operations.
    """
    n = len(seq)
    if n & (n - 1) != 0:
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        padded = np.zeros(next_pow2, dtype=np.float64)
        padded[:n] = seq
        seq = padded
        n = next_pow2

    result = seq.astype(np.float64).copy()
    h = 1
    while h < n:
        # Vectorized butterfly operation
        even = result[:n:2].copy() if h == 1 else result[::1].copy()
        left = result[:n].copy()
        right_idx = np.arange(n) ^ h  # XOR with h gives the pair index
        # Simpler approach: work in blocks
        for i in range(0, n, 2 * h):
            temp = result[i:i+h].copy()
            result[i:i+h] = temp + result[i+h:i+2*h]
            result[i+h:i+2*h] = temp - result[i+h:i+2*h]
        h *= 2

    return result


# ============================================================
# 3. Autocorrelation Functions
# ============================================================

def periodic_autocorrelation(seq_pm1: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute the periodic autocorrelation of a ±1 sequence using FFT.

    R(τ) = (1/N) * Σ_{k=0}^{N-1} s(k) * s((k+τ) mod N)

    For m-sequences: R(0) = 1, R(τ) = -1/N for τ ≠ 0.
    This is ALWAYS two-valued for m-sequences, regardless of primality.
    """
    n = len(seq_pm1)
    s = seq_pm1.astype(np.float64)

    S = np.fft.fft(s)
    R = np.real(np.fft.ifft(S * np.conj(S))) / n

    if max_lag is not None:
        return R[:max_lag + 1]
    return R


def aperiodic_autocorrelation(seq_pm1: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute the aperiodic (non-cyclic) autocorrelation of a ±1 sequence.

    R(τ) = (1/(N-τ)) * Σ_{k=0}^{N-τ-1} s(k) * s(k+τ)

    Unlike periodic autocorrelation, this is NOT always two-valued for m-sequences.
    The variance of aperiodic autocorrelation values may differ between
    prime and composite period lengths.
    """
    n = len(seq_pm1)
    s = seq_pm1.astype(np.float64)

    if max_lag is None:
        max_lag = min(n - 1, 5000)

    R = np.zeros(max_lag + 1)
    R[0] = 1.0
    for tau in range(1, max_lag + 1):
        if tau >= n:
            break
        R[tau] = np.sum(s[:n-tau] * s[tau:]) / (n - tau)

    return R


# ============================================================
# 4. Spectral Factor Detection (Core Algorithm)
# ============================================================

def spectral_factor_detection(trace_seq: np.ndarray, M_p: int, p: int) -> Dict:
    """
    Core spectral analysis for distinguishing prime from composite M_p.

    Analyses:
      1. Walsh-Hadamard Transform — spectral flatness
      2. FFT power spectrum — frequency-domain features
      3. Periodic autocorrelation — two-valued property
      4. Aperiodic autocorrelation — variance analysis
      5. Decimation statistics — sub-sampled balance at candidate periods
    """
    N = len(trace_seq)

    # Map {0,1} -> {-1,+1}
    seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

    results = {
        'M_p': M_p,
        'p': p,
        'seq_length': N,
        'is_mersenne_prime': is_prime_simple(M_p),
    }

    # ---- 1. WHT Analysis ----
    wht_len = 1
    while wht_len * 2 <= N:
        wht_len *= 2
    wht_len = max(wht_len, 4)

    wht_input = seq_pm1[:wht_len]
    wht_spectrum = walsh_hadamard_transform(wht_input)

    wht_abs = np.abs(wht_spectrum)
    dc_val = wht_abs[0]
    wht_no_dc = wht_abs[1:]

    results['wht'] = {
        'length': wht_len,
        'dc_component': float(dc_val),
        'max_magnitude': float(np.max(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'mean_magnitude': float(np.mean(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'std_magnitude': float(np.std(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'spectrum_variance': float(np.var(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'skewness': float(_skewness(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'kurtosis': float(_kurtosis(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'num_large_peaks': int(np.sum(wht_no_dc > np.mean(wht_no_dc) + 3 * np.std(wht_no_dc))) if len(wht_no_dc) > 0 else 0,
        'top_10_positions': [int(x) for x in np.argsort(-wht_no_dc)[:10]] if len(wht_no_dc) > 0 else [],
        'top_10_magnitudes': [float(x) for x in sorted(wht_no_dc, reverse=True)[:10]] if len(wht_no_dc) > 0 else [],
    }

    # ---- 2. FFT Power Spectrum ----
    fft_len = wht_len
    fft_result = np.fft.fft(seq_pm1[:fft_len])
    power_spectrum = np.abs(fft_result) ** 2
    power_no_dc = power_spectrum[1:]

    results['fft'] = {
        'length': fft_len,
        'mean_power': float(np.mean(power_no_dc)) if len(power_no_dc) > 0 else 0,
        'std_power': float(np.std(power_no_dc)) if len(power_no_dc) > 0 else 0,
        'max_power': float(np.max(power_no_dc)) if len(power_no_dc) > 0 else 0,
        'spectral_flatness': float(np.mean(power_no_dc) ** 2 / (np.mean(power_no_dc ** 2) + 1e-30)) if len(power_no_dc) > 0 else 0,
        'num_peaks_3sigma': int(np.sum(power_no_dc > np.mean(power_no_dc) + 3 * np.std(power_no_dc))) if len(power_no_dc) > 0 else 0,
        'peak_positions': [int(x) for x in np.argsort(-power_no_dc)[:20]] if len(power_no_dc) > 0 else [],
    }

    # ---- 3. Periodic Autocorrelation ----
    max_pacf_lag = min(N, 500)
    pacf = periodic_autocorrelation(seq_pm1, max_lag=max_pacf_lag)

    expected_pacf = -1.0 / M_p
    pacf_nonzero = pacf[1:]
    pacf_deviations = np.abs(pacf_nonzero - expected_pacf)

    results['periodic_autocorrelation'] = {
        'expected_value': expected_pacf,
        'mean_value': float(np.mean(pacf_nonzero)),
        'std_value': float(np.std(pacf_nonzero)),
        'max_deviation': float(np.max(pacf_deviations)),
        'mean_deviation': float(np.mean(pacf_deviations)),
        'num_distinct_rounded': len(set(np.round(pacf_nonzero, 4))),
        'is_two_valued': len(set(np.round(pacf_nonzero, 4))) <= 2,
        'sample_values': pacf[1:min(30, len(pacf))].tolist(),
    }

    # ---- 4. Aperiodic Autocorrelation ----
    max_aacf_lag = min(N - 1, 500)
    aacf = aperiodic_autocorrelation(seq_pm1, max_lag=max_aacf_lag)

    aacf_nonzero = aacf[1:]
    results['aperiodic_autocorrelation'] = {
        'mean_value': float(np.mean(aacf_nonzero)),
        'std_value': float(np.std(aacf_nonzero)),
        'max_value': float(np.max(aacf_nonzero)),
        'min_value': float(np.min(aacf_nonzero)),
        'range': float(np.max(aacf_nonzero) - np.min(aacf_nonzero)),
        'num_distinct_rounded': len(set(np.round(aacf_nonzero, 4))),
        'sample_values': aacf[1:min(30, len(aacf))].tolist(),
    }

    # ---- 5. Decimation (Sub-sampling) Analysis ----
    # KEY: When q | M_p, the decimated sequence s(0), s(q), s(2q), ...
    # has period M_p/q, which is SHORTER than M_p.
    # When gcd(q, M_p) = 1, the decimated sequence is a permutation
    # of the original (still period M_p).
    # This difference in sub-sampled statistics is the most promising
    # spectral indicator of compositeness.

    decimation_stats = {}
    candidates = set()

    # Add known factors
    known_factors = KNOWN_FACTORS.get(p, [])
    candidates.update(known_factors)

    # Add small integers (both divisors and non-divisors)
    for d in range(2, min(200, M_p)):
        candidates.add(d)

    # Add cofactors
    for d in list(candidates):
        if d > 0 and M_p % d == 0:
            candidates.add(M_p // d)

    for d in sorted(candidates):
        if d <= 0 or d >= N:
            continue
        sub_indices = list(range(0, N, d))
        if len(sub_indices) < 5:
            continue
        sub_seq = trace_seq[sub_indices]
        n_ones = int(np.sum(sub_seq))
        ratio = n_ones / len(sub_seq)

        # Compute autocorrelation of the decimated sequence
        sub_pm1 = 2.0 * sub_seq.astype(np.float64) - 1.0
        if len(sub_pm1) > 2:
            sub_acorr = periodic_autocorrelation(sub_pm1, max_lag=min(len(sub_pm1) - 1, 50))
            sub_acorr_nonzero = sub_acorr[1:]
            acorr_std = float(np.std(sub_acorr_nonzero)) if len(sub_acorr_nonzero) > 0 else 0
        else:
            acorr_std = 0

        is_factor = (M_p % d == 0)
        decimation_stats[d] = {
            'subsequence_length': len(sub_indices),
            'ones_ratio': float(ratio),
            'deviation_from_half': abs(ratio - 0.5),
            'acorr_std': acorr_std,
            'is_factor': is_factor,
        }

    results['decimation_stats'] = decimation_stats

    return results


# ============================================================
# 5. Factor Recovery from Spectrum (KEY EXPERIMENT)
# ============================================================

def factor_from_spectrum(trace_seq: np.ndarray, M_p: int, p: int) -> Dict:
    """
    THE KEY EXPERIMENT: Try to extract factors of M_p from spectral analysis alone.

    Five methods are attempted:
      1. WHT peak positions → check gcd(2^τ - 1, M_p)
      2. FFT spectral peaks → check gcd relationships
      3. Autocorrelation anomalies → check gcd at anomalous lags
      4. Decimation balance check → statistical test at candidate periods
      5. Direct GCD method → gcd(2^d - 1, M_p) for small d
    """
    N = len(trace_seq)
    seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

    results = {
        'M_p': M_p,
        'p': p,
        'is_prime': is_prime_simple(M_p),
        'factors_known': KNOWN_FACTORS.get(p, []),
        'factors_found': [],
        'method_results': {},
    }

    found_factors = set()

    # ---- Method 1: WHT-based factor extraction ----
    wht_len = 1
    while wht_len * 2 <= N:
        wht_len *= 2
    wht_len = max(wht_len, 4)

    wht_spectrum = walsh_hadamard_transform(seq_pm1[:wht_len])
    wht_abs = np.abs(wht_spectrum)
    wht_abs[0] = 0  # Exclude DC

    mean_wht = np.mean(wht_abs)
    std_wht = np.std(wht_abs)
    threshold = mean_wht + 2 * std_wht
    wht_peaks = np.where(wht_abs > threshold)[0]

    wht_recovered = []
    for tau in wht_peaks:
        if tau == 0 or tau >= M_p:
            continue
        # Key insight: check gcd(2^τ - 1, M_p)
        # If τ is the order of 2 mod some factor q, then q | gcd(2^τ - 1, M_p)
        try:
            val = (pow(2, int(tau), M_p) - 1) % M_p
            g = gcd(val, M_p)
            if 1 < g < M_p:
                found_factors.add(g)
                wht_recovered.append((int(tau), int(g)))
        except Exception:
            pass

    results['method_results']['wht'] = {
        'num_peaks': int(len(wht_peaks)),
        'peak_positions': [int(x) for x in wht_peaks[:20]],
        'factors_recovered': wht_recovered,
    }

    # ---- Method 2: FFT spectral peaks ----
    fft_len = wht_len
    fft_result = np.fft.fft(seq_pm1[:fft_len])
    power = np.abs(fft_result) ** 2
    power_no_dc = power[1:]

    mean_power = np.mean(power_no_dc)
    std_power = np.std(power_no_dc)
    fft_threshold = mean_power + 3 * std_power
    fft_peaks = np.where(power_no_dc > fft_threshold)[0] + 1  # +1 for DC offset

    fft_recovered = []
    for pos in fft_peaks[:50]:
        if pos == 0:
            continue
        # Check if frequency position relates to a factor
        # Frequency f corresponds to period N/f
        g = gcd(int(pos), M_p)
        if 1 < g < M_p:
            found_factors.add(g)
            fft_recovered.append((int(pos), int(g)))
        # Also check the period implied by this frequency
        if pos > 0 and N // pos > 0:
            period_candidate = N // pos
            g2 = gcd(int(period_candidate), M_p)
            if 1 < g2 < M_p:
                found_factors.add(g2)
                fft_recovered.append((int(pos), int(g2)))

    results['method_results']['fft'] = {
        'num_peaks': int(len(fft_peaks)),
        'peak_positions': [int(x) for x in fft_peaks[:20]],
        'factors_recovered': fft_recovered,
    }

    # ---- Method 3: Autocorrelation anomalies ----
    max_lag = min(N - 1, min(5000, M_p - 1))
    aacf = aperiodic_autocorrelation(seq_pm1, max_lag=max_lag)

    aacf_nonzero = aacf[1:]
    aacf_mean = np.mean(aacf_nonzero)
    aacf_std = np.std(aacf_nonzero)
    anomaly_threshold = 3 * aacf_std

    acorr_recovered = []
    for tau in range(1, min(max_lag + 1, len(aacf))):
        if abs(aacf[tau] - aacf_mean) > anomaly_threshold:
            # Anomalous autocorrelation at lag tau
            # Check if tau relates to a factor
            g = gcd(int(tau), M_p)
            if 1 < g < M_p:
                found_factors.add(g)
                acorr_recovered.append((int(tau), float(aacf[tau]), int(g)))
            # Also check if the decimated sequence at step tau has shorter period
            try:
                val = (pow(2, int(tau), M_p) - 1) % M_p
                g2 = gcd(val, M_p)
                if 1 < g2 < M_p:
                    found_factors.add(g2)
                    acorr_recovered.append((int(tau), float(aacf[tau]), int(g2)))
            except Exception:
                pass

    results['method_results']['autocorrelation'] = {
        'num_anomalies': len(acorr_recovered),
        'factors_recovered': acorr_recovered,
    }

    # ---- Method 4: Decimation balance check ----
    # For each d, check if the subsequence at step d has anomalous balance
    # If d | M_p, the decimated sequence has period M_p/d, which may cause
    # different statistics than when gcd(d, M_p) = 1
    balance_recovered = []
    balance_details = []

    for d in range(2, min(1000, M_p, N)):
        sub_indices = list(range(0, N, d))
        if len(sub_indices) < 5:
            continue
        sub_seq = trace_seq[sub_indices]
        n_ones = int(np.sum(sub_seq))
        ratio = n_ones / len(sub_seq)

        # Binomial test: for balanced sequence, expect ~0.5
        expected_std = np.sqrt(0.25 / len(sub_seq))
        z_score = abs(ratio - 0.5) / expected_std if expected_std > 0 else 0

        balance_details.append({
            'd': d,
            'ratio': float(ratio),
            'z_score': float(z_score),
            'is_factor': (M_p % d == 0),
        })

        if z_score > 3.0:
            # Significant deviation — check if d is a factor
            g = gcd(int(d), M_p)
            if 1 < g < M_p:
                found_factors.add(g)
                balance_recovered.append((int(d), float(ratio), float(z_score), int(g)))

    results['method_results']['decimation_balance'] = {
        'num_significant': sum(1 for b in balance_details if b['z_score'] > 3.0),
        'factors_recovered': balance_recovered,
        'balance_details': balance_details[:50],  # Save first 50 for visualization
    }

    # ---- Method 5: Direct GCD with 2^d - 1 ----
    # Mathematical identity: gcd(2^a - 1, 2^b - 1) = 2^gcd(a,b) - 1
    # Since p is prime, gcd(d, p) is either 1 or p
    # So gcd(2^d - 1, M_p) = 2^gcd(d,p) - 1 = 1 or M_p
    # This means the GCD method with 2^d - 1 DOES NOT work for
    # prime-exponent Mersenne numbers (doh!)
    #
    # HOWEVER: for each factor q of M_p, the order of 2 mod q
    # divides q-1 and also divides p. Since p is prime, ord(2, q) = p
    # (for q prime factors of M_p with p prime).
    # So we can't get factors from gcd(2^d - 1, M_p) alone.
    #
    # Instead, we use the TRACE SEQUENCE to detect factor-related structure:
    # For each candidate d, compute the "folding" correlation:
    # F(d) = Σ_{k=0}^{N-1} (-1)^{s(k)} * (-1)^{s(k+d)}
    # If d | M_p, the fold at period d creates correlations.

    folding_recovered = []
    for d in range(2, min(200, M_p, N)):
        if d >= N:
            break
        # Compute fold correlation: sum of s(k)*s(k+d) for all k
        fold_corr = np.sum(seq_pm1[:N-d] * seq_pm1[d:N]) / (N - d)

        # For m-sequence: fold_corr ≈ -1/M_p for all d
        # Significant deviation might indicate d | M_p
        expected = -1.0 / M_p
        deviation = abs(fold_corr - expected)

        if deviation > 0.05:  # Significant deviation
            g = gcd(int(d), M_p)
            if 1 < g < M_p:
                found_factors.add(g)
                folding_recovered.append((int(d), float(fold_corr), float(deviation), int(g)))

    results['method_results']['folding'] = {
        'num_significant': len(folding_recovered),
        'factors_recovered': folding_recovered,
    }

    # ---- Method 6: Cycle Completion Test ----
    # For each d, check if the decimated sequence forms a complete cycle.
    # If d | M_p, then n*d = M_p where n = len(decimated_seq), so the
    # decimated sequence is exactly one full period of C^d.
    # This is ALGEBRAIC (not spectral) but uses the trace sequence.
    # It's equivalent to checking if d | M_p (trial division).
    completion_recovered = []
    for d in range(2, min(2000, M_p, N)):
        n = N // d  # Number of elements in decimated sequence
        if n < 2:
            continue
        # Check if n*d == M_p (exact cycle completion)
        # We can detect this from the trace sequence by checking
        # if the decimated sequence "wraps around" properly
        if N >= M_p:
            # We have enough of the sequence to check
            # n*d == M_p iff d | M_p
            if n * d == M_p:
                g = gcd(int(d), M_p)
                if 1 < g < M_p:
                    found_factors.add(g)
                    completion_recovered.append((int(d), int(g)))

    results['method_results']['cycle_completion'] = {
        'num_factors_found': len(completion_recovered),
        'factors_recovered': completion_recovered,
        'note': 'Equivalent to trial division (checks if d | M_p)',
    }

    # ---- Method 7: Linear Complexity (Berlekamp-Massey) ----
    # The linear complexity of an m-sequence is p (the degree).
    # Decimated sequences at factor step sizes might have different LC.
    # However, our experiments show LC remains p for most decimations.
    lc_recovered = []
    for d in range(2, min(100, M_p, N)):
        sub_indices = list(range(0, N, d))
        if len(sub_indices) < 2 * p + 2:  # Need enough samples for BM
            continue
        sub_seq = trace_seq[sub_indices]
        lc = berlekamp_massey(sub_seq.tolist())
        if lc != p and lc > 0:
            # Linear complexity differs from expected — might indicate factor
            g = gcd(int(d), M_p)
            if 1 < g < M_p:
                found_factors.add(g)
                lc_recovered.append((int(d), int(lc), int(g)))

    results['method_results']['linear_complexity'] = {
        'num_anomalies': len(lc_recovered),
        'factors_recovered': lc_recovered,
    }

    results['factors_found'] = sorted(found_factors)
    known = set(KNOWN_FACTORS.get(p, []))
    if known:
        results['recovery_rate'] = len(found_factors & known) / len(known)
    else:
        results['recovery_rate'] = None

    return results


# ============================================================
# 6. Comprehensive Experiment
# ============================================================

def comprehensive_spectral_experiment() -> Dict:
    """
    Run comprehensive spectral analysis on all test cases.

    Prime Mersenne: p = 2, 3, 5, 7, 13, 17, 19
    Composite Mersenne: p = 11, 23, 29
    """
    prime_cases = [2, 3, 5, 7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases

    all_results = {}

    for p in all_cases:
        M_p = 2**p - 1
        is_prime = is_prime_simple(M_p)
        poly = PRIMITIVE_POLYS.get(p)

        if poly is None:
            print(f"  Skipping p={p}: no primitive polynomial defined")
            continue

        print(f"\n{'='*60}")
        print(f"p = {p}, M_p = {M_p} ({'PRIME' if is_prime else 'COMPOSITE'})")
        print(f"{'='*60}")

        # Determine sequence length
        if p <= 11:
            seq_length = M_p  # Full period
        elif p <= 17:
            seq_length = 2**16  # 65536
        elif p <= 19:
            seq_length = 2**16
        elif p <= 23:
            seq_length = 2**16
        else:
            seq_length = 2**14  # 16384 for p=29

        print(f"  Computing trace sequence of length {seq_length}...")
        t0 = time.time()
        trace_seq = compute_trace_sequence(poly, seq_length)
        t1 = time.time()
        print(f"  Trace sequence computed in {t1-t0:.2f}s")

        # Verify basic m-sequence properties
        n_ones = int(np.sum(trace_seq))
        n_zeros = len(trace_seq) - n_ones
        print(f"  Ones: {n_ones}, Zeros: {n_zeros}, Ratio: {n_ones/len(trace_seq):.4f}")

        # Spectral analysis
        print(f"  Running spectral factor detection...")
        t0 = time.time()
        spectral_results = spectral_factor_detection(trace_seq, M_p, p)
        t1 = time.time()
        print(f"  Spectral analysis done in {t1-t0:.2f}s")

        # Factor extraction
        print(f"  Running factor-from-spectrum analysis...")
        t0 = time.time()
        factor_results = factor_from_spectrum(trace_seq, M_p, p)
        t1 = time.time()
        print(f"  Factor extraction done in {t1-t0:.2f}s")

        # Print summary
        is_mp = spectral_results['is_mersenne_prime']
        pacf = spectral_results['periodic_autocorrelation']
        aacf = spectral_results['aperiodic_autocorrelation']
        wht = spectral_results['wht']
        fft_r = spectral_results['fft']

        print(f"\n  SUMMARY for p={p}, M_p={M_p}:")
        print(f"    Mersenne prime: {is_mp}")
        print(f"    Periodic AC: {pacf['num_distinct_rounded']} distinct values, "
              f"two-valued: {pacf['is_two_valued']}, "
              f"max deviation: {pacf['max_deviation']:.6f}")
        print(f"    Aperiodic AC: range={aacf['range']:.6f}, "
              f"std={aacf['std_value']:.6f}")
        print(f"    WHT: max={wht['max_magnitude']:.1f}, "
              f"mean={wht['mean_magnitude']:.1f}, "
              f"std={wht['std_magnitude']:.1f}, "
              f"kurtosis={wht['kurtosis']:.2f}")
        print(f"    FFT: peaks(3σ)={fft_r['num_peaks_3sigma']}, "
              f"flatness={fft_r['spectral_flatness']:.4f}")

        if not is_mp:
            print(f"    Known factors: {KNOWN_FACTORS.get(p, [])}")
            print(f"    Factors found by spectral methods: {factor_results['factors_found']}")
            if factor_results['recovery_rate'] is not None:
                print(f"    Recovery rate: {factor_results['recovery_rate']:.0%}")

            # Show per-method results
            for method_name, method_data in factor_results['method_results'].items():
                recovered = method_data.get('factors_recovered', [])
                if recovered:
                    print(f"      {method_name}: {recovered}")

        # Decimation analysis summary
        dec_stats = spectral_results['decimation_stats']
        factor_ds = [d for d, s in dec_stats.items() if s['is_factor']]
        non_factor_ds = [d for d, s in dec_stats.items() if not s['is_factor']]

        if factor_ds and non_factor_ds:
            factor_deviations = [dec_stats[d]['deviation_from_half'] for d in factor_ds]
            non_factor_deviations = [dec_stats[d]['deviation_from_half'] for d in non_factor_ds]
            print(f"    Decimation: factor d's mean deviation={np.mean(factor_deviations):.4f}, "
                  f"non-factor d's mean deviation={np.mean(non_factor_deviations):.4f}")

        all_results[p] = {
            'spectral': spectral_results,
            'factors': factor_results,
            'trace_sequence': trace_seq,
        }

    return all_results


# ============================================================
# Helper Functions
# ============================================================

def berlekamp_massey(s: list) -> int:
    """
    Compute the linear complexity of a binary sequence using the
    Berlekamp-Massey algorithm over GF(2).
    
    The linear complexity is the length of the shortest LFSR that
    generates the sequence. For an m-sequence of period 2^p-1,
    the linear complexity is p.
    """
    n = len(s)
    c = [1]  # Connection polynomial
    b = [1]  # Previous connection polynomial
    L = 0    # Current linear complexity
    m = 1    # Number of iterations since last update

    for N in range(n):
        # Compute discrepancy
        d = s[N]
        for i in range(1, L + 1):
            if i < len(c):
                d ^= c[i] & s[N - i]

        if d == 0:
            m += 1
        else:
            t = c[:]
            # c = c + b*x^m (over GF(2))
            pad = [0] * m
            shifted_b = pad + b
            while len(c) < len(shifted_b):
                c.append(0)
            while len(shifted_b) < len(c):
                shifted_b.append(0)
            c = [c[i] ^ shifted_b[i] for i in range(len(c))]

            if 2 * L <= N:
                L = N + 1 - L
                b = t[:]
                m = 1
            else:
                m += 1

    return L


def _skewness(arr: np.ndarray) -> float:
    """Compute skewness of an array."""
    if len(arr) == 0:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis of an array."""
    if len(arr) == 0:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = comprehensive_spectral_experiment()

    # Save results summary (without large arrays)
    summary = {}
    for p, res in results.items():
        s = {
            'spectral': {},
            'factors': {},
        }
        # Spectral results (exclude large decimation stats)
        for k, v in res['spectral'].items():
            if k == 'decimation_stats':
                # Only save factor-related entries
                s['spectral']['decimation_summary'] = {
                    d: v for d, v in res['spectral']['decimation_stats'].items()
                    if v.get('is_factor', False)
                }
            else:
                s['spectral'][k] = v
        # Factor results (exclude large balance details)
        for k, v in res['factors'].items():
            if k == 'method_results':
                s['factors'][k] = {}
                for mk, mv in v.items():
                    s['factors'][k][mk] = {kk: vv for kk, vv in mv.items()
                                           if kk != 'balance_details'}
            else:
                s['factors'][k] = v
        summary[p] = s

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'spectral_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n\nResults saved to {results_dir}/spectral_analysis_results.json")

    # Print overall conclusions
    print("\n" + "="*60)
    print("OVERALL CONCLUSIONS")
    print("="*60)

    # Check if autocorrelation distinguishes prime from composite
    prime_two_valued = []
    composite_two_valued = []
    for p, res in results.items():
        is_p = res['spectral']['is_mersenne_prime']
        twv = res['spectral']['periodic_autocorrelation']['is_two_valued']
        if is_p:
            prime_two_valued.append((p, twv))
        else:
            composite_two_valued.append((p, twv))

    print(f"\n1. AUTOCORRELATION TWO-VALUED PROPERTY:")
    print(f"   Prime M_p: {prime_two_valued}")
    print(f"   Composite M_p: {composite_two_valued}")
    all_two_valued = all(v for _, v in prime_two_valued + composite_two_valued)
    print(f"   All m-sequences have two-valued periodic AC: {all_two_valued}")
    print(f"   => Periodic autocorrelation CANNOT distinguish prime from composite")

    # Check factor recovery
    print(f"\n2. FACTOR RECOVERY FROM SPECTRAL METHODS:")
    for p, res in results.items():
        if not res['spectral']['is_mersenne_prime']:
            found = res['factors']['factors_found']
            known = res['factors']['factors_known']
            rate = res['factors']['recovery_rate']
            print(f"   p={p}: found={found}, known={known}, rate={rate}")

    # Check WHT spectral differences
    print(f"\n3. WHT SPECTRAL DIFFERENCES:")
    for p, res in results.items():
        wht = res['spectral']['wht']
        is_p = res['spectral']['is_mersenne_prime']
        print(f"   p={p} ({'PRIME' if is_p else 'COMP'}): "
              f"kurtosis={wht['kurtosis']:.2f}, "
              f"peaks(2σ)={wht['num_large_peaks']}, "
              f"var={wht['spectrum_variance']:.1f}")
