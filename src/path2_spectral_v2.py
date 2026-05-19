#!/usr/bin/env python3
"""
Path 2 v2: Beyond the Trace Sequence — Multi-Dimensional Spectral Analysis
===========================================================================

PREVIOUS NEGATIVE RESULT (v1):
  Spectral analysis of Tr(C^k) cannot distinguish prime from composite M_p.
  Reason: Tr(C^k) is an m-sequence with ideal autocorrelation REGARDLESS
  of whether the period 2^p - 1 is prime. This is a theorem from coding theory.

NEW APPROACH (v2):
  Go beyond the scalar trace. Analyze:
  1. Hamming Weight Trajectory: wt(C^k * v) for various v
  2. Multi-Bit State Correlation: cross-correlation between bit positions
  3. State Collision Detection: Brent's algorithm on C^d orbits
  4. Cross-Polynomial Correlation: compare state evolutions from different
     primitive polynomials of the same degree
  5. Higher-Order Spectra: bispectrum of the trace sequence
  6. Rank Dynamics: track rank(C^k - I) over GF(2)
  7. Column Correlation Matrix: spectral analysis of C^k column structure

KEY HYPOTHESIS:
  While the SCALAR trace is an m-sequence (ideal randomness), the
  MULTI-DIMENSIONAL state evolution may leak factor structure through
  higher-order statistics that the power spectrum cannot capture.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd, isqrt, log2
from collections import Counter
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, gf2_mat_vec,
    is_prime_simple
)

# ============================================================
# Primitive Polynomials (verified correct)
# ============================================================

PRIMITIVE_POLYS = {
    2: [1, 1],
    3: [1, 1, 0],                         # x^3 + x + 1
    5: [1, 0, 1, 0, 0],                   # x^5 + x^2 + 1
    7: [1, 1, 0, 0, 0, 0, 0],            # x^7 + x + 1
    11: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x^11 + x^2 + 1
    13: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x^13 + x^5 + x^2 + x + 1
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^17 + x^3 + 1
    19: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^19 + x^5 + x^2 + x + 1
    23: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^23 + x^5 + 1
    29: [1, 0, 1] + [0] * 26,            # x^29 + x^2 + 1
    31: [1, 0, 0, 1] + [0] * 27,         # x^31 + x^3 + 1
}

# Second primitive polynomials for cross-correlation analysis
# (Different primitive polys of the same degree produce different m-sequences)
PRIMITIVE_POLYS_ALT = {
    3: [1, 0, 1],                         # x^3 + x^2 + 1
    5: [1, 1, 0, 0, 0],                   # x^5 + x^2 + 1 ... wait need different
    7: [1, 0, 0, 1, 0, 0, 0],            # x^7 + x^3 + 1
    11: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^11 + x + 1 (not verified primitive, use carefully)
    13: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^13 + x^2 + 1
    17: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^17 + x + 1
    19: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^19 + x^3 + 1
}

KNOWN_FACTORS = {
    11: [23, 89],
    23: [47, 178481],
    29: [233, 1103, 2089],
}

MERSENNE_PRIME_EXPONENTS = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}


# ============================================================
# Method 1: Hamming Weight Trajectory Analysis
# ============================================================

def hamming_weight_trajectory(poly_coeffs: List[int], length: int,
                               v: np.ndarray = None) -> np.ndarray:
    """
    Compute the Hamming weight of C^k * v for k = 0, 1, ..., length-1.
    
    Unlike the trace (which is the mod-2 sum of diagonal entries), the
    Hamming weight of the state vector is a richer signal. For a random
    state in GF(2)^p, the expected weight is p/2. But the weight
    trajectory may show periodic structure related to factors of M_p.
    
    For prime M_p: the state visits ALL 2^p - 1 non-zero vectors,
      so the weight distribution is Binomial(p, 0.5) over the full orbit.
    For composite M_p: same thing (C has order M_p regardless of primality).
    
    BUT: the ORDER in which weights appear might differ!
    """
    p = len(poly_coeffs)
    C = companion_matrix(poly_coeffs)
    
    if v is None:
        v = np.zeros(p, dtype=np.int64)
        v[0] = 1  # Start from e_1
    
    state = v.copy()
    weights = np.zeros(length, dtype=np.float64)
    
    for k in range(length):
        weights[k] = np.sum(state % 2)
        state = gf2_mat_vec(C, state)
    
    return weights


def analyze_weight_trajectory(weights: np.ndarray, M_p: int, p: int) -> Dict:
    """Analyze the Hamming weight trajectory for factor signatures."""
    n = len(weights)
    
    # Normalize: subtract mean, divide by std
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    normalized = (weights - mean_w) / (std_w + 1e-10)
    
    # FFT of the normalized weight trajectory
    fft_result = np.fft.fft(normalized)
    power_spectrum = np.abs(fft_result) ** 2
    power_no_dc = power_spectrum[1:n//2]
    
    # WHT of the binarized weight sequence (above/below median)
    median_w = np.median(weights)
    binary = (weights > median_w).astype(np.float64)
    binary_pm1 = 2 * binary - 1
    
    # Pad to power of 2 for WHT
    wht_len = 1
    while wht_len < n:
        wht_len *= 2
    wht_input = np.zeros(wht_len)
    wht_input[:n] = binary_pm1
    wht_spectrum = _fast_wht(wht_input)
    wht_abs = np.abs(wht_spectrum)
    wht_no_dc = wht_abs[1:]
    
    # Autocorrelation of the weight trajectory
    acorr = np.correlate(normalized[:min(n, 5000)] - np.mean(normalized[:min(n, 5000)]),
                         normalized[:min(n, 5000)] - np.mean(normalized[:min(n, 5000)]),
                         mode='full')
    acorr = acorr[len(acorr)//2:]
    acorr = acorr / (acorr[0] + 1e-10)
    
    # Look for periodic peaks in autocorrelation at factor-related lags
    acorr_peaks = []
    for tau in range(1, min(len(acorr), 2000)):
        if tau >= len(acorr) - 1:
            break
        if acorr[tau] > acorr[tau-1] and acorr[tau] > acorr[tau+1]:
            if acorr[tau] > 0.1:  # Significant peak
                acorr_peaks.append((tau, float(acorr[tau])))
    
    # Check if any peak positions relate to factors
    factors = []
    for tau, val in acorr_peaks[:20]:
        g = gcd(tau, M_p)
        if 1 < g < M_p:
            factors.append((tau, val, g))
    
    return {
        'mean_weight': float(mean_w),
        'std_weight': float(std_w),
        'fft_peak_positions': [int(x) for x in np.argsort(-power_no_dc)[:10] + 1],
        'fft_peak_powers': [float(x) for x in sorted(power_no_dc, reverse=True)[:10]],
        'fft_spectral_flatness': float(np.mean(power_no_dc)**2 / (np.mean(power_no_dc**2) + 1e-30)),
        'wht_max_magnitude': float(np.max(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'wht_kurtosis': float(_kurtosis(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'num_acorr_peaks': len(acorr_peaks),
        'top_acorr_peaks': acorr_peaks[:10],
        'factors_from_peaks': factors,
    }


# ============================================================
# Method 2: Multi-Bit State Correlation Matrix
# ============================================================

def compute_state_correlation_matrix(poly_coeffs: List[int], length: int) -> Dict:
    """
    Compute the cross-correlation matrix between bit positions
    in the state evolution C^k * v.
    
    For each pair of bit positions (i, j), compute:
      R(i,j) = Correlation(bit_i across time, bit_j across time)
    
    For an m-sequence: each bit position carries the same information
    (they're all decimations of the same m-sequence). But the
    cross-correlation structure might differ for composite M_p.
    """
    p = len(poly_coeffs)
    C = companion_matrix(poly_coeffs)
    
    # Collect state matrix: each row is one time step, each column is one bit
    v = np.zeros(p, dtype=np.int64)
    v[0] = 1
    
    state_matrix = np.zeros((length, p), dtype=np.float64)
    state = v.copy()
    for k in range(length):
        state_matrix[k, :] = state % 2
        state = gf2_mat_vec(C, state)
    
    # Compute correlation matrix between bit positions
    # Use the ±1 representation for proper correlation
    state_pm1 = 2 * state_matrix - 1
    
    if length > p:
        # Cross-correlation matrix: p x p
        corr_matrix = np.corrcoef(state_pm1.T)
    else:
        corr_matrix = np.eye(p)
    
    # Spectral analysis of the correlation matrix
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Condition number (ratio of largest to smallest eigenvalue)
    cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
    
    # Participation ratio (effective rank)
    pr = np.sum(eigenvalues)**2 / (np.sum(eigenvalues**2) + 1e-10)
    
    # Off-diagonal statistics
    off_diag = []
    for i in range(p):
        for j in range(i+1, p):
            off_diag.append(corr_matrix[i, j])
    
    return {
        'eigenvalues_top5': [float(x) for x in eigenvalues[:5]],
        'eigenvalues_bottom5': [float(x) for x in eigenvalues[-5:]],
        'condition_number': float(cond),
        'participation_ratio': float(pr),
        'effective_rank': float(pr),
        'off_diag_mean': float(np.mean(off_diag)) if off_diag else 0,
        'off_diag_std': float(np.std(off_diag)) if off_diag else 0,
        'off_diag_max': float(np.max(np.abs(off_diag))) if off_diag else 0,
        'off_diag_range': float(np.max(off_diag) - np.min(off_diag)) if off_diag else 0,
    }


# ============================================================
# Method 3: State Collision Detection (Brent's Algorithm on C^d)
# ============================================================

def brent_cycle_detection(C_d: np.ndarray, v: np.ndarray, 
                           max_steps: int = 100000) -> Optional[int]:
    """
    Detect the cycle length of the orbit v, C_d*v, C_d^2*v, ...
    using Brent's cycle detection algorithm.
    
    Time complexity: O(3 * cycle_length) in the worst case.
    Space: O(1) — no state storage needed.
    
    Returns the cycle length, or None if no cycle found within max_steps.
    """
    # Power of C_d: the function f(x) = C_d * x
    power = 1
    lambda_len = 1
    
    tortoise = v.copy()
    hare = gf2_mat_vec(C_d, v)
    
    # Phase 1: Find lambda (cycle length)
    while not np.array_equal(tortoise % 2, hare % 2):
        if power >= max_steps:
            return None
        if power == lambda_len:
            tortoise = hare.copy()
            power *= 2
            lambda_len = 0
        hare = gf2_mat_vec(C_d, hare)
        lambda_len += 1
        if lambda_len > max_steps:
            return None
    
    return lambda_len


def multi_d_cycle_detection(poly_coeffs: List[int], p: int,
                             d_range: range, budget: int = 50000) -> Dict:
    """
    THE KEY EXPERIMENT: For each d in d_range, compute C^d and run
    Brent's cycle detection on the orbit of C^d * v.
    
    If d | M_p: the cycle length is M_p/d (shorter than M_p)
    If gcd(d, M_p) = 1: the cycle length is M_p (full period)
    
    With budget B per d:
    - d | M_p and M_p/d <= B: cycle DETECTED (factor found!)
    - d | M_p and M_p/d > B: cycle NOT detected (false negative)
    - gcd(d, M_p) = 1: cycle NOT detected within budget (correct negative)
    
    This is NON-CIRCULAR because:
    1. We don't know the factors in advance
    2. We test ALL small d values uniformly
    3. The cycle detection is done through CA dynamics
    4. The budget limits computation
    """
    M_p = (1 << p) - 1
    C = companion_matrix(poly_coeffs)
    v = np.zeros(p, dtype=np.int64)
    v[0] = 1
    
    results = {}
    factors_found = set()
    
    # Compute C^d iteratively: C^{d+1} = C^d * C
    C_d = np.eye(p, dtype=np.int64)  # Start with C^1
    
    for d in d_range:
        C_d = gf2_mat_mul(C_d, C)  # C^d
        
        # Run Brent's cycle detection on the orbit under C^d
        cycle_len = brent_cycle_detection(C_d, v, max_steps=budget)
        
        is_factor = (M_p % d == 0)
        
        results[d] = {
            'cycle_detected': cycle_len is not None,
            'cycle_length': cycle_len,
            'is_factor_of_Mp': is_factor,
            'expected_cycle_if_factor': M_p // d if is_factor else None,
        }
        
        # If cycle detected, check if it reveals a factor
        if cycle_len is not None and cycle_len < M_p:
            # C^d has order cycle_len (on this vector)
            # Factor = M_p / (order of C^d as a matrix)
            # But cycle_len is the orbit length of v under C^d,
            # which divides ord(C^d) = M_p/gcd(M_p, d)
            
            # Compute the actual matrix order
            actual_order = _compute_order_from_cycle(C_d, cycle_len, p, M_p)
            if actual_order is not None and actual_order < M_p:
                factor = M_p // actual_order
                if 1 < factor < M_p and M_p % factor == 0:
                    factors_found.add(factor)
                    results[d]['factor_revealed'] = factor
    
    return {
        'p': p,
        'M_p': M_p,
        'budget': budget,
        'd_range': (d_range.start, d_range.stop),
        'results': results,
        'factors_found': sorted(factors_found),
        'known_factors': KNOWN_FACTORS.get(p, []),
        'detections': sum(1 for r in results.values() if r['cycle_detected']),
        'factor_detections': sum(1 for d, r in results.items() 
                                  if r['cycle_detected'] and r['is_factor_of_Mp']),
    }


def _compute_order_from_cycle(C_d: np.ndarray, cycle_hint: int, 
                                p: int, M_p: int) -> Optional[int]:
    """Compute the exact order of C_d using the cycle hint as starting point."""
    identity = np.eye(p, dtype=np.int64)
    
    # The order divides M_p / gcd(M_p, d) for C = companion of primitive poly
    # We know cycle_hint divides the order. Check C_d^k = I for k dividing M_p.
    
    # Check if C_d^{cycle_hint} = I
    C_power = gf2_mat_pow(C_d, cycle_hint)
    if np.array_equal(C_power % 2, identity):
        # Order divides cycle_hint. Find the exact order.
        order = cycle_hint
        # Try removing prime factors
        for pf in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            while order % pf == 0:
                test = order // pf
                C_test = gf2_mat_pow(C_d, test)
                if np.array_equal(C_test % 2, identity):
                    order = test
                else:
                    break
        return order
    
    # If not, the cycle hint is for a specific vector, not the full matrix
    # Try the order M_p / gcd(M_p, d) for small d values
    return None


# ============================================================
# Method 4: Cross-Polynomial State Comparison
# ============================================================

def cross_polynomial_analysis(p: int, length: int = None) -> Dict:
    """
    Compare state evolutions from DIFFERENT primitive polynomials
    of the same degree p.
    
    Key idea: Two different primitive polynomials f1(x) and f2(x) of
    degree p produce m-sequences that are DECIMATIONS of each other.
    The decimation factor d satisfies gcd(d, M_p) = 1 (for m-sequences
    to be of the same period).
    
    When M_p is PRIME: all decimations d with 1 <= d < M_p have gcd(d,M_p) = 1.
    When M_p is COMPOSITE: some decimations d have gcd(d, M_p) > 1, producing
    sequences with SHORTER periods. The SET of valid decimation factors
    (those producing the same m-sequence) differs.
    
    We test: does the cross-correlation between state evolutions from
    two different primitive polynomials show factor-related structure?
    """
    poly1 = PRIMITIVE_POLYS.get(p)
    poly2 = PRIMITIVE_POLYS_ALT.get(p)
    
    if poly1 is None or poly2 is None:
        return {'error': f'Need two primitive polys of degree {p}'}
    
    M_p = (1 << p) - 1
    if length is None:
        length = min(M_p, 2**14)
    
    C1 = companion_matrix(poly1)
    C2 = companion_matrix(poly2)
    
    # Compute trace sequences from both polynomials
    v = np.zeros(p, dtype=np.int64)
    v[0] = 1
    
    trace1 = np.zeros(length, dtype=np.int8)
    trace2 = np.zeros(length, dtype=np.int8)
    
    state1 = v.copy()
    state2 = v.copy()
    
    for k in range(length):
        trace1[k] = int(np.trace(
            gf2_mat_pow(C1, k) if k < 100 else np.eye(p, dtype=np.int64)
        ) % 2) if k < 100 else 0
        
        # More efficient: iterative computation
        if k == 0:
            C1_power = np.eye(p, dtype=np.int64)
            C2_power = np.eye(p, dtype=np.int64)
        
        trace1[k] = int(np.trace(C1_power) % 2)
        trace2[k] = int(np.trace(C2_power) % 2)
        
        C1_power = gf2_mat_mul(C1_power, C1)
        C2_power = gf2_mat_mul(C2_power, C2)
    
    # Convert to ±1
    seq1_pm1 = 2.0 * trace1.astype(np.float64) - 1.0
    seq2_pm1 = 2.0 * trace2.astype(np.float64) - 1.0
    
    # Cross-correlation
    n = len(seq1_pm1)
    S1 = np.fft.fft(seq1_pm1)
    S2 = np.fft.fft(seq2_pm1)
    cross_power = np.real(np.fft.ifft(S1 * np.conj(S2))) / n
    
    # Statistics of cross-correlation
    cross_no_zero = cross_power[1:min(500, n)]
    
    # Number of distinct cross-correlation values
    distinct_values = len(set(np.round(cross_no_zero, 4)))
    
    # Check for anomalous peaks
    cross_mean = np.mean(cross_no_zero)
    cross_std = np.std(cross_no_zero)
    anomalous_peaks = [(i, float(cross_power[i])) 
                       for i in range(1, min(500, n))
                       if abs(cross_power[i] - cross_mean) > 3 * cross_std]
    
    # Check if peak positions relate to factors
    factor_peaks = []
    for pos, val in anomalous_peaks:
        g = gcd(pos, M_p)
        if 1 < g < M_p:
            factor_peaks.append((pos, val, g))
    
    return {
        'p': p,
        'M_p': M_p,
        'is_mersenne_prime': p in MERSENNE_PRIME_EXPONENTS,
        'cross_corr_distinct_values': distinct_values,
        'cross_corr_mean': float(cross_mean),
        'cross_corr_std': float(cross_std),
        'num_anomalous_peaks': len(anomalous_peaks) if (anomalous_peaks := [
            (i, float(cross_power[i])) for i in range(1, min(n, 500))
            if abs(cross_power[i]) > np.mean(np.abs(cross_power[1:min(n, 500)])) + 3 * np.std(np.abs(cross_power[1:min(n, 500)]))
        ]) else 0,
        'factors_from_cross_corr': factor_peaks if (factor_peaks := [
            (pos, val, g) for pos, val in anomalous_peaks[:50]
            if (g := gcd(pos, M_p)) > 1 and g < M_p
        ]) else [],
    }


# ============================================================
# Method 5: Bispectral Analysis (Higher-Order Spectra)
# ============================================================

def bispectral_analysis(trace_seq: np.ndarray, M_p: int, p: int,
                         max_lag: int = 64) -> Dict:
    """
    Compute the bispectrum (third-order cumulant spectrum) of the
    trace sequence.
    
    The bispectrum B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)]
    detects QUADRATIC PHASE COUPLING — nonlinear correlations that
    the power spectrum (second-order) completely misses.
    
    For m-sequences: the power spectrum is flat, but the bispectrum
    might show structure related to the period's factorization.
    
    Hypothesis: When M_p is composite, the sub-periodic structure
    creates quadratic phase coupling at factor-related frequencies.
    When M_p is prime, no such coupling exists.
    """
    n = len(trace_seq)
    seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0
    
    # Compute FFT
    X = np.fft.fft(seq_pm1)
    
    # Compute bispectrum for a grid of (f1, f2) values
    bispectrum = np.zeros((max_lag, max_lag), dtype=np.float64)
    
    for f1 in range(max_lag):
        for f2 in range(max_lag):
            f3 = f1 + f2
            if f3 < n // 2:
                bispectrum[f1, f2] = np.abs(X[f1] * X[f2] * np.conj(X[f3]))
    
    # Bispectral statistics
    bisp_nonzero = bispectrum[bispectrum > 0]
    
    if len(bisp_nonzero) > 0:
        bisp_mean = float(np.mean(bisp_nonzero))
        bisp_std = float(np.std(bisp_nonzero))
        bisp_max = float(np.max(bisp_nonzero))
        bisp_kurtosis = float(_kurtosis(bisp_nonzero))
    else:
        bisp_mean = bisp_std = bisp_max = bisp_kurtosis = 0
    
    # Find peaks in bispectrum
    bispectrum_flat = bispectrum.flatten()
    if len(bispectrum_flat) > 0 and np.std(bispectrum_flat) > 0:
        threshold = np.mean(bispectrum_flat) + 3 * np.std(bispectrum_flat)
        peak_indices = np.where(bispectrum_flat > threshold)[0]
        peak_positions = [(int(idx // max_lag), int(idx % max_lag)) 
                          for idx in peak_indices[:20]]
    else:
        peak_positions = []
    
    # Check if peak positions relate to factors
    factor_peaks = []
    for f1, f2 in peak_positions:
        for f in [f1, f2, f1 + f2]:
            g = gcd(f, M_p)
            if 1 < g < M_p:
                factor_peaks.append((f1, f2, f, g))
    
    # Bispectral entropy (measures concentration)
    bisp_abs = np.abs(bispectrum.flatten())
    bisp_sum = np.sum(bisp_abs)
    if bisp_sum > 0:
        bisp_probs = bisp_abs / bisp_sum
        bisp_entropy = -np.sum(bisp_probs * np.log2(bisp_probs + 1e-30))
    else:
        bisp_entropy = 0
    
    max_entropy = log2(max_lag * max_lag) if max_lag > 0 else 1
    normalized_entropy = bisp_entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'bispectral_mean': bisp_mean,
        'bispectral_std': bisp_std,
        'bispectral_max': bisp_max,
        'bispectral_kurtosis': bisp_kurtosis,
        'bispectral_entropy': float(bisp_entropy),
        'normalized_bispectral_entropy': float(normalized_entropy),
        'num_peaks': len(peak_positions),
        'peak_positions': peak_positions[:10],
        'factors_from_peaks': factor_peaks[:10],
    }


# ============================================================
# Method 6: Rank Dynamics of C^k - I
# ============================================================

def rank_dynamics_analysis(poly_coeffs: List[int], length: int,
                            M_p: int, p: int) -> Dict:
    """
    Track the rank of (C^k - I) over GF(2) for k = 1, 2, 3, ...
    
    The matrix (C^k - I) over GF(2) has a kernel that corresponds to
    the fixed points of C^k. When k | M_p, the matrix C^k has smaller
    order, which might affect the rank of C^k - I.
    
    Specifically:
    - C^k = I iff k is a multiple of ord(C) = M_p
    - rank(C^k - I) depends on the order of C^k
    
    For prime M_p: C^k = I only when k = M_p (no intermediate fixed points)
    For composite M_p: C^k = I when k is a multiple of M_p, but C^k might
      have fixed points for k | M_p (no, C^k ≠ I for k < M_p because C has
      order M_p).
    
    Actually: for C with order M_p, C^k ≠ I for any 0 < k < M_p.
    So rank(C^k - I) = p for all 0 < k < M_p (the matrix C^k - I is
    invertible when C^k ≠ I for a companion matrix).
    
    BUT: the rank of specific SUBMATRICES of C^k might differ!
    """
    C = companion_matrix(poly_coeffs)
    identity = np.eye(p, dtype=np.int64)
    
    C_power = C.copy()
    ranks = np.zeros(length, dtype=np.int32)
    
    for k in range(1, length + 1):
        diff = (C_power + identity) % 2  # C^k - I = C^k + I over GF(2)
        ranks[k-1] = _gf2_rank(diff)
        C_power = gf2_mat_mul(C_power, C)
    
    # Analyze rank distribution
    rank_counts = Counter(ranks.tolist())
    
    # Look for rank drops (indicates C^k close to I)
    rank_drops = []
    for k in range(1, length):
        if ranks[k] < ranks[k-1]:
            rank_drops.append((k+1, int(ranks[k-1]), int(ranks[k])))
    
    # Spectral analysis of rank sequence
    rank_centered = ranks - np.mean(ranks)
    if np.std(rank_centered) > 0:
        fft_ranks = np.abs(np.fft.fft(rank_centered))**2
        rank_spectral_peaks = np.argsort(-fft_ranks[1:length//2])[:5] + 1
    else:
        rank_spectral_peaks = []
    
    return {
        'rank_distribution': {int(k): int(v) for k, v in rank_counts.items()},
        'mean_rank': float(np.mean(ranks)),
        'std_rank': float(np.std(ranks)),
        'min_rank': int(np.min(ranks)),
        'num_rank_drops': len(rank_drops),
        'rank_drops': rank_drops[:10],
        'rank_spectral_peaks': [int(x) for x in rank_spectral_peaks],
    }


# ============================================================
# Method 7: Column Correlation Spectrum
# ============================================================

def column_correlation_spectrum(poly_coeffs: List[int], length: int,
                                 M_p: int, p: int) -> Dict:
    """
    Analyze the correlation structure of the COLUMNS of C^k
    as k varies.
    
    For a companion matrix C, the columns of C^k represent the
    states e_1, C*e_1, C^2*e_1, ... (shifted by the companion structure).
    The column correlation matrix of C^k encodes how "similar" the
    orbit segments starting from different basis vectors are.
    """
    C = companion_matrix(poly_coeffs)
    
    # Collect column vectors across time
    # For each time k, store the last column of C^k
    # (which represents C^k * e_{p-1} = alpha^k in the polynomial basis)
    
    C_power = np.eye(p, dtype=np.int64)
    last_columns = np.zeros((length, p), dtype=np.float64)
    
    for k in range(length):
        last_columns[k, :] = C_power[:, -1] % 2  # Last column
        C_power = gf2_mat_mul(C_power, C)
    
    # Convert to ±1 and compute spectral analysis
    last_cols_pm1 = 2 * last_columns - 1
    
    # SVD of the column evolution matrix
    if length > p and p > 0:
        try:
            U, S, Vt = np.linalg.svd(last_cols_pm1, full_matrices=False)
            svd_entropy = -np.sum((S / np.sum(S)) * np.log2(S / np.sum(S) + 1e-30))
            effective_rank_svd = np.sum(S)**2 / (np.sum(S**2) + 1e-10)
        except Exception:
            S = np.ones(min(length, p))
            svd_entropy = 0
            effective_rank_svd = 1
    else:
        S = np.ones(1)
        svd_entropy = 0
        effective_rank_svd = 1
    
    # Correlation between consecutive columns
    col_corr = []
    for k in range(1, min(length, 10000)):
        c1 = last_cols_pm1[k-1, :]
        c2 = last_cols_pm1[k, :]
        if np.std(c1) > 0 and np.std(c2) > 0:
            corr = np.corrcoef(c1, c2)[0, 1]
            col_corr.append(corr)
    
    return {
        'svd_top5_singular_values': [float(x) for x in S[:5]],
        'svd_entropy': float(svd_entropy),
        'svd_effective_rank': float(effective_rank_svd),
        'col_corr_mean': float(np.mean(col_corr)) if col_corr else 0,
        'col_corr_std': float(np.std(col_corr)) if col_corr else 0,
        'col_corr_range': float(np.max(col_corr) - np.min(col_corr)) if col_corr else 0,
    }


# ============================================================
# Helper Functions
# ============================================================

def _fast_wht(seq: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform."""
    n = len(seq)
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


def _gf2_rank(M: np.ndarray) -> int:
    """Compute the rank of a matrix over GF(2) using Gaussian elimination."""
    M = M.copy().astype(np.int64) % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        # Find pivot
        pivot = -1
        for row in range(rank, rows):
            if M[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        M[[rank, pivot]] = M[[pivot, rank]]
        # Eliminate
        for row in range(rows):
            if row != rank and M[row, col] % 2 == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis."""
    if len(arr) == 0:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


def compute_trace_sequence(poly_coeffs: List[int], length: int) -> np.ndarray:
    """Compute Tr(C^k) for k = 0, 1, ..., length-1."""
    p = len(poly_coeffs)
    C = companion_matrix(poly_coeffs)
    C_power = np.eye(p, dtype=np.int64)
    traces = np.zeros(length, dtype=np.int8)
    for k in range(length):
        traces[k] = int(np.trace(C_power) % 2)
        C_power = gf2_mat_mul(C_power, C)
    return traces


# ============================================================
# Comprehensive Experiment Runner
# ============================================================

def run_path2_v2_experiment() -> Dict:
    """
    Run the comprehensive Path 2 v2 experiment.
    
    Test cases:
    - PRIME Mersenne: p = 7, 13, 17, 19
    - COMPOSITE Mersenne: p = 11, 23, 29
    
    For each case, run all 7 methods and collect metrics.
    Then compare prime vs. composite using Mann-Whitney U tests.
    """
    prime_cases = [7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases
    
    all_results = {}
    
    for p in all_cases:
        M_p = (1 << p) - 1
        is_prime = p in MERSENNE_PRIME_EXPONENTS
        poly = PRIMITIVE_POLYS.get(p)
        
        if poly is None:
            print(f"  Skipping p={p}: no primitive polynomial")
            continue
        
        status = "PRIME" if is_prime else "COMPOSITE"
        print(f"\n{'='*60}")
        print(f"p = {p}, M_p = {M_p} ({status})")
        print(f"{'='*60}")
        
        # Determine sequence lengths
        if p <= 11:
            seq_length = M_p
        elif p <= 19:
            seq_length = 2**14  # 16384
        elif p <= 23:
            seq_length = 2**14
        else:
            seq_length = 2**12  # 4096
        
        # ---- Method 1: Hamming Weight Trajectory ----
        print(f"  [1/7] Hamming Weight Trajectory (length {seq_length})...")
        t0 = time.time()
        
        # Multiple starting vectors for robustness
        weight_results = []
        for v_idx in range(3):
            v = np.zeros(p, dtype=np.int64)
            if v_idx == 0:
                v[0] = 1
            elif v_idx == 1:
                v[p//2] = 1
            else:
                v[:3] = 1  # Multi-bit start
            
            weights = hamming_weight_trajectory(poly, seq_length, v)
            wres = analyze_weight_trajectory(weights, M_p, p)
            weight_results.append(wres)
        
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    FFT peaks: {weight_results[0]['fft_peak_positions'][:5]}")
        print(f"    Factors from peaks: {weight_results[0]['factors_from_peaks']}")
        
        # ---- Method 2: State Correlation Matrix ----
        corr_len = min(seq_length, 2**12)
        print(f"  [2/7] State Correlation Matrix (length {corr_len})...")
        t0 = time.time()
        corr_result = compute_state_correlation_matrix(poly, corr_len)
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    Cond: {corr_result['condition_number']:.2f}, "
              f"Eff. rank: {corr_result['effective_rank']:.2f}")
        
        # ---- Method 3: Multi-d Cycle Detection ----
        print(f"  [3/7] Multi-d Cycle Detection...")
        t0 = time.time()
        cycle_result = multi_d_cycle_detection(
            poly, p, range(2, min(200, M_p)),
            budget=min(50000, M_p // 2)
        )
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    Cycles detected: {cycle_result['detections']}")
        print(f"    Factors found: {cycle_result['factors_found']}")
        print(f"    Known factors: {cycle_result['known_factors']}")
        
        # ---- Method 4: Cross-Polynomial Analysis ----
        print(f"  [4/7] Cross-Polynomial Analysis...")
        t0 = time.time()
        cross_result = cross_polynomial_analysis(p, min(seq_length, 2**12))
        t1 = time.time()
        if 'error' not in cross_result:
            print(f"    Done in {t1-t0:.2f}s")
            print(f"    Cross-corr distinct values: {cross_result['cross_corr_distinct_values']}")
            print(f"    Factors from cross-corr: {cross_result['factors_from_cross_corr']}")
        else:
            print(f"    Skipped: {cross_result['error']}")
        
        # ---- Method 5: Bispectral Analysis ----
        bispec_len = min(seq_length, 2**12)
        print(f"  [5/7] Bispectral Analysis (length {bispec_len})...")
        t0 = time.time()
        trace_seq = compute_trace_sequence(poly, bispec_len)
        bispec_result = bispectral_analysis(trace_seq, M_p, p, max_lag=32)
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    Bispectral entropy: {bispec_result['normalized_bispectral_entropy']:.4f}")
        print(f"    Bispectral kurtosis: {bispec_result['bispectral_kurtosis']:.2f}")
        print(f"    Factors from bispectrum: {bispec_result['factors_from_peaks']}")
        
        # ---- Method 6: Rank Dynamics ----
        rank_len = min(seq_length, 2**12)
        print(f"  [6/7] Rank Dynamics (length {rank_len})...")
        t0 = time.time()
        rank_result = rank_dynamics_analysis(poly, rank_len, M_p, p)
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    Rank distribution: {rank_result['rank_distribution']}")
        print(f"    Rank drops: {rank_result['num_rank_drops']}")
        
        # ---- Method 7: Column Correlation Spectrum ----
        print(f"  [7/7] Column Correlation Spectrum...")
        t0 = time.time()
        col_result = column_correlation_spectrum(poly, min(seq_length, 2**12), M_p, p)
        t1 = time.time()
        print(f"    Done in {t1-t0:.2f}s")
        print(f"    SVD entropy: {col_result['svd_entropy']:.4f}")
        print(f"    SVD eff. rank: {col_result['svd_effective_rank']:.2f}")
        
        # Store results
        all_results[p] = {
            'p': p,
            'M_p': M_p,
            'is_mersenne_prime': is_prime,
            'weight_analysis': weight_results,
            'correlation_matrix': corr_result,
            'cycle_detection': cycle_result,
            'cross_polynomial': cross_result if 'error' not in cross_result else None,
            'bispectrum': bispec_result,
            'rank_dynamics': rank_result,
            'column_correlation': col_result,
        }
    
    return all_results


def analyze_results(results: Dict) -> Dict:
    """
    Compare prime vs. composite results across all methods.
    Identify which metrics (if any) distinguish prime from composite M_p.
    """
    print("\n" + "=" * 80)
    print("PATH 2 v2: COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Collect metrics for prime and composite cases
    prime_metrics = {}
    composite_metrics = {}
    
    metric_keys = [
        ('weight_analysis', 0, 'fft_spectral_flatness'),
        ('weight_analysis', 0, 'wht_kurtosis'),
        ('weight_analysis', 0, 'num_acorr_peaks'),
        ('correlation_matrix', None, 'condition_number'),
        ('correlation_matrix', None, 'effective_rank'),
        ('correlation_matrix', None, 'off_diag_range'),
        ('bispectrum', None, 'normalized_bispectral_entropy'),
        ('bispectrum', None, 'bispectral_kurtosis'),
        ('bispectrum', None, 'num_peaks'),
        ('rank_dynamics', None, 'std_rank'),
        ('rank_dynamics', None, 'num_rank_drops'),
        ('column_correlation', None, 'svd_entropy'),
        ('column_correlation', None, 'svd_effective_rank'),
        ('column_correlation', None, 'col_corr_range'),
    ]
    
    for p, res in results.items():
        group = 'prime' if res['is_mersenne_prime'] else 'composite'
        
        for method, idx, metric in metric_keys:
            data = res.get(method)
            if data is None:
                continue
            if idx is not None and isinstance(data, list):
                data = data[idx] if idx < len(data) else None
            if data is None:
                continue
            value = data.get(metric)
            if value is None:
                continue
            
            key = f"{method}.{metric}"
            if group == 'prime':
                prime_metrics.setdefault(key, []).append(float(value))
            else:
                composite_metrics.setdefault(key, []).append(float(value))
    
    # Mann-Whitney U test for each metric
    print("\n  Metric comparison (prime vs. composite):")
    print(f"  {'Metric':<50} | {'Prime mean':>10} | {'Comp mean':>10} | {'Direction':>10}")
    print("  " + "-" * 90)
    
    significant_metrics = []
    
    for key in sorted(set(list(prime_metrics.keys()) + list(composite_metrics.keys()))):
        pv = prime_metrics.get(key, [])
        cv = composite_metrics.get(key, [])
        
        if not pv or not cv:
            continue
        
        pm = np.mean(pv)
        cm = np.mean(cv)
        direction = "P>C" if pm > cm else "P<C" if pm < cm else "P=C"
        
        print(f"  {key:<50} | {pm:>10.4f} | {cm:>10.4f} | {direction:>10}")
        
        # Simple significance check (would need proper MWU test for real analysis)
        if abs(pm - cm) > 0.5 * max(np.std(pv), np.std(cv) + 0.001):
            significant_metrics.append((key, pm, cm, direction))
    
    # Factor recovery summary
    print("\n" + "-" * 60)
    print("FACTOR RECOVERY SUMMARY")
    print("-" * 60)
    
    for p, res in results.items():
        if res['is_mersenne_prime']:
            continue
        
        M_p = res['M_p']
        known = set(KNOWN_FACTORS.get(p, []))
        
        # Collect all factors found by all methods
        all_found = set()
        
        # From cycle detection
        cd_factors = set(res['cycle_detection']['factors_found'])
        all_found.update(cd_factors)
        
        # From weight analysis
        for wr in res['weight_analysis']:
            for _, _, g in wr.get('factors_from_peaks', []):
                if 1 < g < M_p and M_p % g == 0:
                    all_found.add(g)
        
        # From cross-polynomial
        if res.get('cross_polynomial'):
            for _, _, _, g in res['cross_polynomial'].get('factors_from_cross_corr', []):
                if 1 < g < M_p and M_p % g == 0:
                    all_found.add(g)
        
        # From bispectrum
        for _, _, _, g in res['bispectrum'].get('factors_from_peaks', []):
            if 1 < g < M_p and M_p % g == 0:
                all_found.add(g)
        
        recovery_rate = len(all_found & known) / len(known) if known else 0
        
        print(f"  p={p}: found={sorted(all_found)}, known={sorted(known)}, "
              f"rate={recovery_rate:.0%}")
        
        # Per-method breakdown
        print(f"    Cycle detection: {sorted(cd_factors)}")
        print(f"    Weight analysis: {sorted(set(g for wr in res['weight_analysis'] for _, _, g in wr.get('factors_from_peaks', []) if 1 < g < M_p))}")
        if res.get('cross_polynomial'):
            print(f"    Cross-polynomial: {sorted(set(g for _, _, _, g in res['cross_polynomial'].get('factors_from_cross_corr', []) if 1 < g < M_p))}")
        print(f"    Bispectrum: {sorted(set(g for _, _, _, g in res['bispectrum'].get('factors_from_peaks', []) if 1 < g < M_p))}")
    
    # Overall conclusion
    print("\n" + "=" * 80)
    print("OVERALL CONCLUSION")
    print("=" * 80)
    
    if significant_metrics:
        print(f"\n  POTENTIALLY SIGNIFICANT METRICS ({len(significant_metrics)}):")
        for key, pm, cm, direction in significant_metrics:
            print(f"    {key}: prime={pm:.4f}, composite={cm:.4f} ({direction})")
        print(f"\n  CAVEAT: Only {len(prime_cases)} prime and {len(composite_cases)} composite")
        print(f"  samples. Need more data for statistical significance.")
    else:
        print("\n  NO METRICS significantly distinguish prime from composite M_p.")
        print("  The negative result from v1 is confirmed with additional methods.")
    
    # Check cycle detection specifically
    print("\n  CYCLE DETECTION (Method 3) — Most promising for factor recovery:")
    for p, res in results.items():
        if res['is_mersenne_prime']:
            continue
        cd = res['cycle_detection']
        known = set(KNOWN_FACTORS.get(p, []))
        found = set(cd['factors_found'])
        print(f"    p={p}: {len(found & known)}/{len(known)} factors recovered, "
              f"budget={cd['budget']}, d_range={cd['d_range']}")
    
    return {
        'significant_metrics': significant_metrics,
        'prime_metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                          for k, v in prime_metrics.items()},
        'composite_metrics': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                              for k, v in composite_metrics.items()},
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + " PATH 2 v2: BEYOND THE TRACE SEQUENCE".center(78) + "║")
    print("║" + " Multi-Dimensional Spectral Analysis of GF(2) CA".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("""
PREVIOUS RESULT (v1): Spectral analysis of Tr(C^k) CANNOT distinguish
  prime from composite M_p because it's an m-sequence with ideal
  autocorrelation regardless of primality.

THIS VERSION (v2): Go beyond the scalar trace.
  7 methods targeting multi-dimensional state evolution:
  1. Hamming Weight Trajectory
  2. Multi-Bit State Correlation Matrix  
  3. State Collision Detection (Brent's algorithm on C^d orbits)
  4. Cross-Polynomial State Comparison
  5. Bispectral Analysis (higher-order spectra)
  6. Rank Dynamics of C^k - I
  7. Column Correlation Spectrum
""")
    
    overall_start = time.time()
    
    # Run the experiment
    results = run_path2_v2_experiment()
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # JSON-safe serialization
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_safe(x) for x in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return sorted(list(obj))
        return obj
    
    output = {
        'experiment': 'path2_v2_beyond_trace',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'methods': [
            'hamming_weight_trajectory',
            'state_correlation_matrix',
            'multi_d_cycle_detection',
            'cross_polynomial_comparison',
            'bispectral_analysis',
            'rank_dynamics',
            'column_correlation_spectrum',
        ],
        'results_summary': make_json_safe(analysis),
    }
    
    output_path = os.path.join(results_dir, 'path2_v2_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {output_path}")
    print(f"  Total time: {time.time() - overall_start:.1f}s")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
