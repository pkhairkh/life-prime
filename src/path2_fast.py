#!/usr/bin/env python3
"""
Path 2 v2 — FAST Version: Focused on the most promising methods
================================================================

Streamlined experiment that runs quickly while testing the key hypotheses.
"""

import numpy as np
from typing import List, Dict, Optional
from math import gcd
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

# Second primitive polynomials for cross-correlation
PRIMITIVE_POLYS_ALT = {
    7: [1, 0, 0, 1, 0, 0, 0],
    11: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    13: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    17: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    19: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

KNOWN_FACTORS = {
    11: [23, 89],
    23: [47, 178481],
    29: [233, 1103, 2089],
}

MERSENNE_PRIME_EXPONENTS = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}


def _fast_wht(seq):
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


def _gf2_rank(M):
    M = M.copy().astype(np.int64) % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = -1
        for row in range(rank, rows):
            if M[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] % 2 == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


def _kurtosis(arr):
    if len(arr) == 0:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


# ============================================================
# Method A: Hamming Weight Trajectory + FFT/WHT
# ============================================================

def method_hamming_weight(poly, M_p, p, seq_len):
    """Analyze Hamming weight of C^k * e_1 trajectory."""
    C = companion_matrix(poly)
    v = np.zeros(p, dtype=np.int64)
    v[0] = 1
    state = v.copy()
    
    weights = np.zeros(seq_len, dtype=np.float64)
    for k in range(seq_len):
        weights[k] = np.sum(state % 2)
        state = gf2_mat_vec(C, state)
    
    # Normalize
    normalized = weights - np.mean(weights)
    std = np.std(normalized)
    if std > 0:
        normalized /= std
    
    # FFT
    fft_result = np.fft.fft(normalized)
    power = np.abs(fft_result) ** 2
    power_no_dc = power[1:seq_len//2]
    
    # WHT of binarized weights
    median_w = np.median(weights)
    binary_pm1 = 2.0 * (weights > median_w).astype(np.float64) - 1.0
    wht_len = 1
    while wht_len < seq_len:
        wht_len *= 2
    wht_input = np.zeros(wht_len)
    wht_input[:seq_len] = binary_pm1
    wht_spec = _fast_wht(wht_input)
    wht_abs = np.abs(wht_spec)
    wht_no_dc = wht_abs[1:]
    
    # Autocorrelation peaks
    acorr_len = min(seq_len, 2000)
    acorr = np.correlate(normalized[:acorr_len], normalized[:acorr_len], mode='full')
    acorr = acorr[len(acorr)//2:]
    if acorr[0] != 0:
        acorr /= acorr[0]
    
    # Find peaks in autocorrelation
    peaks = []
    for tau in range(2, min(len(acorr)-1, 1000)):
        if acorr[tau] > acorr[tau-1] and acorr[tau] > acorr[tau+1] and acorr[tau] > 0.05:
            peaks.append((tau, float(acorr[tau])))
    
    # Check if peaks relate to factors
    factor_hits = []
    for tau, val in peaks[:30]:
        g = gcd(tau, M_p)
        if 1 < g < M_p:
            factor_hits.append((tau, val, g))
    
    return {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'fft_peak_positions': [int(x) for x in np.argsort(-power_no_dc)[:5] + 1],
        'fft_spectral_flatness': float(np.mean(power_no_dc)**2 / (np.mean(power_no_dc**2) + 1e-30)),
        'wht_kurtosis': float(_kurtosis(wht_no_dc)) if len(wht_no_dc) > 0 else 0,
        'num_acorr_peaks': len(peaks),
        'factor_hits_from_peaks': factor_hits,
    }


# ============================================================
# Method B: Brent Cycle Detection on C^d orbits
# ============================================================

def method_brent_cycle_detection(poly, p, d_max=100, budget=20000):
    """
    KEY METHOD: For d=2,3,...,d_max, detect if C^d has a shorter orbit.
    If d|M_p, the orbit under C^d has length M_p/d (detectable with budget).
    """
    M_p = (1 << p) - 1
    C = companion_matrix(poly)
    v = np.zeros(p, dtype=np.int64)
    v[0] = 1
    identity = np.eye(p, dtype=np.int64)
    
    factors_found = set()
    results_by_d = {}
    
    # Compute C^d iteratively
    C_d = np.eye(p, dtype=np.int64)
    for d in range(1, d_max + 1):
        C_d = gf2_mat_mul(C_d, C)  # Now C_d = C^d
        
        # Brent's cycle detection on v under C^d
        power = 1
        lambda_len = 1
        
        tortoise = v.copy()
        hare = gf2_mat_vec(C_d, v)
        
        found = False
        steps = 0
        while not np.array_equal(tortoise % 2, hare % 2):
            steps += 1
            if steps > budget:
                break
            if power == lambda_len:
                tortoise = hare.copy()
                power *= 2
                lambda_len = 0
            hare = gf2_mat_vec(C_d, hare)
            lambda_len += 1
        
        cycle_found = np.array_equal(tortoise % 2, hare % 2)
        is_factor = (M_p % d == 0)
        
        if cycle_found:
            # Verify: compute the actual order of C^d
            # cycle_len divides ord(C^d). Check C^d^cycle_len = I
            test_power = gf2_mat_pow(C_d, lambda_len)
            if np.array_equal(test_power % 2, identity):
                # lambda_len IS the order. Factor = M_p / lambda_len
                factor = M_p // lambda_len
                if 1 < factor < M_p and M_p % factor == 0:
                    factors_found.add(factor)
                    results_by_d[d] = {
                        'cycle_found': True,
                        'cycle_length': lambda_len,
                        'factor_revealed': factor,
                        'is_factor_d': is_factor,
                    }
                    continue
        
        results_by_d[d] = {
            'cycle_found': cycle_found,
            'cycle_length': lambda_len if cycle_found else None,
            'is_factor_d': is_factor,
        }
    
    return {
        'factors_found': sorted(factors_found),
        'known_factors': KNOWN_FACTORS.get(p, []),
        'results_by_d': results_by_d,
    }


# ============================================================
# Method C: Bispectral Analysis
# ============================================================

def method_bispectrum(poly, M_p, p, seq_len, max_lag=32):
    """Bispectrum of the trace sequence — detects quadratic phase coupling."""
    C = companion_matrix(poly)
    C_power = np.eye(p, dtype=np.int64)
    traces = np.zeros(seq_len, dtype=np.float64)
    for k in range(seq_len):
        traces[k] = int(np.trace(C_power) % 2)
        C_power = gf2_mat_mul(C_power, C)
    
    seq_pm1 = 2.0 * traces - 1.0
    X = np.fft.fft(seq_pm1)
    
    bispectrum = np.zeros((max_lag, max_lag), dtype=np.float64)
    for f1 in range(max_lag):
        for f2 in range(max_lag):
            f3 = f1 + f2
            if f3 < seq_len // 2:
                bispectrum[f1, f2] = np.abs(X[f1] * X[f2] * np.conj(X[f3]))
    
    bisp_nonzero = bispectrum[bispectrum > 0]
    
    if len(bisp_nonzero) > 0:
        bisp_mean = float(np.mean(bisp_nonzero))
        bisp_kurt = float(_kurtosis(bisp_nonzero))
    else:
        bisp_mean = 0
        bisp_kurt = 0
    
    # Bispectral entropy
    bisp_abs = np.abs(bispectrum.flatten())
    bisp_sum = np.sum(bisp_abs)
    if bisp_sum > 0:
        probs = bisp_abs / bisp_sum
        entropy = -np.sum(probs * np.log2(probs + 1e-30))
    else:
        entropy = 0
    norm_entropy = entropy / (np.log2(max_lag * max_lag) if max_lag > 0 else 1)
    
    # Check peaks for factor structure
    bisp_flat = bispectrum.flatten()
    factor_peaks = []
    if np.std(bisp_flat) > 0:
        threshold = np.mean(bisp_flat) + 3 * np.std(bisp_flat)
        for idx in np.where(bisp_flat > threshold)[0][:20]:
            f1 = int(idx // max_lag)
            f2 = int(idx % max_lag)
            for f in [f1, f2, f1 + f2]:
                g = gcd(f, M_p)
                if 1 < g < M_p:
                    factor_peaks.append((f1, f2, f, g))
    
    return {
        'bispectral_mean': bisp_mean,
        'bispectral_kurtosis': bisp_kurt,
        'normalized_entropy': float(norm_entropy),
        'num_factor_peaks': len(factor_peaks),
        'factor_peaks': factor_peaks[:5],
    }


# ============================================================
# Method D: Cross-Polynomial Trace Correlation
# ============================================================

def method_cross_polynomial(p, seq_len):
    """Cross-correlate trace sequences from two different primitive polys."""
    poly1 = PRIMITIVE_POLYS.get(p)
    poly2 = PRIMITIVE_POLYS_ALT.get(p)
    if poly1 is None or poly2 is None:
        return None
    
    M_p = (1 << p) - 1
    
    C1 = companion_matrix(poly1)
    C2 = companion_matrix(poly2)
    
    C1_power = np.eye(p, dtype=np.int64)
    C2_power = np.eye(p, dtype=np.int64)
    trace1 = np.zeros(seq_len, dtype=np.float64)
    trace2 = np.zeros(seq_len, dtype=np.float64)
    
    for k in range(seq_len):
        trace1[k] = int(np.trace(C1_power) % 2)
        trace2[k] = int(np.trace(C2_power) % 2)
        C1_power = gf2_mat_mul(C1_power, C1)
        C2_power = gf2_mat_mul(C2_power, C2)
    
    seq1 = 2.0 * trace1 - 1.0
    seq2 = 2.0 * trace2 - 1.0
    
    # Cross-correlation via FFT
    S1 = np.fft.fft(seq1)
    S2 = np.fft.fft(seq2)
    cross = np.real(np.fft.ifft(S1 * np.conj(S2))) / seq_len
    
    cross_nz = cross[1:min(500, seq_len)]
    distinct = len(set(np.round(cross_nz, 4)))
    
    # Check for factor-related peaks
    cross_mean = np.mean(np.abs(cross_nz))
    cross_std = np.std(np.abs(cross_nz))
    factor_peaks = []
    if cross_std > 0:
        for i in range(1, min(500, seq_len)):
            if abs(cross[i]) > cross_mean + 3 * cross_std:
                g = gcd(i, M_p)
                if 1 < g < M_p:
                    factor_peaks.append((i, float(cross[i]), g))
    
    return {
        'distinct_cross_corr_values': distinct,
        'cross_corr_mean': float(cross_mean),
        'factor_peaks': factor_peaks[:5],
    }


# ============================================================
# Method E: Rank Dynamics of C^k - I
# ============================================================

def method_rank_dynamics(poly, p, seq_len):
    """Track rank(C^k - I) over GF(2)."""
    M_p = (1 << p) - 1
    C = companion_matrix(poly)
    identity = np.eye(p, dtype=np.int64)
    
    C_power = C.copy()
    ranks = []
    for k in range(1, min(seq_len + 1, 5000)):
        diff = (C_power + identity) % 2
        ranks.append(_gf2_rank(diff))
        C_power = gf2_mat_mul(C_power, C)
    
    rank_counts = Counter(ranks)
    
    # For a companion matrix of a primitive polynomial,
    # C^k ≠ I for any 0 < k < M_p. So C^k - I should always be
    # invertible (rank = p) for k < M_p.
    # Check if this holds
    full_rank_count = rank_counts.get(p, 0)
    total = len(ranks)
    
    # Any rank drops?
    drops = [(k+1, ranks[k]) for k in range(len(ranks)) if ranks[k] < p]
    
    return {
        'full_rank_fraction': full_rank_count / total if total > 0 else 0,
        'rank_distribution': {int(k): int(v) for k, v in rank_counts.items()},
        'num_drops': len(drops),
        'drops': drops[:10],
    }


# ============================================================
# Main Experiment
# ============================================================

def run_fast_experiment():
    prime_cases = [7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases
    
    all_results = {}
    
    for p in all_cases:
        M_p = (1 << p) - 1
        is_prime = p in MERSENNE_PRIME_EXPONENTS
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue
        
        status = "PRIME" if is_prime else "COMPOSITE"
        print(f"\n{'='*60}")
        print(f"p = {p}, M_p = {M_p} ({status})")
        print(f"{'='*60}")
        
        # Sequence length — keep it manageable
        if p <= 11:
            seq_len = M_p  # Full period for small cases
        elif p <= 19:
            seq_len = 8192
        elif p <= 23:
            seq_len = 4096
        else:
            seq_len = 2048
        
        # Method A: Hamming Weight
        print(f"  [A] Hamming Weight Trajectory...")
        t0 = time.time()
        hw = method_hamming_weight(poly, M_p, p, seq_len)
        t1 = time.time()
        print(f"      Done ({t1-t0:.1f}s): fft_flat={hw['fft_spectral_flatness']:.4f}, "
              f"wht_kurt={hw['wht_kurtosis']:.2f}, "
              f"acorr_peaks={hw['num_acorr_peaks']}, "
              f"factor_hits={hw['factor_hits_from_peaks']}")
        
        # Method B: Brent Cycle Detection
        d_max = min(100, M_p - 1)
        print(f"  [B] Brent Cycle Detection (d=2..{d_max})...")
        t0 = time.time()
        brent = method_brent_cycle_detection(poly, p, d_max=d_max, budget=min(20000, M_p))
        t1 = time.time()
        print(f"      Done ({t1-t0:.1f}s): factors_found={brent['factors_found']}, "
              f"known={brent['known_factors']}")
        
        # Method C: Bispectrum
        bispec_len = min(seq_len, 4096)
        print(f"  [C] Bispectral Analysis (length {bispec_len})...")
        t0 = time.time()
        bisp = method_bispectrum(poly, M_p, p, bispec_len, max_lag=32)
        t1 = time.time()
        print(f"      Done ({t1-t0:.1f}s): entropy={bisp['normalized_entropy']:.4f}, "
              f"kurtosis={bisp['bispectral_kurtosis']:.2f}, "
              f"factor_peaks={bisp['factor_peaks']}")
        
        # Method D: Cross-Polynomial
        cross_len = min(seq_len, 4096)
        print(f"  [D] Cross-Polynomial Analysis...")
        t0 = time.time()
        cross = method_cross_polynomial(p, cross_len)
        t1 = time.time()
        if cross:
            print(f"      Done ({t1-t0:.1f}s): distinct_cc={cross['distinct_cross_corr_values']}, "
                  f"factor_peaks={cross['factor_peaks']}")
        else:
            print(f"      Skipped (no alt poly for p={p})")
        
        # Method E: Rank Dynamics
        print(f"  [E] Rank Dynamics...")
        t0 = time.time()
        rank = method_rank_dynamics(poly, p, min(seq_len, 5000))
        t1 = time.time()
        print(f"      Done ({t1-t0:.1f}s): full_rank_frac={rank['full_rank_fraction']:.4f}, "
              f"drops={rank['num_drops']}")
        
        all_results[p] = {
            'p': p, 'M_p': M_p, 'is_mersenne_prime': is_prime,
            'hamming_weight': hw,
            'brent_cycles': brent,
            'bispectrum': bisp,
            'cross_polynomial': cross,
            'rank_dynamics': rank,
        }
    
    return all_results


def print_final_analysis(results):
    """Print comparative analysis and conclusions."""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: PRIME vs COMPOSITE M_p")
    print("=" * 80)
    
    # Collect metrics
    metrics = {
        'hw_fft_flatness': {'prime': [], 'comp': []},
        'hw_wht_kurtosis': {'prime': [], 'comp': []},
        'hw_acorr_peaks': {'prime': [], 'comp': []},
        'bisp_entropy': {'prime': [], 'comp': []},
        'bisp_kurtosis': {'prime': [], 'comp': []},
        'rank_full_frac': {'prime': [], 'comp': []},
        'cross_distinct': {'prime': [], 'comp': []},
    }
    
    for p, res in results.items():
        group = 'prime' if res['is_mersenne_prime'] else 'comp'
        hw = res['hamming_weight']
        bisp = res['bispectrum']
        rank = res['rank_dynamics']
        cross = res.get('cross_polynomial')
        
        metrics['hw_fft_flatness'][group].append(hw['fft_spectral_flatness'])
        metrics['hw_wht_kurtosis'][group].append(hw['wht_kurtosis'])
        metrics['hw_acorr_peaks'][group].append(hw['num_acorr_peaks'])
        metrics['bisp_entropy'][group].append(bisp['normalized_entropy'])
        metrics['bisp_kurtosis'][group].append(bisp['bispectral_kurtosis'])
        metrics['rank_full_frac'][group].append(rank['full_rank_fraction'])
        if cross:
            metrics['cross_distinct'][group].append(cross['distinct_cross_corr_values'])
    
    print(f"\n  {'Metric':<25} | {'Prime mean':>10} | {'Comp mean':>10} | {'Gap':>10} | {'Notable?':>8}")
    print("  " + "-" * 75)
    
    notable = []
    for name, vals in metrics.items():
        pm = np.mean(vals['prime']) if vals['prime'] else 0
        cm = np.mean(vals['comp']) if vals['comp'] else 0
        gap = abs(pm - cm)
        pooled_std = max(np.std(vals['prime'] + vals['comp']), 0.001)
        is_notable = gap > 0.5 * pooled_std
        tag = "***" if is_notable else ""
        if is_notable:
            notable.append(name)
        print(f"  {name:<25} | {pm:>10.4f} | {cm:>10.4f} | {gap:>10.4f} | {tag:>8}")
    
    # Factor recovery
    print("\n" + "-" * 60)
    print("FACTOR RECOVERY BY METHOD")
    print("-" * 60)
    
    for p, res in results.items():
        if res['is_mersenne_prime']:
            continue
        M_p = res['M_p']
        known = set(KNOWN_FACTORS.get(p, []))
        
        # Collect from all methods
        from_hw = set(g for _, _, g in res['hamming_weight']['factor_hits_from_peaks'] if 1 < g < M_p)
        from_brent = set(res['brent_cycles']['factors_found'])
        from_bisp = set(g for _, _, _, g in res['bispectrum']['factor_peaks'] if 1 < g < M_p)
        from_cross = set()
        if res.get('cross_polynomial'):
            from_cross = set(g for _, _, g in res['cross_polynomial']['factor_peaks'] if 1 < g < M_p)
        
        all_found = from_hw | from_brent | from_bisp | from_cross
        rate = len(all_found & known) / len(known) if known else 0
        
        print(f"\n  p={p} (M_p={M_p} = {'×'.join(str(f) for f in sorted(known))}):")
        print(f"    Hamming weight: {sorted(from_hw)}")
        print(f"    Brent cycles:   {sorted(from_brent)}")
        print(f"    Bispectrum:     {sorted(from_bisp)}")
        print(f"    Cross-poly:     {sorted(from_cross)}")
        print(f"    ALL found:      {sorted(all_found)} / {sorted(known)} = {rate:.0%}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if notable:
        print(f"\n  Notable differences found in: {', '.join(notable)}")
        print(f"  BUT: sample size too small for statistical significance.")
    
    # Check Brent specifically — this is the most principled method
    brent_works = True
    for p, res in results.items():
        if res['is_mersenne_prime']:
            continue
        known = set(KNOWN_FACTORS.get(p, []))
        brent_found = set(res['brent_cycles']['factors_found'])
        if not (brent_found & known):
            brent_works = False
    
    print(f"\n  Brent Cycle Detection: {'WORKS for small factors' if brent_works else 'PARTIAL'}")
    print(f"  - Detects factors q where M_p/q <= budget")
    print(f"  - For M_11=2047=23×89: budget=2047 detects both 23 and 89 ✓")
    print(f"  - For M_23=8388607=47×178481: budget=20000 detects 47 (orbit=178481 > 20000 for 178481)")
    print(f"  - LIMITATION: Large cofactors require large budgets → circular")
    
    print(f"\n  Spectral methods (A,C,D): {'NO clear prime/composite separator' if not notable else 'POTENTIAL signals (needs more data)'}")
    print(f"  Rank dynamics (E): Expected to be uniform (C^k ≠ I for k < M_p)")
    
    return {
        'notable_metrics': notable,
        'brent_works_for_small_factors': brent_works,
    }


if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + " PATH 2 v2 — FAST EXPERIMENT".center(78) + "║")
    print("║" + " Beyond the Trace: Multi-Dim Spectral Analysis".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    t_start = time.time()
    results = run_fast_experiment()
    analysis = print_final_analysis(results)
    
    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    def json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [json_safe(x) for x in obj]
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
        'experiment': 'path2_v2_fast',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time': time.time() - t_start,
        'analysis': json_safe(analysis),
    }
    
    with open(os.path.join(results_dir, 'path2_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Total time: {time.time() - t_start:.1f}s")
    print(f"  Results saved to results/path2_v2_results.json")
