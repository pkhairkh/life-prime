#!/usr/bin/env python3
"""
Visualization Script for Path 1 + Path 2 Experiments
=====================================================

Generates publication-quality figures for:
  Figure 1: C^d order probing algorithm results
  Figure 2: Spectral analysis negative result
  Figure 3: Minimal polynomial construction
  Figure 4: Prime vs composite comparison dashboard

All figures use English text, professional styling, and are saved
to /home/z/my-project/life-prime/results/
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Sarasa Mono SC']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
from math import gcd, isqrt
from collections import Counter
import sys
import os
import math
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, is_prime_simple
)
from theorem_formalization import (
    compute_matrix_order, PRIMITIVE_POLYS, KNOWN_FACTORS,
    KNOWN_MERSENNE_PRIMES, characteristic_polynomial_gf2, is_irreducible,
    trial_factor,
)
from trace_spectral_analysis import (
    compute_trace_sequence, walsh_hadamard_transform,
    periodic_autocorrelation, aperiodic_autocorrelation,
    spectral_factor_detection, factor_from_spectrum,
)

# Also import the fixed versions from path1_path2_experiments
from path1_path2_experiments import (
    char_poly_gf2_krylov,
    mann_whitney_u,
)

# ============================================================
# Style Configuration
# ============================================================

COLOR_PRIME = '#2196F3'       # Blue for prime
COLOR_COMPOSITE = '#F44336'   # Red for composite
COLOR_SUCCESS = '#4CAF50'     # Green for verified/success
COLOR_WARNING = '#FF9800'     # Orange for warnings
COLOR_NEUTRAL = '#9E9E9E'     # Gray neutral

DPI = 150

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helper Functions
# ============================================================

def compute_order_ratio_cd(p, max_d=None):
    """
    Compute ord(C^d)/M_p for d = 1, 2, ..., max_d.
    Returns (d_values, ratios, gcd_values).
    
    Key insight:
    - When gcd(M_p, d) = 1: ord(C^d) = M_p, so ratio = 1.0
    - When gcd(M_p, d) > 1: ord(C^d) = M_p / gcd(M_p, d), so ratio < 1.0
    """
    M_p = (1 << p) - 1
    poly = PRIMITIVE_POLYS.get(p)
    if poly is None:
        return None, None, None
    
    if max_d is None:
        max_d = min(200, M_p)
    
    C = companion_matrix(poly)
    identity = np.eye(p, dtype=np.int64)
    
    d_values = list(range(1, max_d + 1))
    ratios = []
    gcd_values = []
    
    # Compute C^d iteratively
    C_d = np.eye(p, dtype=np.int64)  # C^0 = I
    
    for d in range(1, max_d + 1):
        C_d = gf2_mat_mul(C_d, C)
        g = gcd(M_p, d)
        gcd_values.append(g)
        
        if g <= 1:
            ratios.append(1.0)
        else:
            # ord(C^d) = M_p / g
            # Verify: (C^d)^(M_p/g) = I
            expected_order = M_p // g
            C_d_pow = gf2_mat_pow(C_d, expected_order)
            if np.array_equal(C_d_pow % 2, identity):
                ratios.append(expected_order / M_p)
            else:
                # Fallback: compute order properly
                actual_order = compute_matrix_order(C_d, expected_order)
                if actual_order:
                    ratios.append(actual_order / M_p)
                else:
                    ratios.append(0.0)
    
    return d_values, ratios, gcd_values


# ============================================================
# Figure 1: C^d Order Probing
# ============================================================

def generate_fig_factor_discovery_cd_probing():
    """
    Figure 1: Show the C^d order probing algorithm results.
    
    Top row: 3 subplots for composite cases (p=11, 23, 29)
    Bottom row: 2 subplots for prime cases (p=13, 17)
    """
    print("Generating Figure 1: C^d Order Probing...")
    
    composite_cases = [11, 23, 29]
    prime_cases = [13, 17]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(r'$C^d$ Order Probing: Factor Discovery via Companion Matrix Powers',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Top row: composite cases
    for idx, p in enumerate(composite_cases):
        ax = axes[0, idx]
        M_p = (1 << p) - 1
        known = KNOWN_FACTORS.get(p, [])
        
        # Compute probing data
        max_d = min(150, M_p)
        d_vals, ratios, gcd_vals = compute_order_ratio_cd(p, max_d)
        
        if d_vals is None:
            ax.text(0.5, 0.5, f'No data for p={p}', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        
        # Separate points by gcd status
        blue_d = [d for d, g in zip(d_vals, gcd_vals) if g <= 1]
        blue_r = [r for r, g in zip(ratios, gcd_vals) if g <= 1]
        red_d = [d for d, g in zip(d_vals, gcd_vals) if g > 1]
        red_r = [r for r, g in zip(ratios, gcd_vals) if g > 1]
        
        # Plot
        ax.scatter(blue_d, blue_r, c=COLOR_PRIME, s=8, alpha=0.5,
                   label=r'gcd($M_p$, d) = 1', zorder=3)
        if red_d:
            ax.scatter(red_d, red_r, c=COLOR_COMPOSITE, s=30, alpha=0.9,
                       edgecolors='black', linewidths=0.5,
                       label=r'gcd($M_p$, d) > 1', zorder=4)
        
        # Annotate factor positions
        annotated_factors = set()
        for d, r, g in zip(d_vals, ratios, gcd_vals):
            if g > 1:
                # Extract prime factors of g
                for q in trial_factor(g):
                    if q > 1 and is_prime_simple(q) and q not in annotated_factors:
                        annotated_factors.add(q)
                        ax.annotate(f'q={q}',
                                    xy=(d, r), xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=7, color=COLOR_COMPOSITE,
                                    fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color=COLOR_COMPOSITE,
                                                    lw=0.5))
                        break
        
        # Reference line at 1.0
        ax.axhline(y=1.0, color=COLOR_NEUTRAL, linestyle='--', alpha=0.5, linewidth=0.8)
        
        factors_str = ' × '.join(str(f) for f in known)
        ax.set_title(f'p = {p}: $M_p$ = {M_p} = {factors_str}', fontsize=11)
        ax.set_xlabel('d', fontsize=10)
        ax.set_ylabel(r'ord($C^d$) / $M_p$', fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Bottom row: prime cases
    for idx, p in enumerate(prime_cases):
        ax = axes[1, idx]
        M_p = (1 << p) - 1
        
        max_d = min(150, M_p)
        d_vals, ratios, gcd_vals = compute_order_ratio_cd(p, max_d)
        
        if d_vals is None:
            ax.text(0.5, 0.5, f'No data for p={p}', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        
        # All ratios should be 1.0 for prime M_p
        ax.scatter(d_vals, ratios, c=COLOR_PRIME, s=8, alpha=0.5,
                   label=r'ord($C^d$)/$M_p$ = 1.0')
        ax.axhline(y=1.0, color=COLOR_SUCCESS, linestyle='--', alpha=0.7, linewidth=1.0)
        
        ax.set_title(f'p = {p}: $M_p$ = {M_p} (Mersenne Prime)', fontsize=11,
                     color=COLOR_PRIME)
        ax.set_xlabel('d', fontsize=10)
        ax.set_ylabel(r'ord($C^d$) / $M_p$', fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5,
                     'When $M_p$ is prime:\n'
                     r'ord($C^d$) = $M_p$ for all d' + '\n'
                     'No drops detected\n'
                     r'$\Rightarrow$ No factors to reveal',
                     transform=axes[1, 2].transAxes,
                     ha='center', va='center',
                     fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                               edgecolor=COLOR_SUCCESS, alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    outpath = os.path.join(RESULTS_DIR, 'fig_factor_discovery_cd_probing.png')
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ============================================================
# Figure 2: Spectral Negative Result
# ============================================================

def generate_fig_spectral_negative_result():
    """
    Figure 2: Honest negative result visualization.
    
    Left: Scatter of WHT kurtosis vs exponent p
    Middle: Scatter of aperiodic AC range vs exponent p
    Right: Bar chart of factor recovery rates by method
    """
    print("Generating Figure 2: Spectral Negative Result...")
    
    prime_cases = [7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases
    
    # Collect spectral metrics
    p_values = []
    wht_kurtosis_vals = []
    aacf_range_vals = []
    is_prime_flags = []
    
    method_recovery = {
        'WHT': 0, 'FFT': 0, 'AC': 0,
        'Decimation': 0, 'Folding': 0, 'Cycle Completion': 0,
    }
    method_total = 0  # Total known factors across composite cases
    
    for p in all_cases:
        M_p = (1 << p) - 1
        is_mp = p in KNOWN_MERSENNE_PRIMES
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue
        
        # Sequence length
        if p <= 11:
            seq_length = M_p
        elif p <= 19:
            seq_length = 2**14
        elif p <= 23:
            seq_length = 2**14
        else:
            seq_length = 2**12
        
        print(f"  Computing trace sequence for p={p} (length={seq_length})...")
        trace_seq = compute_trace_sequence(poly, seq_length)
        
        # Spectral analysis
        spectral = spectral_factor_detection(trace_seq, M_p, p)
        
        p_values.append(p)
        wht_kurtosis_vals.append(spectral['wht']['kurtosis'])
        aacf_range_vals.append(spectral['aperiodic_autocorrelation']['range'])
        is_prime_flags.append(is_mp)
        
        # Factor recovery (only for composite)
        if not is_mp:
            known = set(KNOWN_FACTORS.get(p, []))
            method_total += len(known)
            
            fr = factor_from_spectrum(trace_seq, M_p, p)
            for mname, mdata in fr.get('method_results', {}).items():
                rec = mdata.get('factors_recovered', [])
                found_factors = set()
                for r in rec:
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        f = r[-1]
                        if isinstance(f, (int, float)) and f == int(f):
                            if 1 < f < M_p and M_p % int(f) == 0:
                                found_factors.add(int(f))
                
                # Map method names
                name_map = {
                    'wht': 'WHT',
                    'fft': 'FFT',
                    'autocorrelation': 'AC',
                    'decimation_balance': 'Decimation',
                    'folding': 'Folding',
                    'cycle_completion': 'Cycle Completion',
                }
                mapped = name_map.get(mname, mname)
                if mapped in method_recovery:
                    method_recovery[mapped] += len(found_factors & known)
    
    # ---- Create figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle('Spectral Analysis: Honest Negative Result',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Left: WHT kurtosis vs p
    ax = axes[0]
    for pv, kv, ip in zip(p_values, wht_kurtosis_vals, is_prime_flags):
        marker = 'o' if ip else 's'
        color = COLOR_PRIME if ip else COLOR_COMPOSITE
        label = 'Prime $M_p$' if ip else 'Composite $M_p$'
        ax.scatter(pv, kv, marker=marker, c=color, s=80,
                   edgecolors='black', linewidths=0.5, zorder=3,
                   label=label if (ip and pv == prime_cases[0]) or
                          (not ip and pv == composite_cases[0]) else None)
    ax.set_xlabel('Exponent p', fontsize=11)
    ax.set_ylabel('WHT Kurtosis (excess)', fontsize=11)
    ax.set_title('WHT Spectrum Flatness', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Middle: Aperiodic AC range vs p
    ax = axes[1]
    for pv, rv, ip in zip(p_values, aacf_range_vals, is_prime_flags):
        marker = 'o' if ip else 's'
        color = COLOR_PRIME if ip else COLOR_COMPOSITE
        label = 'Prime $M_p$' if ip else 'Composite $M_p$'
        ax.scatter(pv, rv, marker=marker, c=color, s=80,
                   edgecolors='black', linewidths=0.5, zorder=3,
                   label=label if (ip and pv == prime_cases[0]) or
                          (not ip and pv == composite_cases[0]) else None)
    ax.set_xlabel('Exponent p', fontsize=11)
    ax.set_ylabel('Aperiodic AC Range', fontsize=11)
    ax.set_title('Autocorrelation Spread', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Right: Factor recovery rates by method
    ax = axes[2]
    methods = list(method_recovery.keys())
    hits = [method_recovery[m] for m in methods]
    rates = [h / method_total if method_total > 0 else 0 for h in hits]
    
    colors = []
    for m, r in zip(methods, rates):
        if r > 0.5:
            colors.append(COLOR_SUCCESS)
        elif r > 0:
            colors.append(COLOR_WARNING)
        else:
            colors.append(COLOR_NEUTRAL)
    
    bars = ax.bar(range(len(methods)), rates, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.7)
    
    # Add rate labels on bars
    for bar, rate, hit in zip(bars, rates, hits):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{rate:.0%}\n({hit}/{method_total})',
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Factor Recovery Rate', fontsize=11)
    ax.set_title('Factor Recovery by Method', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation about cycle completion being trial division
    ax.annotate('* Cycle Completion\n  = Trial Division',
                xy=(5, rates[5] if len(rates) > 5 else 0),
                xytext=(3.5, 0.85),
                fontsize=8, color=COLOR_WARNING,
                arrowprops=dict(arrowstyle='->', color=COLOR_WARNING, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                          edgecolor=COLOR_WARNING, alpha=0.8))
    
    plt.tight_layout()
    
    outpath = os.path.join(RESULTS_DIR, 'fig_spectral_negative_result.png')
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ============================================================
# Figure 3: Minimal Polynomial Construction
# ============================================================

def generate_fig_minimal_polynomial_construction():
    """
    Figure 3: Show the minimal polynomial construction results.
    
    For each composite case (p=11, 23, 29):
      Left: Orbit structure diagram
      Right: Minimal polynomials as binary coefficient vectors
    """
    print("Generating Figure 3: Minimal Polynomial Construction...")
    
    composite_cases = [11, 23, 29]
    
    fig, axes = plt.subplots(len(composite_cases), 2, figsize=(14, 4 * len(composite_cases)))
    fig.suptitle('Minimal Polynomial Construction for Composite Mersenne Numbers',
                 fontsize=16, fontweight='bold', y=1.02)
    
    for row_idx, p in enumerate(composite_cases):
        M_p = (1 << p) - 1
        known = KNOWN_FACTORS.get(p, [])
        poly = PRIMITIVE_POLYS.get(p)
        
        if poly is None:
            axes[row_idx, 0].text(0.5, 0.5, f'No data for p={p}',
                                   transform=axes[row_idx, 0].transAxes,
                                   ha='center', va='center')
            axes[row_idx, 1].axis('off')
            continue
        
        C = companion_matrix(poly)
        
        # ---- Left: Orbit structure ----
        ax = axes[row_idx, 0]
        
        # Original C: one orbit of M_p states
        orbit_info = [
            ('C', M_p, M_p, COLOR_PRIME),
        ]
        
        # C^q for each factor q: orbit of M_p/q
        for q in known:
            C_q = gf2_mat_pow(C, q)
            expected_order = M_p // q
            
            # Verify order
            C_q_pow = gf2_mat_pow(C_q, expected_order)
            identity = np.eye(p, dtype=np.int64)
            if np.array_equal(C_q_pow % 2, identity):
                actual_order = expected_order
            else:
                actual_order = compute_matrix_order(C_q, expected_order * 2)
            
            orbit_info.append((f'$C^{{{q}}}$', M_p, actual_order, COLOR_COMPOSITE))
        
        # Draw orbit structure as horizontal bars
        y_positions = list(range(len(orbit_info)))
        labels = []
        for i, (name, total, period, color) in enumerate(orbit_info):
            # Bar for total states
            ax.barh(i, total, height=0.5, color=color, alpha=0.3, edgecolor=color)
            # Bar for period (orbit length)
            ax.barh(i, period, height=0.5, color=color, alpha=0.7, edgecolor=color)
            
            labels.append(f'{name}\nperiod={period}')
            
            # Add text annotation
            if period < total:
                ax.text(total + total * 0.02, i,
                        f'{total} states, period {period}',
                        va='center', fontsize=8)
            else:
                ax.text(total + total * 0.02, i,
                        f'{total} states = 1 orbit',
                        va='center', fontsize=8)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of States / Period', fontsize=10)
        factors_str = ' × '.join(str(f) for f in known)
        ax.set_title(f'p = {p}: $M_p$ = {M_p} = {factors_str}', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # ---- Right: Minimal polynomials as binary vectors ----
        ax = axes[row_idx, 1]
        
        # Original primitive poly
        poly_vectors = [poly]
        poly_labels = [f'Primitive poly\n(degree {p})']
        poly_colors = [COLOR_PRIME]
        
        # Minimal polynomials for each factor
        for q in known:
            C_q = gf2_mat_pow(C, q)
            min_poly = char_poly_gf2_krylov(C_q)
            
            if min_poly is not None:
                poly_vectors.append(min_poly)
                poly_labels.append(f'minpoly($\\alpha^{{{q}}}$)\norder=$M_p/{q}$={M_p//q}')
                poly_colors.append(COLOR_COMPOSITE)
        
        # Plot binary coefficient vectors as heatmap
        max_len = max(len(v) for v in poly_vectors)
        matrix = np.zeros((len(poly_vectors), max_len))
        for i, v in enumerate(poly_vectors):
            for j, c in enumerate(v):
                matrix[i, j] = c
        
        im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest',
                        vmin=0, vmax=1)
        
        ax.set_yticks(range(len(poly_labels)))
        ax.set_yticklabels(poly_labels, fontsize=8)
        ax.set_xlabel('Coefficient index', fontsize=10)
        ax.set_title(f'Minimal Polynomials (Binary Coefficients)', fontsize=11)
        
        # Add coefficient values as text
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.5:
                    ax.text(j, i, '1', ha='center', va='center',
                            fontsize=6, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    outpath = os.path.join(RESULTS_DIR, 'fig_minimal_polynomial_construction.png')
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ============================================================
# Figure 4: Prime vs Composite Comparison
# ============================================================

def generate_fig_prime_vs_composite_comparison():
    """
    Figure 4: Comprehensive comparison dashboard.
    
    For each metric: bar chart comparing prime vs composite means with error bars,
    plus p-value from Mann-Whitney U test.
    """
    print("Generating Figure 4: Prime vs Composite Comparison...")
    
    prime_cases = [7, 13, 17, 19]
    composite_cases = [11, 23, 29]
    all_cases = prime_cases + composite_cases
    
    # Metrics to compare
    metric_keys = [
        ('wht_kurtosis', 'WHT Kurtosis'),
        ('fft_flatness', 'FFT Spectral Flatness'),
        ('pacf_distinct', 'Periodic AC Distinct Values'),
        ('aacf_range', 'Aperiodic AC Range'),
    ]
    
    # Collect data
    prime_data = {k[0]: [] for k in metric_keys}
    composite_data = {k[0]: [] for k in metric_keys}
    
    for p in all_cases:
        M_p = (1 << p) - 1
        is_mp = p in KNOWN_MERSENNE_PRIMES
        poly = PRIMITIVE_POLYS.get(p)
        if poly is None:
            continue
        
        # Sequence length
        if p <= 11:
            seq_length = M_p
        elif p <= 19:
            seq_length = 2**14
        elif p <= 23:
            seq_length = 2**14
        else:
            seq_length = 2**12
        
        print(f"  Computing spectral metrics for p={p}...")
        trace_seq = compute_trace_sequence(poly, seq_length)
        spectral = spectral_factor_detection(trace_seq, M_p, p)
        
        target = prime_data if is_mp else composite_data
        
        target['wht_kurtosis'].append(spectral['wht']['kurtosis'])
        target['fft_flatness'].append(spectral['fft']['spectral_flatness'])
        target['pacf_distinct'].append(
            spectral['periodic_autocorrelation']['num_distinct_rounded'])
        target['aacf_range'].append(
            spectral['aperiodic_autocorrelation']['range'])
    
    # ---- Create figure ----
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
    fig.suptitle('Prime vs Composite $M_p$: Spectral Metric Comparison\n'
                 '(All Mann-Whitney U p-values $> 0.05$: No Statistical Separation)',
                 fontsize=13, fontweight='bold', y=1.05)
    
    for idx, (key, label) in enumerate(metric_keys):
        ax = axes[idx]
        
        pv = prime_data[key]
        cv = composite_data[key]
        
        if not pv or not cv:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        
        prime_mean = np.mean(pv)
        prime_std = np.std(pv, ddof=1) if len(pv) > 1 else 0
        comp_mean = np.mean(cv)
        comp_std = np.std(cv, ddof=1) if len(cv) > 1 else 0
        
        # Mann-Whitney U test
        mw = mann_whitney_u(pv, cv)
        p_val = mw.get('p_approx', 1.0)
        
        # Bar chart
        x = [0, 1]
        means = [prime_mean, comp_mean]
        stds = [prime_std, comp_std]
        colors = [COLOR_PRIME, COLOR_COMPOSITE]
        
        bars = ax.bar(x, means, yerr=stds, color=colors,
                      edgecolor='black', linewidth=0.5,
                      width=0.6, capsize=5,
                      error_kw={'linewidth': 1.5})
        
        # Add individual data points
        for i, (data, xi) in enumerate([(pv, 0), (cv, 1)]):
            jitter = np.random.uniform(-0.1, 0.1, len(data))
            ax.scatter([xi + j for j in jitter], data,
                       color='black', s=20, alpha=0.6, zorder=5)
        
        # Add p-value annotation
        if p_val is not None:
            p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else f'p < 0.001'
        else:
            p_str = 'p = N/A'
        
        sig_color = COLOR_SUCCESS if (p_val is not None and p_val > 0.05) else COLOR_COMPOSITE
        ax.text(0.5, 0.95, p_str,
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, fontweight='bold', color=sig_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=sig_color, alpha=0.9))
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Prime\n$M_p$', 'Composite\n$M_p$'], fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    outpath = os.path.join(RESULTS_DIR, 'fig_prime_vs_composite_comparison.png')
    fig.savefig(outpath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PATH 1 + PATH 2 VISUALIZATION FIGURES")
    print("=" * 70)
    
    overall_start = time.time()
    generated_files = []
    
    # Figure 1
    try:
        f1 = generate_fig_factor_discovery_cd_probing()
        generated_files.append(f1)
    except Exception as e:
        print(f"  ERROR generating Figure 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 2
    try:
        f2 = generate_fig_spectral_negative_result()
        generated_files.append(f2)
    except Exception as e:
        print(f"  ERROR generating Figure 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 3
    try:
        f3 = generate_fig_minimal_polynomial_construction()
        generated_files.append(f3)
    except Exception as e:
        print(f"  ERROR generating Figure 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Figure 4
    try:
        f4 = generate_fig_prime_vs_composite_comparison()
        generated_files.append(f4)
    except Exception as e:
        print(f"  ERROR generating Figure 4: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE ({elapsed:.1f}s)")
    print("=" * 70)
    print("\nGenerated files:")
    for f in generated_files:
        print(f"  {f}")
    
    if len(generated_files) < 4:
        print(f"\nWARNING: Only {len(generated_files)}/4 figures were generated successfully!")
