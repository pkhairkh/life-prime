#!/usr/bin/env python3
"""
Visualization: Path 2 v2 Results
=================================
Publication-quality plots showing:
1. Brent Cycle Detection: Factor recovery across Mersenne composites
2. Spectral Metrics Comparison: Prime vs Composite
3. Bispectral Entropy Landscape
4. Hamming Weight Trajectory Comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Serif SC']
plt.rcParams['axes.unicode_minus'] = False

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, gf2_mat_vec, is_prime_simple
)

PRIMITIVE_POLYS = {
    2: [1, 1], 3: [1, 1, 0], 5: [1, 0, 1, 0, 0],
    7: [1, 1, 0, 0, 0, 0, 0],
    11: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    13: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    19: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    23: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    29: [1, 0, 1] + [0] * 26,
}

KNOWN_FACTORS = {11: [23, 89], 23: [47, 178481], 29: [233, 1103, 2089]}
MERSENNE_PRIME_EXP = {2, 3, 5, 7, 13, 17, 19, 31}


def generate_all_plots():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # ============================================================
    # Figure 1: Brent Cycle Detection — Factor Recovery Dashboard
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Brent Cycle Detection: Factor Recovery from C^d Orbits', 
                 fontsize=14, fontweight='bold')
    
    # M_11 = 2047 = 23 × 89
    ax = axes[0]
    p = 11; M_p = 2047
    poly = PRIMITIVE_POLYS[p]
    C = companion_matrix(poly)
    v = np.zeros(p, dtype=np.int64); v[0] = 1
    identity = np.eye(p, dtype=np.int64)
    
    # Compute orbit lengths for d = 2..100
    d_range = range(2, 101)
    orbit_lengths = []
    is_factor_d = []
    C_d = np.eye(p, dtype=np.int64)
    for d in d_range:
        C_d = gf2_mat_mul(C_d, C)
        # Compute exact order via matrix powering
        if M_p % d == 0:
            expected = M_p // d
            test = gf2_mat_pow(C_d, expected)
            if np.array_equal(test % 2, identity):
                orbit_lengths.append(expected)
            else:
                orbit_lengths.append(M_p)  # Full orbit
        else:
            orbit_lengths.append(M_p)  # Full orbit (gcd=1)
        is_factor_d.append(M_p % d == 0)
    
    colors = ['#e74c3c' if f else '#3498db' for f in is_factor_d]
    ax.bar(list(d_range), orbit_lengths, color=colors, alpha=0.7, width=0.8)
    ax.set_xlabel('d (exponent for C^d)')
    ax.set_ylabel('Orbit length of C^d')
    ax.set_title(f'$M_{{11}}$ = 2047 = 23 × 89', fontsize=12)
    ax.axhline(y=M_p, color='gray', linestyle='--', alpha=0.5, label=f'M_p = {M_p}')
    ax.legend(loc='best', fontsize=8)
    
    # Add annotation for factor d's
    factor_ds = [d for d in d_range if M_p % d == 0]
    for fd in factor_ds[:5]:
        ax.annotate(f'd={fd}\nλ={M_p//fd}', xy=(fd, M_p//fd),
                   fontsize=7, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
    
    # M_23
    ax = axes[1]
    p = 23; M_p = (1 << p) - 1
    # For M_23, we can't compute all orbit lengths, so show a schematic
    d_vals = list(range(2, 51))
    orbit_lens = []
    factor_flags = []
    for d in d_vals:
        if d == 47 or (M_p % d == 0 and d < 100):
            orbit_lens.append(M_p // d)
            factor_flags.append(True)
        else:
            orbit_lens.append(M_p)  # Full orbit
            factor_flags.append(False)
    
    colors = ['#e74c3c' if f else '#3498db' for f in factor_flags]
    ax.bar(d_vals, [ol / 1e6 for ol in orbit_lens], color=colors, alpha=0.7, width=0.8)
    ax.set_xlabel('d (exponent for C^d)')
    ax.set_ylabel('Orbit length of C^d (×10⁶)')
    ax.set_title(f'$M_{{23}}$ = 8,388,607 = 47 × 178,481', fontsize=11)
    ax.axhline(y=M_p/1e6, color='gray', linestyle='--', alpha=0.5)
    # Mark d=47
    if 47 in d_vals:
        ax.annotate(f'd=47\nλ={M_p//47:,}', xy=(47, M_p//47/1e6),
                   fontsize=7, ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
    
    # M_29 schematic
    ax = axes[2]
    p = 29; M_p = (1 << p) - 1
    d_vals = list(range(2, 51))
    orbit_lens = []
    factor_flags = []
    for d in d_vals:
        if d in [233, 1103, 2089] or (M_p % d == 0 and d < 100):
            orbit_lens.append(M_p // d)
            factor_flags.append(True)
        else:
            orbit_lens.append(M_p)
            factor_flags.append(False)
    
    colors = ['#e74c3c' if f else '#3498db' for f in factor_flags]
    ax.bar(d_vals, [ol / 1e9 for ol in orbit_lens], color=colors, alpha=0.7, width=0.8)
    ax.set_xlabel('d (exponent for C^d)')
    ax.set_ylabel('Orbit length of C^d (×10⁹)')
    ax.set_title(f'$M_{{29}}$ = 536,870,911\n= 233 × 1103 × 2089', fontsize=10)
    ax.axhline(y=M_p/1e9, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig_path2_brent_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_path2_brent_detection.png")
    
    # ============================================================
    # Figure 2: Spectral Metrics Comparison
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Path 2 v2: Spectral Metrics — Prime vs Composite $M_p$', 
                 fontsize=14, fontweight='bold')
    
    # Collect data from the experiment results
    prime_data = {
        7: {'fft_flat': 0.2103, 'wht_kurt': 6.61, 'bisp_ent': 0.9936, 
            'bisp_kurt': 11.33, 'hw_peaks': 7},
        13: {'fft_flat': 0.1986, 'wht_kurt': 7.31, 'bisp_ent': 0.9618,
             'bisp_kurt': 1.07, 'hw_peaks': 15},
        17: {'fft_flat': 0.0356, 'wht_kurt': 32.23, 'bisp_ent': 0.9029,
             'bisp_kurt': 16.29, 'hw_peaks': 41},
        19: {'fft_flat': 0.0644, 'wht_kurt': 17.10, 'bisp_ent': 0.9078,
             'bisp_kurt': 60.72, 'hw_peaks': 26},
    }
    comp_data = {
        11: {'fft_flat': 0.1356, 'wht_kurt': 13.44, 'bisp_ent': 0.9918,
             'bisp_kurt': 11.32, 'hw_peaks': 7},
        23: {'fft_flat': 0.0161, 'wht_kurt': 67.17, 'bisp_ent': 0.8114,
             'bisp_kurt': 619.95, 'hw_peaks': 78},
        29: {'fft_flat': 0.0063, 'wht_kurt': 143.36, 'bisp_ent': 0.4474,
             'bisp_kurt': 934.12, 'hw_peaks': 12},
    }
    
    metrics = [
        ('fft_flat', 'FFT Spectral Flatness\n(Hamming Weight)', False),
        ('wht_kurt', 'WHT Kurtosis\n(Hamming Weight)', True),
        ('bisp_ent', 'Normalized Bispectral\nEntropy', False),
        ('bisp_kurt', 'Bispectral Kurtosis', True),
        ('hw_peaks', 'Autocorrelation Peaks\n(Hamming Weight)', False),
    ]
    
    for idx, (key, title, log_scale) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        
        p_primes = list(prime_data.keys())
        p_comps = list(comp_data.keys())
        
        prime_vals = [prime_data[p][key] for p in p_primes]
        comp_vals = [comp_data[p][key] for p in p_comps]
        
        x_prime = np.arange(len(p_primes))
        x_comp = np.arange(len(p_comps))
        
        ax.bar(x_prime - 0.2, prime_vals, 0.4, color='#2ecc71', alpha=0.8, label='Prime $M_p$')
        ax.bar(x_comp + 0.2, comp_vals, 0.4, color='#e74c3c', alpha=0.8, label='Composite $M_p$')
        
        ax.set_xticks(list(range(max(len(p_primes), len(p_comps)))))
        all_ps = sorted(set(p_primes + p_comps))
        ax.set_xticklabels([f'p={p}' for p in all_ps[:max(len(p_primes), len(p_comps))]], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(loc='best', fontsize=7)
        if log_scale:
            ax.set_yscale('log')
    
    # 6th subplot: Summary radar chart
    ax = axes[1][2]
    ax.axis('off')
    ax.text(0.5, 0.9, 'KEY FINDINGS', fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    findings = [
        '• Brent cycle detection recovers',
        '  factors from C^d orbits',
        '• M₁₁: 100% factor recovery',
        '• M₂₃: factor 47 detected (440K steps)',
        '• M₂₉: factors verified algebraically',
        '',
        '• Spectral metrics show trends',
        '  but confounded by exponent p',
        '• Bispectral entropy: prime > comp',
        '• WHT kurtosis: comp >> prime',
        '',
        '• No non-circular spectral',
        '  factor detection found',
    ]
    ax.text(0.1, 0.75, '\n'.join(findings), fontsize=9,
            ha='left', va='top', transform=ax.transAxes,
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig_path2_spectral_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_path2_spectral_comparison.png")
    
    # ============================================================
    # Figure 3: Hamming Weight Trajectory Comparison
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Hamming Weight Trajectory: $wt(C^k \\cdot e_1)$ for Prime vs Composite $M_p$',
                 fontsize=14, fontweight='bold')
    
    cases = [(7, True), (13, True), (19, True), (11, False), (23, False), (29, False)]
    
    for idx, (p, is_prime) in enumerate(cases):
        ax = axes[idx // 3][idx % 3]
        M_p = (1 << p) - 1
        poly = PRIMITIVE_POLYS[p]
        C = companion_matrix(poly)
        v = np.zeros(p, dtype=np.int64)
        v[0] = 1
        state = v.copy()
        
        seq_len = min(M_p, 2048)
        weights = np.zeros(seq_len, dtype=np.float64)
        for k in range(seq_len):
            weights[k] = np.sum(state % 2)
            state = gf2_mat_vec(C, state)
        
        color = '#2ecc71' if is_prime else '#e74c3c'
        label = f'p={p} (PRIME)' if is_prime else f'p={p} (COMP)'
        
        ax.plot(range(seq_len), weights, color=color, alpha=0.6, linewidth=0.5)
        ax.axhline(y=np.mean(weights), color='black', linestyle='--', alpha=0.3)
        ax.set_title(label, fontsize=11, 
                     color='#2ecc71' if is_prime else '#e74c3c')
        ax.set_xlabel('k (time step)')
        ax.set_ylabel('Hamming weight')
        ax.set_ylim(0, p + 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig_path2_hamming_weight.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_path2_hamming_weight.png")
    
    print("\nAll visualizations generated!")


if __name__ == "__main__":
    generate_all_plots()
