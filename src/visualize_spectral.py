"""
Publication-Quality Visualizations for Spectral Analysis
=========================================================

Generates figures comparing spectral properties of trace sequences
for prime vs composite Mersenne numbers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from trace_spectral_analysis import (
    compute_trace_sequence, walsh_hadamard_transform,
    periodic_autocorrelation, aperiodic_autocorrelation,
    spectral_factor_detection, factor_from_spectrum,
    PRIMITIVE_POLYS, KNOWN_FACTORS, MERSENNE_PRIME_EXPONENTS
)
from matrix_power_ca import is_prime_simple


# ============================================================
# Style Configuration
# ============================================================

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

# Color scheme
COLORS = {
    'prime': '#2196F3',       # Blue
    'composite': '#F44336',   # Red
    'prime_light': '#90CAF9',
    'composite_light': '#EF9A9A',
    'accent1': '#4CAF50',     # Green
    'accent2': '#FF9800',     # Orange
    'accent3': '#9C27B0',     # Purple
    'neutral': '#607D8B',     # Blue-grey
}


def get_results_dir():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ============================================================
# Figure 1: Trace Sequences — Prime vs Composite
# ============================================================

def fig_trace_prime_vs_composite():
    """
    Side-by-side trace sequences for:
      - p=7 (prime M_7=127)
      - p=11 (composite M_11=2047)
    """
    print("  Generating fig_trace_prime_vs_composite.png...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Trace Sequences: Prime vs Composite Mersenne Numbers',
                 fontsize=14, fontweight='bold', y=0.98)

    cases = [
        (7, 'Prime', COLORS['prime']),
        (11, 'Composite', COLORS['composite']),
    ]

    for row, (p, label, color) in enumerate(cases):
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]
        trace_seq = compute_trace_sequence(poly, M_p)
        seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

        # Full sequence
        ax = axes[row, 0]
        ax.imshow(trace_seq.reshape(1, -1), aspect='auto', cmap='binary',
                  interpolation='nearest')
        ax.set_title(f'M_{p} = {M_p} ({label})\nFull Trace Sequence Tr(C^k)',
                     fontweight='bold')
        ax.set_xlabel('k')
        ax.set_yticks([])

        # First 200 samples as ±1 time series
        ax = axes[row, 1]
        n_show = min(200, len(seq_pm1))
        ax.plot(range(n_show), seq_pm1[:n_show], '-', color=color, alpha=0.7, linewidth=0.8)
        ax.fill_between(range(n_show), seq_pm1[:n_show], alpha=0.15, color=color)
        ax.set_title(f'First {n_show} samples (±1 mapping)', fontweight='bold')
        ax.set_xlabel('k')
        ax.set_ylabel('Tr(C^k) → ±1')
        ax.set_ylim(-1.5, 1.5)

        # Add statistics
        n_ones = int(np.sum(trace_seq))
        n_zeros = len(trace_seq) - n_ones
        ax.text(0.02, 0.95, f'Ones: {n_ones}, Zeros: {n_zeros}\n'
                              f'Ratio: {n_ones/len(trace_seq):.4f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_trace_prime_vs_composite.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 2: WHT Spectrum — Prime vs Composite
# ============================================================

def fig_wht_spectrum():
    """
    WHT spectra for prime vs composite Mersenne numbers.
    Shows that m-sequences have flat WHT spectra regardless of primality.
    """
    print("  Generating fig_wht_spectrum.png...")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Walsh-Hadamard Transform Spectra: Prime vs Composite Mersenne',
                 fontsize=14, fontweight='bold', y=0.98)

    cases = [
        (7, 'Prime', COLORS['prime']),
        (11, 'Composite', COLORS['composite']),
        (13, 'Prime', COLORS['prime']),
        (17, 'Prime', COLORS['prime']),
        (19, 'Prime', COLORS['prime']),
        (23, 'Composite', COLORS['composite']),
    ]

    for idx, (p, label, color) in enumerate(cases):
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]

        # Compute trace sequence
        if p <= 17:
            seq_len = M_p
        else:
            seq_len = 2**16
        trace_seq = compute_trace_sequence(poly, seq_len)
        seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

        # Compute WHT (use power-of-2 length)
        wht_len = 1
        while wht_len * 2 <= len(seq_pm1):
            wht_len *= 2
        wht_spectrum = walsh_hadamard_transform(seq_pm1[:wht_len])

        # Plot
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        wht_abs = np.abs(wht_spectrum)
        wht_no_dc = wht_abs[1:]  # Exclude DC

        # Show only first quarter (spectrum is symmetric-ish for real input)
        show_len = min(len(wht_no_dc), 512)
        ax.plot(range(show_len), wht_no_dc[:show_len], '-', color=color,
                alpha=0.6, linewidth=0.5)

        # Highlight mean and ±2σ
        mean_val = np.mean(wht_no_dc)
        std_val = np.std(wht_no_dc)
        ax.axhline(mean_val, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhline(mean_val + 2*std_val, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(mean_val - 2*std_val, color='red', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_title(f'p={p}, M_p={M_p} ({label})\n'
                     f'WHT: mean={mean_val:.1f}, std={std_val:.1f}, kurt={_kurtosis(wht_no_dc):.2f}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Sequency position')
        ax.set_ylabel('|WHT|')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_wht_spectrum.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 3: Autocorrelation — Prime vs Composite
# ============================================================

def fig_autocorrelation():
    """
    Autocorrelation functions for prime vs composite Mersenne numbers.
    Shows that periodic autocorrelation is always two-valued for m-sequences.
    """
    print("  Generating fig_autocorrelation.png...")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('Autocorrelation Analysis: Prime vs Composite Mersenne Numbers',
                 fontsize=14, fontweight='bold', y=0.98)

    cases = [
        (7, 'Prime', COLORS['prime']),
        (11, 'Composite', COLORS['composite']),
        (13, 'Prime', COLORS['prime']),
    ]

    for row, (p, label, color) in enumerate(cases):
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]
        trace_seq = compute_trace_sequence(poly, M_p)
        seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

        # Periodic autocorrelation
        ax = axes[row, 0]
        max_lag = min(M_p, 100)
        pacf = periodic_autocorrelation(seq_pm1, max_lag=max_lag)
        lags = np.arange(len(pacf))
        ax.bar(lags[1:], pacf[1:], width=1.0, color=color, alpha=0.7)
        expected = -1.0 / M_p
        ax.axhline(expected, color='black', linestyle='--', linewidth=1,
                    label=f'Expected: -1/{M_p} = {expected:.4f}')
        ax.set_title(f'p={p} ({label}): Periodic AC\n(2-valued: {len(set(np.round(pacf[1:],4)))<=2})',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Lag τ')
        ax.set_ylabel('R(τ)')
        ax.legend(fontsize=7)

        # Aperiodic autocorrelation
        ax = axes[row, 1]
        max_lag_a = min(M_p - 1, 200)
        aacf = aperiodic_autocorrelation(seq_pm1, max_lag=max_lag_a)
        ax.plot(range(len(aacf)), aacf, '-', color=color, alpha=0.7, linewidth=0.8)
        ax.set_title(f'p={p} ({label}): Aperiodic AC\nrange={np.max(aacf[1:])-np.min(aacf[1:]):.4f}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Lag τ')
        ax.set_ylabel('R(τ)')

        # Histogram of aperiodic AC values
        ax = axes[row, 2]
        aacf_vals = aacf[1:]
        ax.hist(aacf_vals, bins=50, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(aacf_vals), color='black', linestyle='--', linewidth=1)
        ax.set_title(f'p={p} ({label}): AC Value Distribution\n'
                     f'std={np.std(aacf_vals):.4f}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('R(τ)')
        ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_autocorrelation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 4: Spectral Factor Detection — Decimation Analysis
# ============================================================

def fig_spectral_factor_detection():
    """
    Spectral anomalies and their relationship to factors.
    Shows decimation (sub-sampling) statistics for factor vs non-factor step sizes.
    """
    print("  Generating fig_spectral_factor_detection.png...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Spectral Factor Detection: Decimation Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    composite_cases = [(11, [23, 89]), (23, [47, 178481]), (29, [233, 1103, 2089])]

    for idx, (p, known_factors) in enumerate(composite_cases):
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]

        # Determine sequence length
        if p <= 11:
            seq_len = M_p
        elif p <= 23:
            seq_len = 2**16
        else:
            seq_len = 2**14

        trace_seq = compute_trace_sequence(poly, seq_len)

        # ---- Subplot 1: Decimation balance z-scores ----
        ax = fig.add_subplot(gs[idx, 0])

        ds = list(range(2, min(200, M_p, seq_len)))
        z_scores = []
        is_factors = []
        ratios = []

        for d in ds:
            sub_indices = list(range(0, len(trace_seq), d))
            if len(sub_indices) < 5:
                z_scores.append(0)
                is_factors.append(False)
                ratios.append(0.5)
                continue
            sub_seq = trace_seq[sub_indices]
            n_ones = int(np.sum(sub_seq))
            ratio = n_ones / len(sub_seq)
            expected_std = np.sqrt(0.25 / len(sub_seq))
            z_score = abs(ratio - 0.5) / expected_std if expected_std > 0 else 0

            z_scores.append(z_score)
            is_factors.append(M_p % d == 0)
            ratios.append(ratio)

        # Plot z-scores
        for i, d in enumerate(ds):
            if is_factors[i]:
                ax.bar(d, z_scores[i], width=1.0, color=COLORS['composite'],
                       alpha=0.9, edgecolor='black', linewidth=0.5)
            else:
                ax.bar(d, z_scores[i], width=1.0, color=COLORS['neutral'],
                       alpha=0.3)

        ax.axhline(3.0, color='red', linestyle='--', linewidth=1, label='3σ threshold')
        ax.set_title(f'p={p}, M_p={M_p}\nDecimation z-scores (red=factor d)',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Step size d')
        ax.set_ylabel('|z-score| for balance test')
        ax.legend(fontsize=7)

        # ---- Subplot 2: Ones ratio for each d ----
        ax = fig.add_subplot(gs[idx, 1])

        for i, d in enumerate(ds):
            if is_factors[i]:
                ax.scatter(d, ratios[i], color=COLORS['composite'], s=40,
                           zorder=5, edgecolor='black', linewidth=0.5)
            else:
                ax.scatter(d, ratios[i], color=COLORS['neutral'], s=5, alpha=0.3)

        ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, label='Expected (0.5)')

        # Annotate known factors
        for f in known_factors:
            if f < 200:
                ax.annotate(f'q={f}', xy=(f, 0.5), fontsize=8,
                           fontweight='bold', color=COLORS['composite'])

        ax.set_title(f'p={p}: Ones ratio vs step size d\n(red dots = factor d)',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Step size d')
        ax.set_ylabel('Ones ratio in decimated sequence')
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_spectral_factor_detection.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 5: Factor Recovery Rates
# ============================================================

def fig_factor_recovery():
    """
    Recovery rates for different spectral methods.
    Compares WHT, FFT, autocorrelation, decimation, and GCD methods.
    """
    print("  Generating fig_factor_recovery.png...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Factor Recovery from Spectral Analysis',
                 fontsize=14, fontweight='bold', y=0.98)

    composite_cases = [(11, [23, 89]), (23, [47, 178481]), (29, [233, 1103, 2089])]

    for idx, (p, known_factors) in enumerate(composite_cases):
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]

        if p <= 11:
            seq_len = M_p
        elif p <= 23:
            seq_len = 2**16
        else:
            seq_len = 2**14

        trace_seq = compute_trace_sequence(poly, seq_len)
        factor_results = factor_from_spectrum(trace_seq, M_p, p)

        # Collect method results
        methods = ['wht', 'fft', 'autocorrelation', 'decimation_balance', 'folding',
                   'cycle_completion', 'linear_complexity']
        method_labels = ['WHT\nPeaks', 'FFT\nPeaks', 'AC\nAnomalies', 'Decimation\nBalance', 'Folding\nCorr.',
                         'Cycle\nCompletion', 'Linear\nComplex.']
        method_colors = [COLORS['prime'], COLORS['accent1'], COLORS['accent2'],
                         COLORS['composite'], COLORS['accent3'], '#795548', '#00BCD4']

        found_per_method = []
        for method in methods:
            mr = factor_results['method_results'].get(method, {})
            recovered = mr.get('factors_recovered', [])
            # Count how many known factors each method found
            found = set()
            for item in recovered:
                # Different methods return different tuple formats
                factor_val = item[-1] if isinstance(item[-1], int) and item[-1] > 1 else None
                if factor_val and 1 < factor_val < M_p:
                    found.add(factor_val)
                    # Also add prime factors of composite factor
                    for f in _trial_factor(factor_val):
                        if M_p % f == 0:
                            found.add(f)
            found_per_method.append(len(found & set(known_factors)))

        ax = axes[idx]
        max_count = len(known_factors)
        bars = ax.bar(range(len(methods)), found_per_method, color=method_colors,
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, fontsize=8)
        ax.set_ylabel(f'Known factors found (of {max_count})')
        ax.set_title(f'p={p}, M_p={M_p}\nFactors: {"×".join(str(f) for f in known_factors)}',
                     fontweight='bold', fontsize=10)
        ax.set_ylim(0, max_count + 0.5)

        # Add value labels
        for bar, val in zip(bars, found_per_method):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{val}/{max_count}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_factor_recovery.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 6: Comprehensive Spectral Comparison Dashboard
# ============================================================

def fig_spectral_dashboard():
    """
    Dashboard comparing all spectral metrics across prime and composite cases.
    """
    print("  Generating fig_spectral_dashboard.png...")

    all_cases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    # Collect data
    data = {}
    for p in all_cases:
        M_p = 2**p - 1
        poly = PRIMITIVE_POLYS[p]
        is_prime = is_prime_simple(M_p)

        if p <= 11:
            seq_len = M_p
        elif p <= 17:
            seq_len = 2**16
        elif p <= 23:
            seq_len = 2**16
        else:
            seq_len = 2**14

        trace_seq = compute_trace_sequence(poly, seq_len)
        seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

        # WHT
        wht_len = 1
        while wht_len * 2 <= len(seq_pm1):
            wht_len *= 2
        wht_spectrum = walsh_hadamard_transform(seq_pm1[:wht_len])
        wht_no_dc = np.abs(wht_spectrum[1:])

        # Autocorrelation
        max_lag = min(len(seq_pm1) - 1, 500)
        pacf = periodic_autocorrelation(seq_pm1, max_lag=max_lag)
        aacf = aperiodic_autocorrelation(seq_pm1, max_lag=max_lag)

        # FFT
        fft_result = np.fft.fft(seq_pm1[:wht_len])
        power = np.abs(fft_result) ** 2
        power_no_dc = power[1:]

        data[p] = {
            'is_prime': is_prime,
            'M_p': M_p,
            'wht_kurtosis': _kurtosis(wht_no_dc),
            'wht_std': float(np.std(wht_no_dc)),
            'pacf_max_dev': float(np.max(np.abs(pacf[1:] - (-1.0/M_p)))),
            'aacf_range': float(np.max(aacf[1:]) - np.min(aacf[1:])),
            'aacf_std': float(np.std(aacf[1:])),
            'fft_peaks_3sigma': int(np.sum(power_no_dc > np.mean(power_no_dc) + 3*np.std(power_no_dc))),
        }

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Spectral Analysis Dashboard: Prime vs Composite Mersenne Numbers',
                 fontsize=14, fontweight='bold', y=0.98)

    ps = sorted(data.keys())
    is_primes = [data[p]['is_prime'] for p in ps]
    colors = [COLORS['prime'] if ip else COLORS['composite'] for ip in is_primes]
    markers = ['o' if ip else 's' for ip in is_primes]

    metrics = [
        ('wht_kurtosis', 'WHT Kurtosis (excess)', 'Kurtosis of |WHT| spectrum'),
        ('aacf_range', 'Aperiodic AC Range', 'Range of aperiodic autocorrelation'),
        ('aacf_std', 'Aperiodic AC Std', 'Std dev of aperiodic autocorrelation'),
        ('pacf_max_dev', 'Periodic AC Max Deviation', 'Max |R(τ) - (-1/M_p)|'),
        ('fft_peaks_3sigma', 'FFT Peaks (3σ)', 'Number of spectral peaks above 3σ'),
        ('wht_std', 'WHT Std Dev', 'Std dev of |WHT| spectrum'),
    ]

    for idx, (key, title, ylabel) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        values = [data[p][key] for p in ps]

        for i, (p, v) in enumerate(zip(ps, values)):
            ax.scatter(p, v, c=colors[i], marker=markers[i], s=80,
                       edgecolor='black', linewidth=0.5, zorder=5)

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Exponent p')
        ax.set_ylabel(ylabel)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['prime'],
                   markersize=10, label='Prime M_p'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['composite'],
                   markersize=10, label='Composite M_p'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_spectral_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Figure 7: Decimation Period Analysis
# ============================================================

def fig_decimation_period_analysis():
    """
    For composite M_p, show how decimated sequences at factor step sizes
    have shorter periods than the original m-sequence.
    """
    print("  Generating fig_decimation_period_analysis.png...")

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Decimation Analysis: Sub-sampled Trace Sequences',
                 fontsize=14, fontweight='bold', y=0.98)

    # Focus on M_11 = 2047 = 23 × 89
    p = 11
    M_p = 2047
    poly = PRIMITIVE_POLYS[p]
    trace_seq = compute_trace_sequence(poly, M_p)
    seq_pm1 = 2.0 * trace_seq.astype(np.float64) - 1.0

    # Test different step sizes
    step_sizes = [1, 2, 5, 10, 23, 50, 89, 100, 200]

    for idx, d in enumerate(step_sizes):
        ax = axes[idx // 3, idx % 3]

        # Decimated sequence
        sub_indices = list(range(0, len(trace_seq), d))
        sub_seq = trace_seq[sub_indices]
        sub_pm1 = 2.0 * sub_seq.astype(np.float64) - 1.0

        is_factor = (M_p % d == 0)

        # Compute autocorrelation of decimated sequence
        if len(sub_pm1) > 10:
            sub_acorr = periodic_autocorrelation(sub_pm1, max_lag=min(len(sub_pm1)-1, 80))
        else:
            sub_acorr = np.zeros(2)

        # Plot decimated sequence
        n_show = min(150, len(sub_pm1))
        ax.plot(range(n_show), sub_pm1[:n_show], '-',
                color=COLORS['composite'] if is_factor else COLORS['neutral'],
                alpha=0.7, linewidth=0.8)

        # Title with info
        color_tag = 'FACTOR' if is_factor else 'non-factor'
        period_info = f'Period: M_p/{d if is_factor else "—"}'
        ax.set_title(f'd={d} ({color_tag})\n'
                     f'Sub-seq length: {len(sub_seq)}, '
                     f'balance: {np.sum(sub_seq)/len(sub_seq):.3f}',
                     fontweight='bold', fontsize=9,
                     color=COLORS['composite'] if is_factor else COLORS['neutral'])
        ax.set_xlabel('k')
        ax.set_ylabel('s(k·d)')
        ax.set_ylim(-1.5, 1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    results_dir = get_results_dir()
    fig.savefig(os.path.join(results_dir, 'fig_decimation_period_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Helper Functions
# ============================================================

def _kurtosis(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3)


def _trial_factor(n: int) -> list:
    """Trial division factorization."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


# ============================================================
# Main
# ============================================================

def generate_all_figures():
    """Generate all publication-quality figures."""
    print("="*60)
    print("GENERATING SPECTRAL ANALYSIS FIGURES")
    print("="*60)

    fig_trace_prime_vs_composite()
    fig_wht_spectrum()
    fig_autocorrelation()
    fig_spectral_factor_detection()
    fig_factor_recovery()
    fig_spectral_dashboard()
    fig_decimation_period_analysis()

    results_dir = get_results_dir()
    print(f"\nAll figures saved to {results_dir}/")
    print("  - fig_trace_prime_vs_composite.png")
    print("  - fig_wht_spectrum.png")
    print("  - fig_autocorrelation.png")
    print("  - fig_spectral_factor_detection.png")
    print("  - fig_factor_recovery.png")
    print("  - fig_spectral_dashboard.png")
    print("  - fig_decimation_period_analysis.png")


if __name__ == "__main__":
    generate_all_figures()
