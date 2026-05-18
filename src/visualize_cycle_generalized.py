"""
Visualization for Cycle Factor Extraction and Generalized CA Results
=====================================================================

Generates publication-quality figures for:
1. Cycle spectrum comparison (prime vs composite Mersenne)
2. Factor recovery rates across Mersenne exponents
3. Companion matrix C^d factor extraction demonstration
4. Fermat/Proth CA fold comparison
5. CA rule hierarchy diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from math import gcd
from collections import Counter

# Font setup
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = '/home/z/my-project/life-prime/results'


def fig1_prime_vs_composite_spectrum():
    """
    Figure 1: Cycle spectrum comparison for Mersenne prime vs composite.
    Shows the fundamental structural difference in squaring map cycles.
    """
    from src.cycle_factor_extraction import squaring_map_cycle_decomposition
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # M_7 = 127 (Mersenne PRIME)
    info_7 = squaring_map_cycle_decomposition(127)
    spectrum_7 = info_7['cycle_spectrum']
    
    ax = axes[0]
    lengths_7 = sorted(spectrum_7.keys())
    counts_7 = [spectrum_7[l] for l in lengths_7]
    colors_7 = ['#2ecc71' if l > 1 else '#e74c3c' for l in lengths_7]
    bars = ax.bar(range(len(lengths_7)), counts_7, color=colors_7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(lengths_7)))
    ax.set_xticklabels([str(l) for l in lengths_7])
    ax.set_xlabel('Cycle Length', fontsize=12)
    ax.set_ylabel('Number of Cycles', fontsize=12)
    ax.set_title(r'$M_7 = 127$ (Mersenne Prime)', fontsize=13, fontweight='bold')
    
    # Add annotation
    ax.annotate('Multiple cycle lengths\n(field structure)', 
                xy=(2, 2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # M_11 = 2047 (COMPOSITE)
    info_11 = squaring_map_cycle_decomposition(2047)
    spectrum_11 = info_11['cycle_spectrum']
    
    ax = axes[1]
    lengths_11 = sorted(spectrum_11.keys())
    counts_11 = [spectrum_11[l] for l in lengths_11]
    colors_11 = ['#e74c3c' if l > 1 else '#3498db' for l in lengths_11]
    bars = ax.bar(range(len(lengths_11)), counts_11, color=colors_11, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(lengths_11)))
    ax.set_xticklabels([str(l) for l in lengths_11])
    ax.set_xlabel('Cycle Length', fontsize=12)
    ax.set_ylabel('Number of Cycles', fontsize=12)
    ax.set_title(r'$M_{11} = 2047 = 23 \times 89$ (Composite)', fontsize=13, fontweight='bold')
    
    # Annotate factor connection
    ax.annotate('Cycle length 10\nrelated to factors\n23-1=22, 89-1=88\nord(2,11)=10', 
                xy=(1, 14), fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Squaring Map Cycle Spectrum: Prime vs Composite Mersenne', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_cycle_spectrum_prime_vs_composite.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_cycle_spectrum_prime_vs_composite.png")


def fig2_factor_recovery_rates():
    """
    Figure 2: Factor recovery rates across different Mersenne exponents.
    Shows 100% recovery for all tested composites using C^d method.
    """
    from src.cycle_factor_extraction import CycleStructureFactorer
    
    exponents = [11, 23, 29, 37, 41, 43]
    recovery_rates = []
    known_factor_counts = []
    found_factor_counts = []
    
    for p in exponents:
        factorer = CycleStructureFactorer(p)
        result = factorer.analyze_companion_orbits(num_polys=15)
        recovery_rates.append(result['recovery_rate'] * 100)
        known_factor_counts.append(len(result['known_factors']))
        found_factor_counts.append(len(set(result['prime_factors']) & set(result['known_factors'])))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Recovery rates
    colors = ['#2ecc71' if r == 100 else '#e74c3c' for r in recovery_rates]
    bars = ax1.bar(range(len(exponents)), recovery_rates, color=colors, 
                   edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(exponents)))
    ax1.set_xticklabels([f'$M_{{{p}}}$' for p in exponents])
    ax1.set_ylabel('Factor Recovery Rate (%)', fontsize=12)
    ax1.set_title('CA Cycle-Based Factor Recovery', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    for i, (rate, p) in enumerate(zip(recovery_rates, exponents)):
        ax1.text(i, rate + 2, f'{rate:.0f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Right: Known vs found factors
    x = np.arange(len(exponents))
    width = 0.35
    ax2.bar(x - width/2, known_factor_counts, width, label='Known factors', 
            color='#3498db', edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, found_factor_counts, width, label='Found by CA', 
            color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$M_{{{p}}}$' for p in exponents])
    ax2.set_ylabel('Number of Prime Factors', fontsize=12)
    ax2.set_title('Known vs CA-Recovered Factors', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    
    fig.suptitle('Factor Extraction from CA Cycle Structure (C^d Method)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_factor_recovery_rates.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_factor_recovery_rates.png")


def fig3_cd_factor_extraction():
    """
    Figure 3: Visual explanation of the C^d factor extraction method.
    Shows how computing C^d for various d reveals factors of M_p.
    """
    from src.cycle_factor_extraction import CycleStructureFactorer
    from src.matrix_power_ca import companion_matrix, gf2_mat_pow
    
    p = 11
    M_p = 2047  # = 23 × 89
    prim_coeffs = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    C = companion_matrix(prim_coeffs)
    identity = np.eye(p, dtype=np.int64)
    
    # Compute order of C^d for d = 1 to 100
    d_values = list(range(1, 101))
    orders = []
    
    for d in d_values:
        C_d = gf2_mat_pow(C, d)
        g = gcd(M_p, d)
        if g > 1:
            expected_order = M_p // g
            C_d_power = gf2_mat_pow(C_d, expected_order)
            if np.array_equal(C_d_power % 2, identity):
                orders.append(expected_order)
            else:
                orders.append(M_p)  # Still full order somehow
        else:
            orders.append(M_p)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top: Order of C^d as a function of d
    ax1.plot(d_values, orders, 'b-', linewidth=0.8, alpha=0.7)
    ax1.scatter(d_values, orders, c=['#e74c3c' if o < M_p else '#3498db' for o in orders], 
                s=15, zorder=5)
    ax1.axhline(y=M_p, color='gray', linestyle='--', alpha=0.5, label=f'$M_{{11}} = {M_p}$')
    ax1.axhline(y=89, color='#e74c3c', linestyle=':', alpha=0.7, label='Order = 89 (= $M_p$/23)')
    ax1.axhline(y=23, color='#2ecc71', linestyle=':', alpha=0.7, label='Order = 23 (= $M_p$/89)')
    ax1.set_xlabel('d (exponent in $C^d$)', fontsize=12)
    ax1.set_ylabel('Order of $C^d$', fontsize=12)
    ax1.set_title(f'Order of $C^d$ Reveals Factors of $M_{{11}} = 2047 = 23 \\times 89$', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_yscale('log')
    
    # Bottom: GCD(M_p, d) reveals which d values give factor info
    gcds = [gcd(M_p, d) for d in d_values]
    colors = ['#e74c3c' if g > 1 else '#3498db' for g in gcds]
    ax2.bar(d_values, gcds, color=colors, edgecolor='none', width=1.0)
    ax2.set_xlabel('d', fontsize=12)
    ax2.set_ylabel('gcd($M_{11}$, d)', fontsize=12)
    ax2.set_title('gcd($M_{11}$, d) > 1 ⟹ Factor Revealed by CA', fontsize=13, fontweight='bold')
    
    # Annotate
    ax2.annotate('d=23: gcd=23\n→ factor 23', xy=(23, 23), xytext=(40, 50),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    ax2.annotate('d=89: gcd=89\n→ factor 89', xy=(89, 89), xytext=(60, 80),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_cd_factor_extraction.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_cd_factor_extraction.png")


def fig4_m23_cycle_spectrum():
    """
    Figure 4: Detailed cycle spectrum for M_23 = 8388607 = 47 × 178481.
    Shows the rich cycle structure that encodes the factorization.
    """
    from src.cycle_factor_extraction import squaring_map_cycle_decomposition
    
    info = squaring_map_cycle_decomposition(8388607)
    spectrum = info['cycle_spectrum']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Cycle spectrum bar chart
    lengths = sorted(spectrum.keys())
    counts = [spectrum[l] for l in lengths]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lengths)))
    
    bars = ax1.bar(range(len(lengths)), counts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels([str(l) for l in lengths], rotation=45)
    ax1.set_xlabel('Cycle Length', fontsize=12)
    ax1.set_ylabel('Number of Cycles', fontsize=12)
    ax1.set_title(r'$M_{23} = 8388607 = 47 \times 178481$' + '\nSquaring Map Cycle Spectrum', 
                  fontsize=13, fontweight='bold')
    
    # Annotate factor connection
    ax1.annotate('Length 11: ord(2, 23)=11\n(factor 47: 47-1=46=2×23)', 
                xy=(2, 52), fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Right: Log-scale version
    ax2.bar(range(len(lengths)), [np.log10(c) for c in counts], color=colors, 
            edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(lengths)))
    ax2.set_xticklabels([str(l) for l in lengths], rotation=45)
    ax2.set_xlabel('Cycle Length', fontsize=12)
    ax2.set_ylabel('log₁₀(Number of Cycles)', fontsize=12)
    ax2.set_title(r'$M_{23}$ Cycle Spectrum (Log Scale)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_m23_cycle_spectrum.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_m23_cycle_spectrum.png")


def fig5_fermat_proth_ca():
    """
    Figure 5: Fermat and Proth CA results.
    Shows Pepin's test results and Proth fold rule complexity.
    """
    from src.gf2_generalized_ca import FermatGF2CA, ProthGF2CA, ProthTestSuite, is_prime_simple
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: Fermat Pepin's test
    ax = axes[0]
    n_values = list(range(6))
    is_prime_list = []
    for n in n_values:
        ca = FermatGF2CA(n)
        result = ca.pepin_test_ca()
        is_prime_list.append(result['is_prime'])
    
    colors = ['#2ecc71' if p else '#e74c3c' for p in is_prime_list]
    labels = ['PRIME' if p else 'composite' for p in is_prime_list]
    bars = ax.bar(n_values, [1]*6, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(n_values)
    ax.set_xticklabels([f'$F_{{{n}}}$' for n in n_values])
    ax.set_ylabel('Primality', fontsize=12)
    ax.set_title("Pepin's Test via CA\n(Negative Fold)", fontsize=13, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Composite', 'Prime'])
    
    for i, (n, is_p) in enumerate(zip(n_values, is_prime_list)):
        F_n = (1 << (1 << n)) + 1
        F_str = str(F_n) if F_n < 10**10 else f'$2^{{2^{n}}}+1$'
        ax.text(i, 1.05, F_str, ha='center', fontsize=8, rotation=30)
    
    # Middle: Proth test results
    ax = axes[1]
    suite = ProthTestSuite()
    results = suite.run_tests()
    
    k_values = [r['k'] for r in results]
    n_values_proth = [r['n'] for r in results]
    P_values = [r['P'] for r in results]
    correct = [r['ca_correct'] for r in results]
    
    colors_proth = ['#2ecc71' if c else '#e74c3c' for c in correct]
    ax.scatter(k_values, n_values_proth, c=colors_proth, s=80, edgecolor='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('n', fontsize=12)
    ax.set_title("Proth's Theorem via CA\n(Weighted Fold)", fontsize=13, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Correct'),
                       Patch(facecolor='#e74c3c', edgecolor='black', label='Incorrect')]
    ax.legend(handles=legend_elements, loc='best')
    
    # Right: CA rule hierarchy
    ax = axes[2]
    ax.axis('off')
    
    hierarchy_text = """
    CA FOLD HIERARCHY FOR PRIMALITY TESTING
    ────────────────────────────────────────
    
    MERSENNE  2^p - 1
    ├── Fold: POSITIVE (add upper + lower)
    ├── 2^p ≡ +1 (mod M_p)
    ├── GF(2) CA: Cycle structure → primality
    └── Rule: Rule 90 (pure XOR)
    
    FERMAT   2^(2^n) + 1
    ├── Fold: NEGATIVE (lower - upper)
    ├── 2^(2^n) ≡ -1 (mod F_n)
    ├── GF(2) CA: Computation only
    └── Rule: Signed Rule 90
    
    PROTH    k·2^n + 1
    ├── Fold: WEIGHTED (lower - upper·k⁻¹)
    ├── 2^n ≡ -k⁻¹ (mod P)
    ├── GF(2) CA: Computation only
    └── Rule: Rule 90 with coefficients
    
    KEY: Only Mersenne allows cycle-based
    primality. Fermat/Proth need computation.
    """
    
    ax.text(0.05, 0.95, hierarchy_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Generalized CA Primality Testing: Fermat & Proth Numbers', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_fermat_proth_ca.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_fermat_proth_ca.png")


def fig6_mersenne_cycle_structure_detailed():
    """
    Figure 6: Detailed cycle structure analysis for M_11 showing
    how cycle lengths connect to factors.
    """
    from src.cycle_factor_extraction import squaring_map_cycle_decomposition
    
    # Compute cycle structure for Z/23 and Z/89 (the factors of M_11)
    info_23 = squaring_map_cycle_decomposition(23)
    info_89 = squaring_map_cycle_decomposition(89)
    info_2047 = squaring_map_cycle_decomposition(2047)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, (N, info, title) in zip(axes, [
        (23, info_23, r'$\mathbb{Z}/23\mathbb{Z}$ (factor 23)'),
        (89, info_89, r'$\mathbb{Z}/89\mathbb{Z}$ (factor 89)'),
        (2047, info_2047, r'$\mathbb{Z}/2047\mathbb{Z}$ ($M_{11} = 23 \times 89$)')
    ]):
        spectrum = info['cycle_spectrum']
        lengths = sorted(spectrum.keys())
        counts = [spectrum[l] for l in lengths]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(lengths)))
        bars = ax.bar(range(len(lengths)), counts, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([str(l) for l in lengths])
        ax.set_xlabel('Cycle Length', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    fig.suptitle('CRT Decomposition: Cycle Structure of $x \\to x^2$ mod $N$\n'
                 'Cycle lengths in $\\mathbb{Z}/2047$ encode factorization $23 \\times 89$', 
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_crt_cycle_decomposition.png', 
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved fig_crt_cycle_decomposition.png")


def generate_all_figures():
    """Generate all visualization figures."""
    print("Generating visualization figures...")
    print()
    
    print("  Figure 1: Prime vs Composite Cycle Spectrum")
    fig1_prime_vs_composite_spectrum()
    
    print("  Figure 2: Factor Recovery Rates")
    fig2_factor_recovery_rates()
    
    print("  Figure 3: C^d Factor Extraction Method")
    fig3_cd_factor_extraction()
    
    print("  Figure 4: M_23 Cycle Spectrum")
    fig4_m23_cycle_spectrum()
    
    print("  Figure 5: Fermat/Proth CA Results")
    fig5_fermat_proth_ca()
    
    print("  Figure 6: CRT Cycle Decomposition")
    fig6_mersenne_cycle_structure_detailed()
    
    print("\nAll figures generated!")


if __name__ == "__main__":
    generate_all_figures()
