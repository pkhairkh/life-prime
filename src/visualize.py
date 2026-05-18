"""
Comprehensive Visualization and Analysis
=========================================

Generates all plots and analysis figures demonstrating:
1. Rule 90 Sierpiński triangle with Mersenne number highlights
2. Gould's sequence and Mersenne number connection
3. Rule 90 cyclic grid period analysis
4. LFSR period vs Mersenne prime relationship
5. LLT bit evolution as CA
6. Game of Life population dynamics with prime detection
7. Summary diagram of all connections
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import json
import os
import sys

# Font setup
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC']
except Exception:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

RESULTS_DIR = '/home/z/my-project/life-prime/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ============================================================
# Figure 1: Rule 90 Sierpiński Triangle with Mersenne Highlights
# ============================================================
def fig_rule90_sierpinski():
    """Generate Sierpiński triangle from Rule 90 with Mersenne step highlights."""
    width, steps = 257, 128

    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1
    history = [state.copy()]

    for _ in range(steps):
        left = np.zeros_like(state)
        left[1:] = state[:-1]
        right = np.zeros_like(state)
        right[:-1] = state[1:]
        state = (left ^ right).astype(np.uint8)
        history.append(state.copy())

    history_array = np.array(history)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Left: Full Sierpiński triangle
    ax = axes[0]
    ax.imshow(history_array, cmap='binary', aspect='auto', interpolation='nearest')
    ax.set_title('Rule 90: Sierpiński Triangle from Single Seed', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Generation (step n)')

    # Highlight Mersenne-numbered rows
    mersenne_rows = []
    for k in range(1, 8):
        n = 2**k - 1
        if n < steps:
            mersenne_rows.append(n)

    for n in mersenne_rows:
        ax.axhline(y=n, color='red', alpha=0.4, linewidth=0.8)
        ax.text(width + 2, n, f'n=2^{int(np.log2(n+1))}-1={n}',
                fontsize=7, color='red', va='center')

    # Right: Live cell counts with Gould's sequence
    ax = axes[1]
    live_counts = [int(np.sum(h)) for h in history]
    gould = [2 ** bin(i).count('1') for i in range(len(history))]

    ax.plot(range(len(live_counts)), live_counts, 'b-', alpha=0.7, linewidth=0.8, label='Rule 90 live cells')
    ax.plot(range(len(gould)), gould, 'r--', alpha=0.7, linewidth=0.8, label="Gould's sequence 2^popcount(n)")
    ax.set_title("Live Cell Count = Gould's Sequence = 2^popcount(n)", fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation n')
    ax.set_ylabel('Number of live cells')
    ax.legend(loc='best')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Highlight Mersenne steps
    for k in range(1, 8):
        n = 2**k - 1
        if n < len(live_counts):
            ax.annotate(f'M_{k}={n}\ncount={live_counts[n]}',
                       xy=(n, live_counts[n]), xytext=(n+5, live_counts[n]*1.2),
                       fontsize=7, color='red',
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.6))

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig1_rule90_sierpinski.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 1: Rule 90 Sierpiński triangle saved")


# ============================================================
# Figure 2: Mersenne Number Connection - Maximum Live Cells
# ============================================================
def fig_mersenne_peak():
    """Show that Mersenne-numbered steps give maximum live cells."""
    width, steps = 513, 256

    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1

    live_counts = [int(np.sum(state))]
    for _ in range(steps):
        left = np.zeros_like(state)
        left[1:] = state[:-1]
        right = np.zeros_like(state)
        right[:-1] = state[1:]
        state = (left ^ right).astype(np.uint8)
        live_counts.append(int(np.sum(state)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Live cell counts with Mersenne peaks highlighted
    ax = axes[0, 0]
    ax.plot(range(len(live_counts)), live_counts, 'b-', alpha=0.5, linewidth=0.6)

    mersenne_exponents = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 127}
    for k in range(1, 9):
        n = 2**k - 1
        if n < len(live_counts):
            is_mp = k in mersenne_exponents
            color = 'red' if is_mp else 'orange'
            marker = '*' if is_mp else 'o'
            size = 100 if is_mp else 30
            label = f'M_{k}={n}' + (' (prime)' if is_mp else '')
            ax.scatter(n, live_counts[n], c=color, marker=marker, s=size, zorder=5, label=label)

    ax.set_title('Rule 90: Live Cell Peaks at Mersenne Numbers n=2^k-1', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation n')
    ax.set_ylabel('Live cells')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Top-right: Binary representation showing why Mersenne numbers maximize popcount
    ax = axes[0, 1]
    k_values = range(1, 9)
    mersenne_ns = [2**k - 1 for k in k_values]
    expected_counts = [2**k for k in k_values]

    # Also show a non-Mersenne number with same bit-width for comparison
    random_ns = [2**k - 2 for k in k_values]  # One less than Mersenne
    random_counts = [2 ** bin(n).count('1') for n in random_ns]

    x = np.arange(len(k_values))
    width_bar = 0.35
    bars1 = ax.bar(x - width_bar/2, expected_counts, width_bar, label='n=2^k-1 (Mersenne)', color='crimson', alpha=0.8)
    bars2 = ax.bar(x + width_bar/2, random_counts, width_bar, label='n=2^k-2 (non-Mersenne)', color='steelblue', alpha=0.8)

    ax.set_xlabel('k')
    ax.set_ylabel('Live cells at step n')
    ax.set_title('Mersenne Numbers Maximize Live Cell Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Pascal's triangle mod 2 - visual
    ax = axes[1, 0]
    pascal_size = 32
    pascal = np.zeros((pascal_size, 2*pascal_size+1), dtype=np.uint8)
    pascal[0, pascal_size] = 1
    for i in range(1, pascal_size):
        for j in range(1, 2*pascal_size):
            pascal[i, j] = pascal[i-1, j-1] ^ pascal[i-1, j+1]

    ax.imshow(pascal, cmap='binary', aspect='auto', interpolation='nearest')

    # Highlight prime rows
    for i in range(pascal_size):
        if is_prime(i):
            ax.axhline(y=i, color='red', alpha=0.3, linewidth=0.5)

    ax.set_title("Pascal's Triangle mod 2 = Rule 90 Evolution", fontsize=12, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Row n (= generation)')

    # Bottom-right: Binary weight (popcount) analysis
    ax = axes[1, 1]
    n_values = np.arange(256)
    popcounts = [bin(n).count('1') for n in n_values]
    gould_values = [2**pc for pc in popcounts]

    colors = ['red' if is_prime(n) else 'steelblue' for n in n_values]
    ax.scatter(n_values, gould_values, c=colors, s=3, alpha=0.6)
    ax.set_title("Gould's Sequence: 2^popcount(n) — Red = prime n", fontsize=12, fontweight='bold')
    ax.set_xlabel('n')
    ax.set_ylabel('2^popcount(n)')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig2_mersenne_peak.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 2: Mersenne peak analysis saved")


# ============================================================
# Figure 3: Cyclic Rule 90 Period Analysis
# ============================================================
def fig_cyclic_periods():
    """Analyze Rule 90 periods on cyclic grids and their Mersenne connection."""
    # Pre-computed periods for small cyclic grids
    # For even N=2k: max period = 2^k - 1

    def compute_period(size, max_steps=2**22):
        state = np.zeros(size, dtype=np.uint8)
        state[0] = 1
        seen = {state.tobytes(): 0}

        for step in range(1, max_steps):
            left = np.roll(state, 1)
            right = np.roll(state, -1)
            state = (left ^ right).astype(np.uint8)
            sb = state.tobytes()
            if sb in seen:
                return step - seen[sb]
            seen[sb] = step
        return None

    sizes = list(range(2, 28))
    periods = []
    theoretical_maxes = []

    for size in sizes:
        if size % 2 == 0:
            k = size // 2
            theoretical_max = 2**k - 1
        else:
            theoretical_max = None

        max_steps = min(2**(size//2 + 2), 2**24)
        period = compute_period(size, max_steps) if size <= 26 else None

        periods.append(period)
        theoretical_maxes.append(theoretical_max)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Period vs grid size
    ax = axes[0, 0]
    even_sizes = [(s, p, t) for s, p, t in zip(sizes, periods, theoretical_maxes) if s % 2 == 0 and p is not None]
    odd_sizes = [(s, p) for s, p, t in zip(sizes, periods, theoretical_maxes) if s % 2 != 0 and p is not None]

    if even_sizes:
        es, ep, et = zip(*even_sizes)
        ax.scatter(es, ep, c='crimson', s=50, label='Actual period (even N)', zorder=5)
        ax.scatter(es, et, c='steelblue', marker='x', s=50, label='Max period 2^(N/2)-1')

    if odd_sizes:
        os_vals, op = zip(*odd_sizes)
        ax.scatter(os_vals, op, c='green', s=30, label='Actual period (odd N)')

    ax.set_title('Rule 90 Period on Cyclic Grids', fontsize=12, fontweight='bold')
    ax.set_xlabel('Grid size N')
    ax.set_ylabel('Period')
    ax.set_yscale('log', base=2)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: Period / Theoretical Max ratio for even grids
    ax = axes[0, 1]
    if even_sizes:
        es, ep, et = zip(*even_sizes)
        ratios = [p / t if t else 0 for p, t in zip(ep, et)]
        mersenne_exponents = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}

        colors = []
        for s in es:
            k = s // 2
            if k in mersenne_exponents:
                colors.append('red')
            else:
                colors.append('steelblue')

        ax.bar(es, ratios, color=colors, alpha=0.8)
        ax.set_title('Period Achievement Ratio (even N=2k)\nRed = k is Mersenne prime exponent', fontsize=11, fontweight='bold')
        ax.set_xlabel('Grid size N (=2k)')
        ax.set_ylabel('Period / Max Period')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Mersenne number periods specifically
    ax = axes[1, 0]
    mersenne_k = list(range(1, 13))
    mersenne_nums = [2**k - 1 for k in mersenne_k]
    mersenne_periods = []

    for k in mersenne_k:
        size = 2 * k
        if size <= 26:
            p = compute_period(size, 2**(k+2))
            mersenne_periods.append(p)
        else:
            mersenne_periods.append(None)

    mp_flags = [k in {2, 3, 5, 7, 13, 17, 19, 31} for k in mersenne_k]

    for i, (k, m, p, mp) in enumerate(zip(mersenne_k, mersenne_nums, mersenne_periods, mp_flags)):
        if p is not None:
            color = 'crimson' if mp else 'steelblue'
            label = f'2^{k}-1={m}' + (' (prime!)' if mp else '')
            ax.bar(k, p, color=color, alpha=0.8, label=label if i < 8 else None)

    ax.set_title('Rule 90 Period for Grid Size N=2k\nPeriod = 2^k - 1 when polynomial is primitive', fontsize=11, fontweight='bold')
    ax.set_xlabel('k')
    ax.set_ylabel('Period')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-right: LFSR equivalence diagram
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'Rule 90 on Cyclic Grid = LFSR', fontsize=14, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.82, 'Linear operator over GF(2)\nCharacteristic poly = factor of x^N + 1',
            fontsize=10, ha='center', va='top', transform=ax.transAxes)

    connections = [
        ('Rule 90\n(cyclic)', 'LFSR\n(register len k=N/2)'),
        ('LFSR', 'Max period\n= 2^k - 1'),
        ('Max period', 'Mersenne\nnumber'),
        ('Mersenne\nprime M_k', 'Primitive\npoly over GF(2)'),
        ('Primitive\npoly', 'Single cycle\nof length M_k'),
    ]

    y_positions = [0.65, 0.50, 0.35, 0.20]
    labels_left = ['Rule 90\n(cyclic)', 'LFSR', 'Max period\n= Mersenne #', 'Primitive\npolynomial']
    labels_right = ['LFSR\n(register k=N/2)', 'Max period\n= 2^k - 1', 'Mersenne\nprime M_k', 'Single cycle\nlength M_k']

    for y, ll, lr in zip(y_positions, labels_left, labels_right):
        ax.text(0.15, y, ll, fontsize=9, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        ax.text(0.85, y, lr, fontsize=9, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))
        ax.annotate('', xy=(0.75, y), xytext=(0.25, y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                   transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig3_cyclic_periods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 3: Cyclic period analysis saved")


# ============================================================
# Figure 4: Lucas-Lehmer Test as CA
# ============================================================
def fig_llt_ca():
    """Visualize the LLT as a bit-level cellular automaton."""

    def lucas_lehmer(p):
        if p == 2:
            return True, [4]
        m_p = 2**p - 1
        s = 4
        seq = [s]
        for _ in range(p - 2):
            s = (s * s - 2) % m_p
            seq.append(s)
        return (s == 0), seq

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: LLT bit evolution for p=5 (M_5=31, prime)
    ax = axes[0, 0]
    is_prime, seq = lucas_lehmer(5)
    bit_grid = np.zeros((len(seq), 5), dtype=np.uint8)
    for i, s in enumerate(seq):
        for j in range(5):
            bit_grid[i, j] = (s >> j) & 1

    ax.imshow(bit_grid, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    ax.set_title('LLT Bit Evolution: p=5, M_5=31 (PRIME)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bit position')
    ax.set_ylabel('LLT step')
    ax.set_yticks(range(len(seq)))
    ax.set_yticklabels([f's_{i}={seq[i]}' for i in range(len(seq))], fontsize=7)

    # Top-right: LLT bit evolution for p=11 (M_11=2047, composite)
    ax = axes[0, 1]
    is_prime2, seq2 = lucas_lehmer(11)
    bit_grid2 = np.zeros((min(20, len(seq2)), 11), dtype=np.uint8)
    for i in range(min(20, len(seq2))):
        for j in range(11):
            bit_grid2[i, j] = (seq2[i] >> j) & 1

    ax.imshow(bit_grid2, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    ax.set_title('LLT Bit Evolution: p=11, M_11=2047 (composite)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Bit position')
    ax.set_ylabel('LLT step')

    # Bottom-left: Hamming weight evolution for various exponents
    ax = axes[1, 0]
    exponents = [3, 5, 7, 11, 13, 17, 19, 23]
    for p in exponents:
        ip, sq = lucas_lehmer(p)
        hw = [bin(s).count('1') for s in sq]
        style = '-' if ip else '--'
        lw = 2 if ip else 1
        label = f'p={p}' + (' (prime!)' if ip else ' (comp.)')
        ax.plot(range(len(hw)), hw, style, linewidth=lw, alpha=0.7, label=label)

    ax.set_title('LLT Hamming Weight Evolution', fontsize=12, fontweight='bold')
    ax.set_xlabel('LLT step')
    ax.set_ylabel('Hamming weight of s_i')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Modular reduction as CA fold
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'Modular Reduction mod M_p = CA Fold',
            fontsize=14, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    # Demonstrate the fold operation
    p = 7
    M_p = 2**7 - 1
    examples = [128, 200, 255, 300, 500]

    ax.text(0.5, 0.82, f'For M_p = 2^{p}-1 = {M_p}:\n2^{p} ≡ 1 (mod M_p)\nSo reduction = FOLD: split at bit p, XOR halves',
            fontsize=10, ha='center', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    y_start = 0.60
    for val in examples:
        upper = val >> p
        lower = val & M_p
        folded = upper + lower
        actual = val % M_p

        ax.text(0.5, y_start,
                f'{val} = {upper}×2^{p} + {lower} → {upper} + {lower} = {folded} = {val} mod {M_p}',
                fontsize=9, ha='center', va='top', transform=ax.transAxes,
                family='monospace')
        y_start -= 0.08

    ax.text(0.5, 0.10, 'This fold is IDENTICAL in spirit to Rule 90:\nXOR of spatially separated cells!',
            fontsize=11, ha='center', va='top', transform=ax.transAxes,
            fontweight='bold', color='crimson',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig4_llt_ca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 4: LLT as CA saved")


# ============================================================
# Figure 5: Game of Life Primer Mechanism
# ============================================================
def fig_gol_primer():
    """Visualize the Game of Life primer mechanism."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Sieve visualization
    ax = axes[0, 0]
    max_n = 50
    primes = []
    is_prime_arr = [True] * (max_n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for p in range(2, max_n + 1):
        if is_prime_arr[p]:
            primes.append(p)
            for m in range(2*p, max_n + 1, p):
                is_prime_arr[m] = False

    # Create sieve grid
    prime_list = [p for p in primes if p <= max_n][:12]  # First 12 primes
    grid = np.zeros((len(prime_list), max_n + 1), dtype=np.uint8)

    for i, p in enumerate(prime_list):
        for m in range(2*p, max_n + 1, p):
            grid[i, m] = 1

    # Mark primes
    for n in range(2, max_n + 1):
        if is_prime_arr[n]:
            grid[:, n] = 2  # Special value for primes

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'steelblue', 'crimson'])
    ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')
    ax.set_title("Primer Sieve: Prime Guns Destroy Composite LWSS", fontsize=12, fontweight='bold')
    ax.set_xlabel('Number N')
    ax.set_ylabel('Prime gun p')
    ax.set_yticks(range(len(prime_list)))
    ax.set_yticklabels([f'p={p}' for p in prime_list], fontsize=8)

    # Top-right: LWSS escape pattern
    ax = axes[0, 1]
    n_values = list(range(2, 51))
    escape = [1 if is_prime_arr[n] else 0 for n in n_values]

    ax.bar(n_values, escape, color=['crimson' if e else 'steelblue' for e in escape], alpha=0.8)
    ax.set_title('Primer Output: LWSS Escapes iff N is Prime', fontsize=12, fontweight='bold')
    ax.set_xlabel('N (generation / 120)')
    ax.set_ylabel('LWSS escapes?')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No (composite)', 'Yes (PRIME)'])

    # Bottom-left: GoL population dynamics
    ax = axes[1, 0]

    # Simulate GoL
    size = 40
    gol_grid = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.RandomState(42)
    for y in range(size):
        for x in range(size):
            if rng.random() < 0.3:
                gol_grid[y, x] = 1

    populations = [int(np.sum(gol_grid))]
    for _ in range(200):
        neighbors = np.zeros_like(gol_grid, dtype=np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors += np.roll(np.roll(gol_grid, dy, axis=0), dx, axis=1)

        new_grid = np.zeros_like(gol_grid)
        new_grid[(gol_grid == 0) & (neighbors == 3)] = 1
        new_grid[(gol_grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1
        gol_grid = new_grid
        populations.append(int(np.sum(gol_grid)))

    ax.plot(range(len(populations)), populations, 'b-', alpha=0.7, linewidth=0.8)
    ax.set_title('GoL Population Dynamics (random start)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Live cells')
    ax.grid(True, alpha=0.3)

    # Highlight prime generations
    prime_gens = [i for i in range(len(populations)) if is_prime(i)]
    prime_pops = [populations[i] for i in prime_gens]
    ax.scatter(prime_gens, prime_pops, c='red', s=5, alpha=0.5, zorder=5, label='Prime generations')
    ax.legend(loc='best', fontsize=8)

    # Bottom-right: Primer mechanism explanation
    ax = axes[1, 1]
    ax.text(0.5, 0.95, "Dean Hickerson's Primer (1991)",
            fontsize=14, fontweight='bold', ha='center', va='top', transform=ax.transAxes)

    explanation = (
        "HOW IT WORKS:\n\n"
        "1. Counter machines count through N = 1, 2, 3, ...\n"
        "   (implemented as glider streams in loops)\n\n"
        "2. For each N, a Lightweight Spaceship (LWSS)\n"
        "   is fired westward\n\n"
        "3. For each prime p, a Gosper Glider Gun fires\n"
        "   interceptors that destroy the LWSS at\n"
        "   generations 120p, 120(2p), 120(3p), ...\n\n"
        "4. LWSS survives (escapes) iff N is NOT a\n"
        "   multiple of any smaller number\n\n"
        "5. Surviving LWSS → N is PRIME\n\n"
        "This is the Sieve of Eratosthenes,\n"
        "implemented in Conway's Game of Life!"
    )
    ax.text(0.5, 0.82, explanation, fontsize=9, ha='center', va='top',
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig5_gol_primer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 5: GoL Primer visualization saved")


# ============================================================
# Figure 6: Grand Summary — All Connections
# ============================================================
def fig_summary():
    """Create a comprehensive summary diagram of all connections."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.7, "Cellular Automata and Prime/Mersenne Prime Prediction",
            fontsize=16, fontweight='bold', ha='center', va='top')
    ax.text(5, 9.3, "How and Why: Complete Connection Map",
            fontsize=12, ha='center', va='top', color='gray')

    # Node positions and labels
    nodes = {
        'rule90': (2.5, 8, "Rule 90\n(s_i = left XOR right)", 'lightblue'),
        'sierpinski': (2.5, 6.5, "Sierpinski Triangle\n= Pascal mod 2", 'lightyellow'),
        'gould': (0.8, 7.5, "Gould's Seq\n2^popcount(n)", 'lightgreen'),
        'mersenne_num': (5, 7.5, "Mersenne Numbers\nn = 2^k - 1", 'lightsalmon'),
        'cyclic_r90': (7.5, 8, "Rule 90\n(cyclic grid)", 'lightblue'),
        'lfsr': (7.5, 6.5, "LFSR\n(Linear Feedback\nShift Register)", 'lightyellow'),
        'max_period': (5, 5.5, "Max Period = 2^k - 1\n(Mersenne Number!)", 'lightsalmon'),
        'primitive_poly': (9, 5, "Primitive Poly\nover GF(2)", 'lightgreen'),
        'mersenne_prime': (5, 3.5, "MERSENNE PRIME\nM_p = 2^p - 1", 'crimson'),
        'llt': (2.5, 4.5, "Lucas-Lehmer\nTest (LLT)", 'lightyellow'),
        'llt_ca': (0.8, 3, "LLT as CA:\nsquaring + fold + -2", 'lightgreen'),
        'fold': (2.5, 1.5, "mod M_p = FOLD\n(shift & XOR\n= Rule 90!)", 'lightblue'),
        'gol': (7.5, 3.5, "Conway's Game\nof Life", 'lightyellow'),
        'primer': (7.5, 1.5, "Hickerson's Primer\n(Sieve in GoL)", 'lightgreen'),
        'wolfram': (9.5, 2, "Wolfram's\n16-color CA", 'lightyellow'),
    }

    # Draw nodes
    for name, (x, y, label, color) in nodes.items():
        alpha = 0.9
        if color == 'crimson':
            box = FancyBboxPatch((x-1, y-0.4), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color, alpha=0.3,
                                edgecolor=color, linewidth=2)
            ax.add_patch(box)
        else:
            box = FancyBboxPatch((x-1, y-0.4), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color, alpha=alpha,
                                edgecolor='gray', linewidth=1)
            ax.add_patch(box)
        ax.text(x, y, label, fontsize=8, ha='center', va='center',
                fontweight='bold' if name == 'mersenne_prime' else 'normal')

    # Draw connections (arrows)
    connections = [
        ('rule90', 'sierpinski', 'generates'),
        ('rule90', 'gould', 'live cells ='),
        ('sierpinski', 'mersenne_num', 'max at n=2^k-1'),
        ('gould', 'mersenne_num', 'peak when all bits=1'),
        ('cyclic_r90', 'lfsr', 'equivalent to'),
        ('lfsr', 'max_period', 'period ≤ 2^k - 1'),
        ('max_period', 'mersenne_prime', 'when 2^k-1 prime'),
        ('primitive_poly', 'max_period', 'implies max period'),
        ('lfsr', 'primitive_poly', 'needs'),
        ('mersenne_prime', 'llt', 'tested by'),
        ('llt', 'llt_ca', 'implemented as'),
        ('llt_ca', 'fold', 'uses'),
        ('fold', 'rule90', 'same XOR structure'),
        ('gol', 'primer', 'Turing-complete'),
        ('primer', 'mersenne_prime', 'detects primes\nfor M_p test'),
        ('wolfram', 'mersenne_prime', 'sieve gaps\n= primes'),
        ('cyclic_r90', 'rule90', 'same rule,\ncyclic boundary'),
    ]

    for src_name, dst_name, label in connections:
        sx, sy = nodes[src_name][0], nodes[src_name][1]
        dx, dy = nodes[dst_name][0], nodes[dst_name][1]

        # Simple arrow
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                   arrowprops=dict(arrowstyle='->', color='gray',
                                  lw=1.2, connectionstyle='arc3,rad=0.1'))

        # Label at midpoint
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        ax.text(mx, my, f' {label}', fontsize=6, ha='center', va='center',
                color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

    # Key insight box
    insight_text = (
        "THE DEEP CONNECTION:\n"
        "Rule 90 is linear over GF(2), making it equivalent to an LFSR on cyclic grids.\n"
        "LFSR max period = 2^k - 1 (Mersenne number), achieved with primitive polynomials.\n"
        "When 2^k - 1 is MERSENNE PRIME, all non-zero states form a single maximal cycle.\n"
        "The LLT tests Mersenne primes using squaring (CA convolution) + fold (CA XOR) — \n"
        "the SAME algebraic structure as Rule 90."
    )
    ax.text(5, 0.5, insight_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9,
                     edgecolor='crimson', linewidth=2))

    plt.savefig(f'{RESULTS_DIR}/fig6_summary_connections.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 6: Summary connection map saved")


# ============================================================
# Figure 7: Comprehensive LLT Results Table
# ============================================================
def fig_llt_results():
    """Create a detailed LLT results visualization."""
    def lucas_lehmer(p):
        if p == 2:
            return True, [4]
        m_p = 2**p - 1
        s = 4
        seq = [s]
        for _ in range(p - 2):
            s = (s * s - 2) % m_p
            seq.append(s)
        return (s == 0), seq

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: LLT sequence values for various exponents
    ax = axes[0]
    test_exponents = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    for p in test_exponents:
        ip, seq = lucas_lehmer(p)
        style = 'o-' if ip else 'x--'
        lw = 2 if ip else 1
        ms = 5 if ip else 3
        label = f'p={p}' + (' (prime!)' if ip else '')
        ax.plot(range(len(seq)), seq, style, linewidth=lw, markersize=ms,
                alpha=0.7, label=label)

    ax.set_title('LLT Sequence Values s_i for Various p', fontsize=12, fontweight='bold')
    ax.set_xlabel('Step i')
    ax.set_ylabel('s_i mod M_p')
    ax.set_yscale('symlog')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Right: Table of Mersenne prime test results
    ax = axes[1]
    ax.axis('off')

    mersenne_exponents = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    results = []
    for p in mersenne_exponents:
        ip, seq = lucas_lehmer(p)
        m_p = 2**p - 1
        results.append((p, m_p, ip, seq[-1]))

    # Create table
    col_labels = ['p', 'M_p = 2^p-1', 'Prime?', 'Final s_{p-2}']
    table_data = [[str(r[0]), str(r[1]), 'YES' if r[2] else 'no', str(r[3])] for r in results]

    table = ax.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Color prime rows
    for i, r in enumerate(results):
        if r[2]:
            for j in range(4):
                table[i+1, j].set_facecolor('lightyellow')

    ax.set_title('Lucas-Lehmer Test Results for Mersenne Candidates', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig7_llt_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 7: LLT results table saved")


# ============================================================
# Figure 8: Rule 90 Pattern Comparison at Different Steps
# ============================================================
def fig_rule90_patterns():
    """Show Rule 90 patterns at prime vs composite vs Mersenne steps."""
    width, steps = 513, 256

    state = np.zeros(width, dtype=np.uint8)
    state[width // 2] = 1
    history = [state.copy()]

    for _ in range(steps):
        left = np.zeros_like(state)
        left[1:] = state[:-1]
        right = np.zeros_like(state)
        right[:-1] = state[1:]
        state = (left ^ right).astype(np.uint8)
        history.append(state.copy())

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))

    interesting_steps = [
        (2, 'Prime: n=2'), (3, 'Prime: n=3'), (5, 'Prime: n=5'), (7, 'Prime: n=7'),
        (3, 'Mersenne: n=3=2^2-1'), (7, 'Mersenne: n=7=2^3-1'),
        (15, 'Mersenne: n=15=2^4-1'), (31, 'Mersenne: n=31=2^5-1'),
        (4, 'Composite: n=4'), (6, 'Composite: n=6'), (8, 'Composite: n=8'), (9, 'Composite: n=9'),
    ]

    row_labels = ['Prime Steps', 'Mersenne Steps', 'Composite Steps']

    for col in range(4):
        for row in range(3):
            ax = axes[row, col]
            idx = row * 4 + col
            if idx < len(interesting_steps):
                step, title = interesting_steps[idx]
                if step < len(history):
                    # Show center portion
                    center = width // 2
                    half = min(64, step + 5)
                    pattern = history[step][center-half:center+half+1]
                    ax.imshow(pattern.reshape(1, -1), cmap='binary', aspect='auto',
                             interpolation='nearest')
                    lc = int(np.sum(pattern))
                    ax.set_title(f'{title}\n(live={lc})', fontsize=8)
                    ax.set_yticks([])

    plt.suptitle('Rule 90 Patterns: Prime vs Mersenne vs Composite Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig8_rule90_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 8: Rule 90 pattern comparison saved")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating comprehensive visualizations...")
    print()

    fig_rule90_sierpinski()
    fig_mersenne_peak()
    fig_cyclic_periods()
    fig_llt_ca()
    fig_gol_primer()
    fig_summary()
    fig_llt_results()
    fig_rule90_patterns()

    print()
    print(f"All figures saved to {RESULTS_DIR}/")
    print("Figures generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith('.png'):
            fpath = os.path.join(RESULTS_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")
