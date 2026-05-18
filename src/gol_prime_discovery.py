#!/usr/bin/env python3
"""
GoL-Prime Discovery Engine
===========================
MASSIVE statistical discovery of prime-number correlations in
Conway's Game of Life dynamics.

This module performs data-driven simulation to discover NOVEL, previously
unknown correlations between GoL evolution properties and number-theoretic
properties (especially primality and Mersenne primes).

Seven core components:
  1. Massive GoL Simulator (numpy + scipy optimized)
  2. Structured Initial Condition Generator (number-encoded patterns)
  3. Feature Extraction (population, entropy, autocorrelation, oscillation...)
  4. Primality Correlation Discovery (chi-squared, mutual information, etc.)
  5. Machine Learning Prime Predictor (sklearn classifiers + feature importances)
  6. Mersenne-Specific Analysis (phase transitions at Mersenne primes)
  7. Novel Discovery Engine (systematic search for "magic generations")

Scientific rigor: honest p-values, Bonferroni correction, confidence intervals,
and transparent reporting of negative results.

Author: GoL-Prime Research Project
"""

import json
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq

# Suppress noisy sklearn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = "/home/z/my-project/life-prime/results"
N_RANGE = range(2, 501)           # integers to test
GOL_GENERATIONS = 200             # generations per simulation
GRID_SIZE = 80                    # grid dimension (GRID_SIZE x GRID_SIZE)
BORDER = 10                       # dead border to reduce wrap artefacts
RANDOM_SEED = 42
MERSENNE_P_RANGE = range(2, 32)   # p values for 2^p - 1

# Known Mersenne primes (exponents p where 2^p - 1 is prime) for p <= 31
KNOWN_MERSENNE_PRIMES_P = {2, 3, 5, 7, 13, 17, 19, 31}


# ---------------------------------------------------------------------------
# Utility: number theory helpers
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    """Deterministic primality test for n >= 2."""
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


def is_mersenne_number(n: int) -> bool:
    """True if n = 2^p - 1 for some integer p >= 2."""
    m = n + 1
    return m > 1 and (m & (m - 1)) == 0


def is_mersenne_prime(n: int) -> bool:
    """True if n is a Mersenne prime (2^p - 1 with p prime, n prime)."""
    return is_mersenne_number(n) and is_prime(n)


def prime_factors(n: int) -> List[int]:
    """Return sorted list of prime factors of n (with multiplicity)."""
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


def factor_pairs(n: int) -> List[Tuple[int, int]]:
    """Return all factor pairs (a, b) of n with a <= b."""
    pairs = []
    for a in range(1, int(math.isqrt(n)) + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs


def primes_up_to(n: int) -> List[int]:
    """Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if sieve[p]:
            for m in range(p * p, n + 1, p):
                sieve[m] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# Pre-compute for labelling
PRIMES_SET = set(primes_up_to(max(N_RANGE)))


# ---------------------------------------------------------------------------
# 1. Massive GoL Simulator
# ---------------------------------------------------------------------------
# Conway's kernel for neighbor counting (3x3 with center 0)
_CONV_KERNEL = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.int32)


class GoLSimulator:
    """
    Efficient numpy-based Game of Life simulator.

    Uses scipy.signal.convolve2d for neighbor counting, which is significantly
    faster than the naive roll-based approach for large grids.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)

    def set_cell(self, x: int, y: int, value: int = 1):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = value

    def set_pattern(self, pattern: List[Tuple[int, int]],
                    offset: Tuple[int, int] = (0, 0)):
        for x, y in pattern:
            self.set_cell(x + offset[0], y + offset[1])

    def step(self) -> np.ndarray:
        """Advance one generation. Returns new grid."""
        neighbors = signal.convolve2d(self.grid, _CONV_KERNEL,
                                      mode='same', boundary='fill')
        birth = (self.grid == 0) & (neighbors == 3)
        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))
        self.grid = (birth | survive).astype(np.uint8)
        return self.grid

    def run(self, steps: int) -> List[np.ndarray]:
        """Run for *steps* generations, returning list of grids."""
        history = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())
        return history

    def population(self) -> int:
        return int(self.grid.sum())

    def reset(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 2. Structured Initial Condition Generator
# ---------------------------------------------------------------------------

# --- Common GoL patterns (relative coordinates) ---
GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
LWSS = [(0, 1), (0, 3), (1, 0), (2, 0), (3, 0), (3, 3), (4, 0), (4, 1), (4, 2)]
BLINKER = [(0, 0), (1, 0), (2, 0)]
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]
BEACON = [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)]
TOAD = [(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)]

# Gosper Glider Gun (period 30, emits a glider every 30 gens)
GOSPER_GUN = [
    (0, 4), (0, 5), (1, 4), (1, 5),
    (10, 4), (10, 5), (10, 6), (11, 3), (11, 7), (12, 2), (12, 8),
    (13, 2), (13, 8), (14, 5), (15, 3), (15, 7), (16, 4), (16, 5),
    (16, 6), (17, 5),
    (20, 2), (20, 3), (20, 4), (21, 2), (21, 3), (21, 4), (22, 1),
    (22, 5), (24, 0), (24, 1), (24, 5), (24, 6),
    (34, 2), (34, 3), (35, 2), (35, 3),
]


def _bits_of_n(n: int) -> List[int]:
    """Return list of bits of n (MSB first), at least 1 bit."""
    if n == 0:
        return [0]
    bits = []
    while n > 0:
        bits.append(n & 1)
        n >>= 1
    return bits[::-1]


def binary_encoding(n: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Binary encoding: place live cells in a row representing binary digits of N.
    Each '1' bit becomes a 3x3 block (to ensure stability); each '0' bit is
    left empty.  The row is centered in the grid.
    """
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    bits = _bits_of_n(n)
    num_bits = len(bits)
    # Place blocks centered vertically, starting near the left-center
    y_center = grid_h // 2
    x_start = max(2, (grid_w - num_bits * 4) // 2)

    for i, bit in enumerate(bits):
        if bit:
            x = x_start + i * 4
            # 3x3 block for a '1' bit
            for dy in range(3):
                for dx in range(3):
                    yy, xx = y_center - 1 + dy, x + dx
                    if 0 <= yy < grid_h and 0 <= xx < grid_w:
                        grid[yy, xx] = 1
    return grid


def prime_grid_encoding(n: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Prime-grid encoding: place a glider at position (p * spacing, 0)
    for each prime p <= N.  Gliders point diagonally downward-right.
    """
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    primes = primes_up_to(n)
    spacing = max(5, min(15, (grid_w - 20) // max(len(primes), 1)))
    y_start = 5

    for idx, p in enumerate(primes):
        offset_x = 5 + idx * spacing
        if offset_x + 3 >= grid_w:
            break
        # Place a glider; vary y slightly with p to create spatial signature
        offset_y = y_start + (p % 7) * 3
        if offset_y + 3 >= grid_h:
            offset_y = y_start
        for dx, dy in GLIDER:
            yy, xx = offset_y + dy, offset_x + dx
            if 0 <= yy < grid_h and 0 <= xx < grid_w:
                grid[yy, xx] = 1
    return grid


def factor_encoding(n: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Factor encoding: place pairs of gliders for each factor pair (a, b) of N.
    One glider at (a*3, 5), paired with another at (b*3, grid_h//2).
    """
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    pairs = factor_pairs(n)
    for a, b in pairs:
        # First glider in top half
        x1 = min(a * 3 + 3, grid_w - 5)
        y1 = 5
        for dx, dy in GLIDER:
            yy, xx = y1 + dy, x1 + dx
            if 0 <= yy < grid_h and 0 <= xx < grid_w:
                grid[yy, xx] = 1
        # Paired glider in bottom half
        x2 = min(b * 3 + 3, grid_w - 5)
        y2 = grid_h // 2
        for dx, dy in GLIDER:
            yy, xx = y2 + dy, x2 + dx
            if 0 <= yy < grid_h and 0 <= xx < grid_w:
                grid[yy, xx] = 1
    return grid


def mersenne_encoding(n: int, grid_w: int, grid_h: int) -> np.ndarray:
    """
    Mersenne encoding: if n = 2^p - 1 for some p, place p live cells
    in a diagonal line (a "Mersenne bar").  Otherwise, place the
    same number of cells as the total bit-count of n in a horizontal bar.
    """
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    if is_mersenne_number(n):
        p = int(math.log2(n + 1))
        # Diagonal bar of p cells
        cx, cy = grid_w // 2, grid_h // 2
        for i in range(p):
            xx = cx - p // 2 + i
            yy = cy - p // 2 + i
            # Make each cell a small 2x2 block for visual weight
            for ddx in range(2):
                for ddy in range(2):
                    xxx, yyy = xx + ddx, yy + ddy
                    if 0 <= yyy < grid_h and 0 <= xxx < grid_w:
                        grid[yyy, xxx] = 1
    else:
        # Place popcount(n) cells as a horizontal bar centered in the grid
        pop = bin(n).count('1')
        cx, cy = grid_w // 2, grid_h // 2
        for i in range(pop):
            xx = cx - pop // 2 + i * 2
            if 0 <= cy < grid_h and 0 <= xx < grid_w:
                grid[cy, xx] = 1
                if cy + 1 < grid_h:
                    grid[cy + 1, xx] = 1
    return grid


ENCODING_FUNCS = {
    'binary': binary_encoding,
    'prime_grid': prime_grid_encoding,
    'factor': factor_encoding,
    'mersenne': mersenne_encoding,
}


# ---------------------------------------------------------------------------
# 3. Feature Extraction
# ---------------------------------------------------------------------------

def spatial_entropy(grid: np.ndarray) -> float:
    """Shannon entropy of the grid treated as a Bernoulli source per cell."""
    p = grid.mean()
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def spatial_autocorrelation(grid: np.ndarray) -> float:
    """
    Moran's I — spatial autocorrelation of the binary grid.
    Positive = clustered, Negative = dispersed, ~0 = random.
    """
    flat = grid.flatten().astype(float)
    n = len(flat)
    mean_val = flat.mean()
    if n <= 1 or mean_val == 0 or mean_val == 1:
        return 0.0
    # Simple rook-contiguity weight matrix approximation:
    # shift grid right and down, correlate with original
    shifted_right = np.roll(grid, 1, axis=1)
    shifted_down = np.roll(grid, 1, axis=0)
    diff = flat - mean_val
    num = (np.sum((grid - mean_val) * (shifted_right - mean_val)) +
           np.sum((grid - mean_val) * (shifted_down - mean_val)))
    den = np.sum(diff ** 2)
    if den == 0:
        return 0.0
    # Normalize: 2*(rows+cols) adjacencies, each counted once
    W = 2 * grid.size  # approximate weight sum
    I = (n / W) * (num / den) if W > 0 else 0.0
    return float(I)


def center_of_mass(grid: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) center of mass of live cells."""
    total = grid.sum()
    if total == 0:
        return (0.0, 0.0)
    ys, xs = np.nonzero(grid)
    return (float(xs.mean()), float(ys.mean()))


def moment_of_inertia(grid: np.ndarray) -> float:
    """Spatial spread: mean squared distance from center of mass."""
    total = grid.sum()
    if total == 0:
        return 0.0
    ys, xs = np.nonzero(grid)
    cx, cy = xs.mean(), ys.mean()
    return float(np.mean((xs - cx) ** 2 + (ys - cy) ** 2))


def quadrant_densities(grid: np.ndarray) -> Dict[str, float]:
    """Fraction of live cells in each quadrant."""
    h, w = grid.shape
    mid_y, mid_x = h // 2, w // 2
    total = max(grid.sum(), 1)
    return {
        'q1_tl': float(grid[:mid_y, :mid_x].sum()) / total,
        'q2_tr': float(grid[:mid_y, mid_x:].sum()) / total,
        'q3_bl': float(grid[mid_y:, :mid_x].sum()) / total,
        'q4_br': float(grid[mid_y:, mid_x:].sum()) / total,
    }


def detect_oscillation(pop_series: np.ndarray,
                       min_period: int = 2,
                       max_period: int = 60) -> Tuple[int, float]:
    """
    Detect the dominant period of a population time series via FFT.

    Returns (period, strength) where strength is the normalized peak power.
    If no clear period is found, returns (0, 0.0).
    """
    series = pop_series.astype(float)
    if len(series) < 8:
        return (0, 0.0)
    # De-trend
    series = series - series.mean()
    if np.std(series) < 1e-10:
        return (1, 1.0)  # constant → period 1

    # FFT
    N = len(series)
    yf = np.abs(fft(series))[:N // 2]
    xf = fftfreq(N, d=1.0)[:N // 2]

    # Only look at frequencies corresponding to periods in [min_period, max_period]
    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    mask = (xf >= freq_min) & (xf <= freq_max) & (xf > 0)

    if not mask.any():
        return (0, 0.0)

    yf_masked = yf[mask]
    xf_masked = xf[mask]

    if yf_masked.max() < 1e-10:
        return (0, 0.0)

    best_idx = np.argmax(yf_masked)
    best_freq = xf_masked[best_idx]
    best_power = yf_masked[best_idx]
    total_power = yf_masked.sum() + 1e-15

    period = int(round(1.0 / best_freq)) if best_freq > 0 else 0
    strength = float(best_power / total_power)
    return (max(period, 1), strength)


def convergence_time(pop_series: np.ndarray,
                     window: int = 20,
                     threshold: float = 0.5) -> int:
    """
    Estimate the generation at which the population stabilizes.
    'Stabilizes' means the rolling std drops below *threshold* for
    the remainder of the simulation.
    """
    if len(pop_series) < window * 2:
        return len(pop_series)

    for start in range(0, len(pop_series) - window):
        seg = pop_series[start:start + window]
        if np.std(seg) < threshold:
            # Verify it stays stable
            if start + window < len(pop_series):
                remainder = pop_series[start + window:]
                if np.std(remainder) < threshold * 2:
                    return start
    return len(pop_series)


def count_still_lifes_oscillators(final_grid: np.ndarray,
                                   prev_grid: np.ndarray) -> Dict[str, int]:
    """
    Rough count of still lifes (unchanged cells) and oscillators
    (cells that changed between the last two generations).
    """
    unchanged = np.sum((final_grid == prev_grid) & (final_grid == 1))
    changed = np.sum((final_grid != prev_grid) & ((final_grid == 1) | (prev_grid == 1)))
    return {
        'still_life_cells': int(unchanged),
        'oscillator_cells': int(changed),
    }


def extract_features_from_history(history: List[np.ndarray]) -> Dict[str, Any]:
    """
    Given a list of grid snapshots, compute the full feature vector.

    Returns a flat dictionary of scalar features.
    """
    num_steps = len(history)
    pop_series = np.array([int(g.sum()) for g in history])

    features: Dict[str, Any] = {}

    # --- Population time series features ---
    features['pop_initial'] = float(pop_series[0])
    features['pop_final'] = float(pop_series[-1])
    features['pop_max'] = float(pop_series.max())
    features['pop_min'] = float(pop_series[1:].min() if len(pop_series) > 1 else 0)
    features['pop_mean'] = float(pop_series.mean())
    features['pop_std'] = float(pop_series.std())
    features['pop_skew'] = float(stats.skew(pop_series)) if num_steps > 2 else 0.0
    features['pop_kurtosis'] = float(stats.kurtosis(pop_series)) if num_steps > 3 else 0.0

    # --- Population delta ---
    if len(pop_series) > 1:
        deltas = np.diff(pop_series).astype(float)
        features['delta_mean'] = float(deltas.mean())
        features['delta_std'] = float(deltas.std())
        features['delta_max'] = float(deltas.max())
        features['delta_min'] = float(deltas.min())
        features['delta_abs_mean'] = float(np.abs(deltas).mean())
        # Proportion of positive, negative, zero deltas
        features['delta_frac_positive'] = float(np.mean(deltas > 0))
        features['delta_frac_negative'] = float(np.mean(deltas < 0))
        features['delta_frac_zero'] = float(np.mean(deltas == 0))
    else:
        for k in ['delta_mean', 'delta_std', 'delta_max', 'delta_min',
                   'delta_abs_mean', 'delta_frac_positive',
                   'delta_frac_negative', 'delta_frac_zero']:
            features[k] = 0.0

    # --- Population autocorrelation at lags 1..5 ---
    pop_centered = pop_series - pop_series.mean()
    pop_var = pop_centered.var()
    for lag in range(1, 6):
        if num_steps > lag and pop_var > 1e-10:
            features[f'pop_autocorr_lag{lag}'] = float(
                np.mean(pop_centered[:-lag] * pop_centered[lag:]) / pop_var
            )
        else:
            features[f'pop_autocorr_lag{lag}'] = 0.0

    # --- Oscillation detection ---
    period, strength = detect_oscillation(pop_series)
    features['osc_period'] = float(period)
    features['osc_strength'] = float(strength)

    # --- Convergence time ---
    features['convergence_time'] = float(convergence_time(pop_series))

    # --- Spatial features on final grid ---
    final = history[-1]
    features['spatial_entropy_final'] = spatial_entropy(final)
    features['spatial_autocorr_final'] = spatial_autocorrelation(final)
    features['moment_of_inertia_final'] = moment_of_inertia(final)
    cx, cy = center_of_mass(final)
    features['com_x_final'] = cx
    features['com_y_final'] = cy

    # --- Quadrant densities on final grid ---
    qd = quadrant_densities(final)
    features.update({f'quadrant_{k}': v for k, v in qd.items()})

    # --- Spatial entropy and autocorrelation over time (sampled) ---
    sample_gens = [1, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    sample_gens = [g for g in sample_gens if g < num_steps]
    for i, gen in enumerate(sample_gens):
        g = history[gen]
        features[f'entropy_gen{gen}'] = spatial_entropy(g)
        features[f'autocorr_gen{gen}'] = spatial_autocorrelation(g)
        features[f'moment_inertia_gen{gen}'] = moment_of_inertia(g)

    # --- Still lifes and oscillators in final state ---
    if num_steps >= 2:
        so = count_still_lifes_oscillators(history[-1], history[-2])
        features['still_life_cells'] = so['still_life_cells']
        features['oscillator_cells'] = so['oscillator_cells']
    else:
        features['still_life_cells'] = 0
        features['oscillator_cells'] = 0

    # --- Population at specific generations (key for "magic generation" search) ---
    key_gens = [1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23, 29, 30, 31, 37,
                41, 43, 47, 50, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                100, 127, 131, 137, 139, 149, 150, 151, 157, 163, 167,
                173, 179, 181, 191, 193, 197, 199, 200]
    for g in key_gens:
        if g < num_steps:
            features[f'pop_gen{g}'] = float(pop_series[g])
        # Don't add missing ones — we handle this in the ML pipeline

    # --- Alive-dead ratio in specific regions ---
    h, w = final.shape
    for name, region in [
        ('center', final[h//4:3*h//4, w//4:3*w//4]),
        ('top', final[:h//4, :]),
        ('bottom', final[3*h//4:, :]),
        ('left', final[:, :w//4]),
        ('right', final[:, 3*w//4:]),
    ]:
        total_cells = region.size
        alive = region.sum()
        features[f'alive_dead_ratio_{name}'] = float(alive / max(total_cells - alive, 1))

    return features


# ---------------------------------------------------------------------------
# 4. Primality Correlation Discovery
# ---------------------------------------------------------------------------

def run_gol_with_encoding(n: int, encoding_name: str,
                          generations: int = GOL_GENERATIONS,
                          grid_size: int = GRID_SIZE) -> Dict[str, Any]:
    """
    Run a single GoL simulation for integer n with the given encoding.
    Returns the full feature dictionary plus metadata.
    """
    encode_fn = ENCODING_FUNCS[encoding_name]
    initial_grid = encode_fn(n, grid_size, grid_size)

    sim = GoLSimulator(grid_size, grid_size)
    sim.grid = initial_grid.copy()

    history = sim.run(generations)
    features = extract_features_from_history(history)

    features['n'] = n
    features['encoding'] = encoding_name
    features['is_prime'] = int(is_prime(n))
    features['is_mersenne_number'] = int(is_mersenne_number(n))
    features['is_mersenne_prime'] = int(is_mersenne_prime(n))

    return features


def compute_correlations(feature_matrix: np.ndarray,
                         labels: np.ndarray,
                         feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    For each feature, compute:
      - Pearson correlation with the label
      - Spearman correlation with the label
      - Point-biserial correlation (equivalent to Pearson for binary label)
      - Mutual information
      - p-value (Bonferroni-corrected)
    """
    results = []
    n_features = feature_matrix.shape[1]
    n_tests = n_features  # for Bonferroni

    for j in range(n_features):
        col = feature_matrix[:, j]
        # Skip constant features
        if np.std(col) < 1e-10:
            results.append({
                'feature': feature_names[j],
                'pearson_r': 0.0, 'pearson_p': 1.0,
                'spearman_r': 0.0, 'spearman_p': 1.0,
                'mutual_info': 0.0,
                'bonferroni_p': 1.0,
                'significant_bonferroni': False,
            })
            continue

        # Pearson
        try:
            pr, pp = stats.pearsonr(col, labels)
        except Exception:
            pr, pp = 0.0, 1.0

        # Spearman
        try:
            sr, sp = stats.spearmanr(col, labels)
        except Exception:
            sr, sp = 0.0, 1.0

        # Mutual information (discretize feature into 10 bins)
        try:
            from sklearn.metrics import mutual_info_score
            col_binned = np.digitize(col, np.linspace(col.min(), col.max(), 10))
            mi = mutual_info_score(col_binned, labels)
        except Exception:
            mi = 0.0

        bonf_p = min(pp * n_tests, 1.0)

        results.append({
            'feature': feature_names[j],
            'pearson_r': float(pr),
            'pearson_p': float(pp),
            'spearman_r': float(sr),
            'spearman_p': float(sp),
            'mutual_info': float(mi),
            'bonferroni_p': float(bonf_p),
            'significant_bonferroni': bool(bonf_p < 0.05),
        })

    # Sort by absolute Pearson correlation (descending)
    results.sort(key=lambda d: abs(d['pearson_r']), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 5. Machine Learning Prime Predictor
# ---------------------------------------------------------------------------

def train_prime_predictor(feature_dict_list: List[Dict[str, Any]],
                          target: str = 'is_prime') -> Dict[str, Any]:
    """
    Train sklearn classifiers to predict *target* from GoL features.
    Returns model performance, feature importances, and cross-validation scores.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report

    # Build feature matrix
    # Use only numeric, non-metadata features
    meta_keys = {'n', 'encoding', 'is_prime', 'is_mersenne_number', 'is_mersenne_prime'}
    all_feature_keys = sorted(
        k for k in feature_dict_list[0].keys()
        if k not in meta_keys and isinstance(feature_dict_list[0].get(k), (int, float))
    )

    X = []
    y = []
    valid_keys = []

    # Find which features have valid (non-NaN) data across all samples
    for k in all_feature_keys:
        vals = [d.get(k, np.nan) for d in feature_dict_list]
        if all(isinstance(v, (int, float)) and not math.isnan(v) for v in vals):
            valid_keys.append(k)

    for d in feature_dict_list:
        row = [d.get(k, 0.0) for k in valid_keys]
        X.append(row)
        y.append(d.get(target, 0))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Impute any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if len(np.unique(y)) < 2:
        return {
            'error': f'Only one class present for target={target}. Cannot train classifier.',
            'n_samples': len(y),
            'class_distribution': dict(Counter(y)),
        }

    results: Dict[str, Any] = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'feature_names': valid_keys,
        'class_distribution': dict(Counter(y)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    rf.fit(X, y)

    results['random_forest'] = {
        'cv_accuracy_mean': float(rf_scores.mean()),
        'cv_accuracy_std': float(rf_scores.std()),
        'cv_scores': [float(s) for s in rf_scores],
        'feature_importances': {
            valid_keys[i]: float(rf.feature_importances_[i])
            for i in range(len(valid_keys))
        },
        'top_10_features': sorted(
            zip(valid_keys, rf.feature_importances_),
            key=lambda x: -x[1]
        )[:10],
    }

    # --- Logistic Regression ---
    lr_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    )
    lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring='accuracy')
    lr_pipe.fit(X, y)

    lr_coefs = lr_pipe.named_steps['logisticregression'].coef_[0]
    results['logistic_regression'] = {
        'cv_accuracy_mean': float(lr_scores.mean()),
        'cv_accuracy_std': float(lr_scores.std()),
        'cv_scores': [float(s) for s in lr_scores],
        'top_10_features_by_abs_coef': sorted(
            zip(valid_keys, np.abs(lr_coefs)),
            key=lambda x: -x[1]
        )[:10],
    }

    # --- Baseline: always predict majority class ---
    majority_acc = max(Counter(y).values()) / len(y)
    results['baseline_majority_accuracy'] = float(majority_acc)

    return results


# ---------------------------------------------------------------------------
# 6. Mersenne-Specific Analysis
# ---------------------------------------------------------------------------

def mersenne_analysis(generations: int = GOL_GENERATIONS,
                      grid_size: int = GRID_SIZE) -> Dict[str, Any]:
    """
    Special analysis for numbers of the form 2^p - 1.

    Compare GoL dynamics for Mersenne primes vs composite Mersenne numbers.
    Look for phase transitions, population correlations, etc.
    """
    results: Dict[str, Any] = {
        'mersenne_numbers': [],
        'population_at_key_gens': {},
        'phase_transition_analysis': {},
    }

    mersenne_data = []

    for p in MERSENNE_P_RANGE:
        n = 2 ** p - 1
        is_mp = p in KNOWN_MERSENNE_PRIMES_P
        label = 'mersenne_prime' if is_mp else 'mersenne_composite'

        print(f"  Mersenne p={p:2d} → N={n:>10d}  ({label})")

        # Use mersenne encoding
        features = run_gol_with_encoding(n, 'mersenne', generations, grid_size)
        features['p'] = p
        features['mersenne_label'] = label
        mersenne_data.append(features)

    results['mersenne_numbers'] = mersenne_data

    # --- Compare population time series at key generations ---
    key_gens = [1, 5, 10, 30, 50, 100, 150, 200]
    pop_at_gen: Dict[int, Dict[str, List[float]]] = {}

    for g in key_gens:
        pop_at_gen[g] = {'mersenne_prime': [], 'mersenne_composite': []}
        for d in mersenne_data:
            val = d.get(f'pop_gen{g}', None)
            if val is not None:
                pop_at_gen[g][d['mersenne_label']].append(float(val))

    results['population_at_key_gens'] = {}
    for g in key_gens:
        prime_vals = pop_at_gen[g].get('mersenne_prime', [])
        comp_vals = pop_at_gen[g].get('mersenne_composite', [])
        entry: Dict[str, Any] = {
            'prime_mean': float(np.mean(prime_vals)) if prime_vals else None,
            'composite_mean': float(np.mean(comp_vals)) if comp_vals else None,
        }
        # t-test if we have enough data
        if len(prime_vals) >= 3 and len(comp_vals) >= 3:
            t_stat, p_val = stats.ttest_ind(prime_vals, comp_vals, equal_var=False)
            entry['t_statistic'] = float(t_stat)
            entry['p_value'] = float(p_val)
            entry['significant'] = p_val < 0.05
        results['population_at_key_gens'][g] = entry

    # --- Phase transition analysis ---
    # Does the convergence time or oscillation period show a
    # discontinuity at Mersenne primes vs composites?
    prime_convergence = [d['convergence_time'] for d in mersenne_data
                         if d['mersenne_label'] == 'mersenne_prime']
    comp_convergence = [d['convergence_time'] for d in mersenne_data
                        if d['mersenne_label'] == 'mersenne_composite']

    phase_results: Dict[str, Any] = {}
    if prime_convergence and comp_convergence:
        phase_results['convergence_time_prime_mean'] = float(np.mean(prime_convergence))
        phase_results['convergence_time_composite_mean'] = float(np.mean(comp_convergence))
        if len(prime_convergence) >= 2 and len(comp_convergence) >= 2:
            t, p = stats.ttest_ind(prime_convergence, comp_convergence, equal_var=False)
            phase_results['convergence_t_test'] = {'t': float(t), 'p': float(p)}

    prime_osc = [d['osc_period'] for d in mersenne_data
                 if d['mersenne_label'] == 'mersenne_prime']
    comp_osc = [d['osc_period'] for d in mersenne_data
                if d['mersenne_label'] == 'mersenne_composite']
    if prime_osc and comp_osc:
        phase_results['osc_period_prime_mean'] = float(np.mean(prime_osc))
        phase_results['osc_period_composite_mean'] = float(np.mean(comp_osc))

    results['phase_transition_analysis'] = phase_results

    return results


# ---------------------------------------------------------------------------
# 7. Novel Discovery Engine
# ---------------------------------------------------------------------------

def discover_magic_generations(feature_dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Systematically search for:
    - Which generation's population is most correlated with primality?
    - Which spatial region's density is most predictive?
    - Is there a "magic generation" where prime/composite distinction is maximized?

    Returns discovery report with p-values and confidence intervals.
    """
    results: Dict[str, Any] = {
        'generation_correlations': [],
        'region_predictiveness': [],
        'magic_generation': None,
        'magic_generation_confidence': None,
    }

    # Extract population-at-generation features
    pop_gen_keys = sorted(
        k for k in feature_dict_list[0].keys()
        if k.startswith('pop_gen') and isinstance(feature_dict_list[0].get(k), (int, float))
    )

    labels = np.array([d['is_prime'] for d in feature_dict_list], dtype=int)

    # --- Which generation's population is most correlated with primality? ---
    best_gen = None
    best_abs_r = 0.0
    best_p = 1.0

    for key in pop_gen_keys:
        vals = np.array([d.get(key, np.nan) for d in feature_dict_list], dtype=float)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 10:
            continue

        v = vals[valid_mask]
        l = labels[valid_mask]
        if np.std(v) < 1e-10:
            continue

        try:
            r, p = stats.pointbiserialr(l, v)
        except Exception:
            continue

        gen_num = int(key.replace('pop_gen', ''))
        entry = {
            'generation': gen_num,
            'point_biserial_r': float(r),
            'p_value': float(p),
            'n_valid': int(valid_mask.sum()),
        }
        results['generation_correlations'].append(entry)

        if abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_gen = gen_num
            best_p = p

    # Bonferroni correction
    n_tests = len(results['generation_correlations'])
    for entry in results['generation_correlations']:
        entry['bonferroni_p'] = min(entry['p_value'] * n_tests, 1.0)
        entry['significant_bonferroni'] = entry['bonferroni_p'] < 0.05

    results['generation_correlations'].sort(
        key=lambda d: abs(d['point_biserial_r']), reverse=True
    )

    if best_gen is not None:
        # Bootstrap confidence interval for the best correlation
        n_boot = 2000
        rng = np.random.RandomState(RANDOM_SEED)
        key = f'pop_gen{best_gen}'
        vals_all = np.array([d.get(key, np.nan) for d in feature_dict_list], dtype=float)
        valid_mask = ~np.isnan(vals_all)
        v = vals_all[valid_mask]
        l = labels[valid_mask]

        # Check that feature is not constant
        if np.std(v) < 1e-10:
            results['magic_generation'] = best_gen
            results['magic_generation_confidence'] = {
                'r': float(best_abs_r),
                'p_value': float(best_p),
                'bonferroni_p': float(min(best_p * n_tests, 1.0)),
                'ci_95': (0.0, 0.0),
                'bootstrap_samples': 0,
                'note': 'Feature is constant; CI undefined',
            }
        else:
            boot_rs = []
            for _ in range(n_boot):
                idx = rng.choice(len(v), size=len(v), replace=True)
                bv, bl = v[idx], l[idx]
                if len(np.unique(bl)) < 2:
                    continue
                try:
                    br, _ = stats.pointbiserialr(bl, bv)
                    if not (math.isnan(br) or math.isinf(br)):
                        boot_rs.append(br)
                except Exception:
                    continue

            if boot_rs:
                ci_low = float(np.percentile(boot_rs, 2.5))
                ci_high = float(np.percentile(boot_rs, 97.5))
            else:
                ci_low, ci_high = 0.0, 0.0

            results['magic_generation'] = best_gen
            results['magic_generation_confidence'] = {
                'r': float(best_abs_r),
                'p_value': float(best_p),
                'bonferroni_p': float(min(best_p * n_tests, 1.0)),
                'ci_95': (ci_low, ci_high),
                'bootstrap_samples': len(boot_rs),
            }

    # --- Which spatial region's density is most predictive? ---
    region_keys = [k for k in feature_dict_list[0].keys()
                   if k.startswith('alive_dead_ratio_') or k.startswith('quadrant_')]

    for key in region_keys:
        vals = np.array([d.get(key, np.nan) for d in feature_dict_list], dtype=float)
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 10:
            continue
        v = vals[valid_mask]
        l = labels[valid_mask]
        if np.std(v) < 1e-10:
            continue
        try:
            r, p = stats.pointbiserialr(l, v)
            results['region_predictiveness'].append({
                'region': key,
                'point_biserial_r': float(r),
                'p_value': float(p),
            })
        except Exception:
            continue

    results['region_predictiveness'].sort(
        key=lambda d: abs(d['point_biserial_r']), reverse=True
    )

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_massive_discovery(n_range: range = N_RANGE,
                          generations: int = GOL_GENERATIONS,
                          grid_size: int = GRID_SIZE,
                          encodings: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run the full discovery pipeline:

    1. Run GoL simulations for all N in n_range with each encoding
    2. Extract features
    3. Compute correlations
    4. Train ML predictors
    5. Run Mersenne-specific analysis
    6. Search for magic generations
    7. Compile and return results
    """
    if encodings is None:
        encodings = ['binary', 'prime_grid', 'factor', 'mersenne']

    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()
    all_features: Dict[str, List[Dict[str, Any]]] = {}
    all_results: Dict[str, Any] = {}

    total_sims = len(n_range) * len(encodings)
    print(f"{'='*70}")
    print(f"GoL-Prime Discovery Engine")
    print(f"{'='*70}")
    print(f"Running {total_sims} simulations: N∈[{n_range.start}, {n_range.stop}), "
          f"{len(encodings)} encodings, {generations} generations each")
    print(f"Grid size: {grid_size}x{grid_size}")
    print()

    # --- Phase 1: Run simulations ---
    for enc in encodings:
        print(f"\n--- Encoding: {enc} ---")
        enc_features = []
        sim_count = 0

        for n in n_range:
            sim_count += 1
            if sim_count % 50 == 0 or sim_count <= 3:
                elapsed = time.time() - t0
                rate = sim_count / max(elapsed, 0.01)
                print(f"  [{enc}] N={n:4d}  "
                      f"({sim_count}/{len(n_range)})  "
                      f"[{rate:.1f} sims/sec]")

            try:
                feat = run_gol_with_encoding(n, enc, generations, grid_size)
                enc_features.append(feat)
            except Exception as e:
                print(f"  WARNING: N={n}, encoding={enc} failed: {e}")
                continue

        all_features[enc] = enc_features
        print(f"  Completed {len(enc_features)} simulations for encoding '{enc}'")

    elapsed_sim = time.time() - t0
    print(f"\nAll simulations complete in {elapsed_sim:.1f}s")

    # --- Phase 2: Correlation analysis ---
    print(f"\n{'='*70}")
    print("Phase 2: Correlation Analysis")
    print(f"{'='*70}")

    all_correlations: Dict[str, Dict[str, List[Dict]]] = {}

    for enc in encodings:
        all_correlations[enc] = {}
        feat_list = all_features[enc]
        if not feat_list:
            continue

        meta_keys = {'n', 'encoding', 'is_prime', 'is_mersenne_number', 'is_mersenne_prime'}
        numeric_keys = sorted(
            k for k in feat_list[0].keys()
            if k not in meta_keys and isinstance(feat_list[0].get(k), (int, float))
        )

        for target_name in ['is_prime', 'is_mersenne_number', 'is_mersenne_prime']:
            labels = np.array([d[target_name] for d in feat_list], dtype=int)
            if labels.sum() < 3 or (len(labels) - labels.sum()) < 3:
                all_correlations[enc][target_name] = []
                continue

            # Build feature matrix
            X = []
            valid_feat_names = []
            for k in numeric_keys:
                vals = [d.get(k, np.nan) for d in feat_list]
                if all(isinstance(v, (int, float)) and not math.isnan(v) for v in vals):
                    X.append([float(v) for v in vals])
                    valid_feat_names.append(k)

            if not X:
                continue

            X = np.array(X, dtype=float).T  # shape: (n_samples, n_features)
            corrs = compute_correlations(X, labels, valid_feat_names)
            all_correlations[enc][target_name] = corrs

            # Print top correlations
            top5 = corrs[:5]
            print(f"\n  [{enc}] Top correlations with {target_name}:")
            for c in top5:
                sig = "***" if c['significant_bonferroni'] else ""
                print(f"    {c['feature']:40s}  r={c['pearson_r']:+.4f}  "
                      f"p={c['pearson_p']:.4e}  Bonf={c['bonferroni_p']:.4e}  {sig}")

    all_results['correlations'] = all_correlations

    # --- Phase 3: ML Prime Prediction ---
    print(f"\n{'='*70}")
    print("Phase 3: Machine Learning Prime Prediction")
    print(f"{'='*70}")

    ml_results: Dict[str, Any] = {}
    for enc in encodings:
        feat_list = all_features[enc]
        if len(feat_list) < 20:
            continue

        print(f"\n  Training classifiers on encoding '{enc}'...")
        ml_res = train_prime_predictor(feat_list, target='is_prime')
        ml_results[enc] = ml_res

        if 'error' in ml_res:
            print(f"    ERROR: {ml_res['error']}")
            continue

        print(f"    Random Forest  CV accuracy: "
              f"{ml_res['random_forest']['cv_accuracy_mean']:.4f} "
              f"± {ml_res['random_forest']['cv_accuracy_std']:.4f}")
        print(f"    Logistic Regr  CV accuracy: "
              f"{ml_res['logistic_regression']['cv_accuracy_mean']:.4f} "
              f"± {ml_res['logistic_regression']['cv_accuracy_std']:.4f}")
        print(f"    Baseline (majority): {ml_res['baseline_majority_accuracy']:.4f}")
        print(f"    Top 5 RF features:")
        for fname, imp in ml_res['random_forest']['top_10_features'][:5]:
            print(f"      {fname:40s}  importance={imp:.4f}")

    all_results['ml'] = ml_results

    # --- Phase 4: Mersenne Analysis ---
    print(f"\n{'='*70}")
    print("Phase 4: Mersenne-Specific Analysis")
    print(f"{'='*70}")

    mersenne_res = mersenne_analysis(generations, grid_size)
    all_results['mersenne'] = mersenne_res

    # Print Mersenne summary
    print("\n  Population at key generations (Mersenne primes vs composites):")
    for gen, data in mersenne_res['population_at_key_gens'].items():
        pm = data.get('prime_mean', 'N/A')
        cm = data.get('composite_mean', 'N/A')
        sig = ""
        if data.get('significant', False):
            sig = " *** SIGNIFICANT"
        p_val = data.get('p_value', 'N/A')
        print(f"    Gen {gen:4d}: prime_mean={pm}, comp_mean={cm}, "
              f"p={p_val}{sig}")

    # --- Phase 5: Discovery Engine ---
    print(f"\n{'='*70}")
    print("Phase 5: Novel Discovery Engine — Searching for Magic Generations")
    print(f"{'='*70}")

    discovery_results: Dict[str, Any] = {}
    for enc in encodings:
        feat_list = all_features[enc]
        if len(feat_list) < 20:
            continue

        print(f"\n  Encoding: {enc}")
        disc = discover_magic_generations(feat_list)
        discovery_results[enc] = disc

        mg = disc.get('magic_generation')
        if mg is not None:
            conf = disc['magic_generation_confidence']
            print(f"    ★ Magic generation: {mg}")
            print(f"      r = {conf['r']:.4f}, "
                  f"p = {conf['p_value']:.4e}, "
                  f"Bonferroni p = {conf['bonferroni_p']:.4e}")
            print(f"      95% CI: ({conf['ci_95'][0]:.4f}, {conf['ci_95'][1]:.4f})")
            if conf['bonferroni_p'] >= 0.05:
                print(f"      ⚠ NOT significant after Bonferroni correction")
        else:
            print(f"    No magic generation found (insufficient data)")

        # Top region predictiveness
        if disc['region_predictiveness']:
            top_region = disc['region_predictiveness'][0]
            print(f"    Most predictive region: {top_region['region']}  "
                  f"r={top_region['point_biserial_r']:.4f}  "
                  f"p={top_region['p_value']:.4e}")

    all_results['discovery'] = discovery_results

    # --- Save results ---
    total_time = time.time() - t0
    all_results['metadata'] = {
        'n_range': [n_range.start, n_range.stop],
        'generations': generations,
        'grid_size': grid_size,
        'encodings': encodings,
        'total_simulations': total_sims,
        'total_time_seconds': total_time,
    }

    results_path = os.path.join(RESULTS_DIR, "gol_prime_discovery_results.json")

    # Custom JSON encoder that handles numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if math.isnan(obj) if isinstance(obj, float) else False:
                return None
            if math.isinf(obj) if isinstance(obj, float) else False:
                return None
            return super().default(obj)

    # Convert any remaining problematic types recursively
    def sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    sanitized = sanitize_for_json(all_results)
    with open(results_path, 'w') as f:
        json.dump(sanitized, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Print Discovery Report ---
    print_discovery_report(all_results)

    return all_results


def print_discovery_report(results: Dict[str, Any]) -> None:
    """Print the comprehensive Discovery Report."""
    print(f"\n{'='*70}")
    print("DISCOVERY REPORT — GoL-Prime Correlation Analysis")
    print(f"{'='*70}")

    meta = results.get('metadata', {})
    print(f"\nSimulations: {meta.get('total_simulations', 'N/A')}")
    print(f"Total time:  {meta.get('total_time_seconds', 0):.1f}s")
    print(f"N range:     [{meta.get('n_range', ['?','?'])[0]}, {meta.get('n_range', ['?','?'])[1]})")
    print(f"Generations: {meta.get('generations', 'N/A')}")
    print(f"Grid size:   {meta.get('grid_size', 'N/A')}")

    # --- Correlation findings ---
    print(f"\n{'─'*70}")
    print("1. CORRELATION FINDINGS")
    print(f"{'─'*70}")

    any_significant = False
    for enc, targets in results.get('correlations', {}).items():
        for target, corrs in targets.items():
            sig_corrs = [c for c in corrs if c.get('significant_bonferroni', False)]
            if sig_corrs:
                any_significant = True
                print(f"\n  [{enc} → {target}] {len(sig_corrs)} feature(s) significant "
                      f"after Bonferroni correction:")
                for c in sig_corrs[:5]:
                    print(f"    • {c['feature']:40s}  r={c['pearson_r']:+.4f}  "
                          f"Bonferroni p={c['bonferroni_p']:.4e}")
            else:
                top = corrs[:1] if corrs else []
                if top:
                    print(f"  [{enc} → {target}] No features survive Bonferroni. "
                          f"Best: {top[0]['feature']} r={top[0]['pearson_r']:+.4f} "
                          f"p={top[0]['pearson_p']:.4e}")

    if not any_significant:
        print("\n  ⚠  NO features survive Bonferroni correction across any encoding.")
        print("  This suggests that any apparent correlations are likely due to")
        print("  multiple testing rather than genuine GoL-prime relationships.")

    # --- ML findings ---
    print(f"\n{'─'*70}")
    print("2. MACHINE LEARNING FINDINGS")
    print(f"{'─'*70}")

    for enc, ml in results.get('ml', {}).items():
        if 'error' in ml:
            print(f"\n  [{enc}] {ml['error']}")
            continue

        rf_acc = ml['random_forest']['cv_accuracy_mean']
        lr_acc = ml['logistic_regression']['cv_accuracy_mean']
        baseline = ml['baseline_majority_accuracy']
        improvement = max(rf_acc, lr_acc) - baseline

        print(f"\n  [{enc}]")
        print(f"    Baseline (majority class): {baseline:.4f}")
        print(f"    Random Forest accuracy:    {rf_acc:.4f} "
              f"(+{rf_acc - baseline:.4f} over baseline)")
        print(f"    Logistic Regression:       {lr_acc:.4f} "
              f"(+{lr_acc - baseline:.4f} over baseline)")

        if improvement < 0.02:
            print(f"    → Marginal improvement ({improvement:.4f}). "
                  f"GoL features weakly predict primality.")
        elif improvement < 0.10:
            print(f"    → Moderate improvement ({improvement:.4f}). "
                  f"Some GoL features carry weak primality signal.")
        else:
            print(f"    → Notable improvement ({improvement:.4f}). "
                  f"GoL features encode meaningful primality signal!")

        print(f"    Top RF features:")
        for fname, imp in ml['random_forest']['top_10_features'][:5]:
            print(f"      {fname:40s}  importance={imp:.4f}")

    # --- Mersenne findings ---
    print(f"\n{'─'*70}")
    print("3. MERSENNE-SPECIFIC FINDINGS")
    print(f"{'─'*70}")

    mersenne = results.get('mersenne', {})
    phase = mersenne.get('phase_transition_analysis', {})
    pop_gens = mersenne.get('population_at_key_gens', {})

    mersenne_sig = False
    for gen, data in pop_gens.items():
        if data.get('significant', False):
            mersenne_sig = True
            print(f"  ★ Generation {gen}: population differs significantly "
                  f"between Mersenne primes and composites (p={data['p_value']:.4e})")

    if not mersenne_sig:
        print("  No significant population differences between Mersenne primes")
        print("  and Mersenne composites at any key generation (p > 0.05).")

    conv_prime = phase.get('convergence_time_prime_mean')
    conv_comp = phase.get('convergence_time_composite_mean')
    if conv_prime is not None and conv_comp is not None:
        diff = abs(conv_prime - conv_comp)
        print(f"\n  Convergence time: Mersenne prime mean = {conv_prime:.1f}, "
              f"composite mean = {conv_comp:.1f}, diff = {diff:.1f}")
        conv_test = phase.get('convergence_t_test', {})
        if conv_test:
            print(f"  t-test: t={conv_test['t']:.3f}, p={conv_test['p']:.4e}")

    # --- Magic generation findings ---
    print(f"\n{'─'*70}")
    print("4. MAGIC GENERATION FINDINGS")
    print(f"{'─'*70}")

    for enc, disc in results.get('discovery', {}).items():
        mg = disc.get('magic_generation')
        if mg is not None:
            conf = disc['magic_generation_confidence']
            sig_str = ("SIGNIFICANT" if conf['bonferroni_p'] < 0.05
                       else "NOT significant after Bonferroni correction")
            print(f"  [{enc}] Magic generation = {mg}:  "
                  f"r={conf['r']:.4f}, Bonferroni p={conf['bonferroni_p']:.4e}  "
                  f"[{sig_str}]")
            print(f"           95% CI: ({conf['ci_95'][0]:.4f}, {conf['ci_95'][1]:.4f})")
        else:
            print(f"  [{enc}] No magic generation identified")

    # --- Honest assessment ---
    print(f"\n{'─'*70}")
    print("5. HONEST ASSESSMENT")
    print(f"{'─'*70}")

    # Gather all ML improvements
    improvements = []
    for enc, ml in results.get('ml', {}).items():
        if 'error' in ml:
            continue
        rf_acc = ml['random_forest']['cv_accuracy_mean']
        baseline = ml['baseline_majority_accuracy']
        improvements.append(rf_acc - baseline)

    max_improvement = max(improvements) if improvements else 0

    any_sig_corr = any(
        c.get('significant_bonferroni', False)
        for enc, targets in results.get('correlations', {}).items()
        for target, corrs in targets.items()
        for c in corrs
    )

    if not any_sig_corr and max_improvement < 0.05:
        print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  NEGATIVE RESULT (with nuance)                                  │
  │                                                                  │
  │  After exhaustive simulation of GoL dynamics with number-encoded │
  │  initial conditions, we find NO statistically significant        │
  │  correlation between GoL evolution features and primality that   │
  │  survives Bonferroni correction.                                 │
  │                                                                  │
  │  The ML classifiers show at most marginal improvement over the   │
  │  majority-class baseline, confirming that GoL population         │
  │  dynamics do not naturally encode primality information when     │
  │  using these encoding schemes.                                   │
  │                                                                  │
  │  This is a genuine negative result and is scientifically         │
  │  valuable: it establishes that simple number-encoding schemes    │
  │  do not create emergent primality signatures in GoL dynamics.    │
  │                                                                  │
  │  Why? Conway's Game of Life is computationally universal but     │
  │  the specific encodings tested here do not leverage GoL's        │
  │  computational machinery in a way that would reflect arithmetic  │
  │  properties back into population statistics. The Primer pattern  │
  │  (Hickerson 1991) works by implementing the Sieve of Eratosthenes│
  │  using structured logic gates (glider guns, reflectors, etc.) —  │
  │  not by encoding numbers as cell patterns and reading off        │
  │  statistical features.                                           │
  │                                                                  │
  │  Future directions:                                              │
  │  • Use structured computing elements (glider guns, logic gates)  │
  │  • Look at specific structured patterns, not statistical moments │
  │  • Consider 1D cellular automata (Rule 90, Rule 30) where the   │
  │    connection to number theory is more direct                     │
  │  • Investigate whether GoL Turing machines computing primality   │
  │    exhibit measurable phase transitions in their dynamics        │
  └─────────────────────────────────────────────────────────────────┘
""")
    elif any_sig_corr:
        print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  POSITIVE RESULT — CORRELATION DETECTED                         │
  │                                                                  │
  │  Some features survive Bonferroni correction, suggesting a       │
  │  genuine (though possibly small) correlation between GoL         │
  │  dynamics and primality under specific encodings.                │
  │                                                                  │
  │  HOWEVER: correlation ≠ causation. The correlation likely        │
  │  arises because the encoding schemes create systematically       │
  │  different initial conditions for primes vs composites (e.g.,    │
  │  prime_grid encoding places more gliders for primes), not       │
  │  because GoL dynamics "detect" primality.                        │
  │                                                                  │
  │  This is an encoding artifact, not an emergent property of GoL.  │
  └─────────────────────────────────────────────────────────────────┘
""")
    else:
        print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  MIXED RESULT                                                    │
  │                                                                  │
  │  Some weak correlations exist but do not survive rigorous        │
  │  Bonferroni correction. The ML classifiers show small but        │
  │  consistent improvements over baseline.                          │
  │                                                                  │
  │  This likely reflects subtle encoding biases rather than genuine │
  │  emergent GoL-primality connections. See detailed analysis above.│
  └─────────────────────────────────────────────────────────────────┘
""")

    print(f"{'='*70}")
    print("End of Discovery Report")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow custom parameters via command line
    n_start = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_end = int(sys.argv[2]) if len(sys.argv) > 2 else 501
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else GOL_GENERATIONS
    gs = int(sys.argv[4]) if len(sys.argv) > 4 else GRID_SIZE

    print(f"Parameters: N=[{n_start}, {n_end}), generations={gens}, grid={gs}")

    results = run_massive_discovery(
        n_range=range(n_start, n_end),
        generations=gens,
        grid_size=gs,
    )
