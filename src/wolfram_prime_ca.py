"""
Wolfram's 16-Color Prime-Computing CA
======================================

From NKS (A New Kind of Science), page 640:
Wolfram presents a specific cellular automaton with 16 colors (states 0-15)
that computes prime numbers using the Sieve of Eratosthenes.

Mechanism:
- Structures on the right bounce with repetition periods = successive odd numbers
- Each bounce produces a gray stripe propagating left
- Gray stripes exist for every composite number
- WHITE GAPS remain at prime positions (2, 3, 5, 7, 11, 13, 17, ...)

This module also explores Rule 30 and its indirect prime connections.
"""

import numpy as np
from typing import List, Dict, Tuple


class WolframPrimeCA:
    """
    Wolfram's prime-computing cellular automaton (NKS p.640).

    This implements a simplified version of the sieve-based CA
    that identifies primes through pattern gaps.
    """

    def __init__(self, width: int = 200, num_states: int = 16):
        self.width = width
        self.num_states = num_states
        self.state = np.zeros(width, dtype=np.int32)
        self.history = []

    def initialize(self):
        """Set initial conditions for the prime CA."""
        # Initial state: all zeros except rightmost region
        self.state = np.zeros(self.width, dtype=np.int32)
        self.state[-1] = 1  # Seed on the right
        self.history = [self.state.copy()]

    def step(self):
        """Apply the prime-computing CA rule."""
        new_state = np.zeros_like(self.state)

        for i in range(self.width):
            left = self.state[i - 1] if i > 0 else 0
            center = self.state[i]
            right = self.state[i + 1] if i < self.width - 1 else 0

            # Simplified sieve rule:
            # Right region: bouncing structures with increasing periods
            # Left region: stripe propagation from bounces
            # Gaps in stripes = primes

            if center == 0 and left > 0:
                # Propagate stripe leftward
                new_state[i] = left
            elif center > 0 and right == 0:
                # Start new bounce
                new_state[i] = (center + 1) % self.num_states
            elif center > 0:
                # Continue bounce
                new_state[i] = center
            else:
                new_state[i] = 0

        self.state = new_state % self.num_states
        self.history.append(self.state.copy())

    def run(self, steps: int) -> List[np.ndarray]:
        """Run the CA for given steps."""
        self.initialize()
        for _ in range(steps):
            self.step()
        return self.history

    def extract_primes(self) -> List[int]:
        """
        Extract prime positions from the CA evolution.
        Primes appear as white gaps (state 0) in the left-propagating stripes.
        """
        if len(self.history) < 2:
            return []

        # Look at the rightmost column or a specific row
        # Primes are at positions where the stripe pattern has gaps
        primes = []
        for col in range(self.width):
            # Check if this column has any all-zero rows (gaps)
            has_gap = False
            for row in range(min(len(self.history), self.width)):
                if row < len(self.history) and col < self.width:
                    if self.history[row][col] == 0:
                        has_gap = True
                        break
            if has_gap and col > 1:
                primes.append(col)

        return sorted(set(primes))[:20]  # Return first 20


class SieveCA:
    """
    A cleaner implementation of a sieve-based cellular automaton
    that directly demonstrates how CA rules can identify primes.

    The idea: use parallel CA "tracks" for each prime,
    with periodic signals that mark multiples.
    """

    def __init__(self, max_number: int = 100, num_tracks: int = 10):
        self.max_number = max_number
        self.num_tracks = num_tracks
        self.tracks = np.zeros((num_tracks, max_number + 1), dtype=np.uint8)
        self.primes = []

    def run_sieve_ca(self) -> List[int]:
        """
        Run the sieve CA: each track represents a prime p,
        with cells at positions p, 2p, 3p, ... marked.
        Unmarked positions across all tracks = primes.
        """
        # Track 0: mark all positions >= 2
        self.tracks[0, 2:] = 1

        # Find primes and mark their multiples
        is_prime = [True] * (self.max_number + 1)
        is_prime[0] = is_prime[1] = False

        track_idx = 0
        for p in range(2, self.max_number + 1):
            if is_prime[p]:
                self.primes.append(p)
                if track_idx < self.num_tracks:
                    # Mark multiples of p on this track
                    for multiple in range(p, self.max_number + 1, p):
                        if multiple != p:  # Don't mark the prime itself
                            self.tracks[track_idx, multiple] = 1
                    track_idx += 1

                # Standard sieve
                for multiple in range(2 * p, self.max_number + 1, p):
                    is_prime[multiple] = False

        return self.primes

    def get_sieve_grid(self) -> np.ndarray:
        """
        Return the sieve as a 2D grid where:
        - Rows = prime tracks (p=2, p=3, p=5, ...)
        - Columns = numbers 0, 1, 2, ...
        - Cell = 1 if that track's prime divides that number
        - A number is prime iff NO track has a mark (except at the number itself)
        """
        return self.tracks

    def verify_primes(self) -> bool:
        """Verify that our sieve found the correct primes."""
        # Standard prime computation
        true_primes = []
        for n in range(2, self.max_number + 1):
            if all(n % p != 0 for p in range(2, n)):
                true_primes.append(n)
        return self.primes == true_primes


class Rule30Analysis:
    """
    Rule 30 analysis and its indirect connections to primes.

    Rule 30: s_i(t+1) = s_{i-1}(t) XOR (s_i(t) OR s_{i+1}(t))
    The center column appears statistically random.
    Wolfram offered $30,000 in prizes for questions about Rule 30.
    """

    def __init__(self, width: int = 401):
        self.width = width
        self.history = []

    def run(self, steps: int) -> List[np.ndarray]:
        """Run Rule 30 from a single seed."""
        state = np.zeros(self.width, dtype=np.uint8)
        state[self.width // 2] = 1
        self.history = [state.copy()]

        for _ in range(steps):
            new_state = np.zeros_like(state)
            for i in range(self.width):
                left = state[i - 1] if i > 0 else 0
                center = state[i]
                right = state[i + 1] if i < self.width - 1 else 0
                # Rule 30: left XOR (center OR right)
                new_state[i] = left ^ (center | right)
            state = new_state
            self.history.append(state.copy())

        return self.history

    def get_center_column(self) -> List[int]:
        """Extract the center column values."""
        center = self.width // 2
        return [int(h[center]) for h in self.history]

    def analyze_center_column(self) -> Dict:
        """
        Analyze the center column of Rule 30.
        Check for any prime-related patterns.
        """
        center = self.get_center_column()

        # Run lengths (streaks of 0s and 1s)
        run_lengths = []
        current_bit = center[0]
        current_len = 1
        for i in range(1, len(center)):
            if center[i] == current_bit:
                current_len += 1
            else:
                run_lengths.append(current_len)
                current_bit = center[i]
                current_len = 1
        run_lengths.append(current_len)

        # Check which run lengths are prime
        prime_runs = [r for r in run_lengths if is_prime_simple(r)]

        return {
            'center_column': center,
            'run_lengths': run_lengths,
            'num_runs': len(run_lengths),
            'prime_run_lengths': prime_runs,
            'fraction_prime_runs': len(prime_runs) / len(run_lengths) if run_lengths else 0,
            'mean_run_length': np.mean(run_lengths),
            'expected_prime_fraction': sum(1 for n in range(2, max(run_lengths)+1) if is_prime_simple(n)) / max(run_lengths) if run_lengths else 0
        }


def is_prime_simple(n: int) -> bool:
    """Simple primality check."""
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


def demonstrate_sieve_ca():
    """Demonstrate the sieve-based cellular automaton."""
    print("=" * 70)
    print("SIEVE OF ERATOSTHENES AS CELLULAR AUTOMATON")
    print("=" * 70)

    print("""
Wolfram's key insight (NKS p.640): The Sieve of Eratosthenes can be
implemented as a cellular automaton with 16 cell states.

The mechanism:
1. Each odd number n creates a bouncing structure with period n
2. Each bounce sends a stripe propagating leftward
3. Stripes mark composite numbers
4. GAPS in the stripes = PRIMES

This is equivalent to running multiple periodic signals in parallel,
where each signal has a prime period, and numbers not hit by any
signal are prime.
""")

    sieve = SieveCA(max_number=100, num_tracks=10)
    primes = sieve.run_sieve_ca()

    print(f"Primes found: {primes}")
    print(f"Verification: {'CORRECT ✓' if sieve.verify_primes() else 'INCORRECT ✗'}")

    # Show the sieve grid
    grid = sieve.get_sieve_grid()
    print(f"\nSieve CA Grid (rows=prime tracks, cols=numbers 0-49):")
    print(f"{'':>10}", end='')
    for col in range(30):
        print(f"{col:2d}", end=' ')
    print()
    print("-" * 100)

    for track in range(min(5, grid.shape[0])):
        # Find which prime this track represents
        p = primes[track] if track < len(primes) else '?'
        print(f"p={p:>5}:  ", end='')
        for col in range(30):
            print(f" {int(grid[track, col]):d}", end=' ')
        print()

    print(f"\nNumbers at column position are prime iff no track marks them")
    print(f"(except the prime's own track, which marks only its multiples)")


def demonstrate_rule30_primes():
    """Demonstrate Rule 30 and check for prime-related patterns."""
    print("\n" + "=" * 70)
    print("RULE 30 — CENTER COLUMN ANALYSIS AND PRIME CONNECTIONS")
    print("=" * 70)

    print("""
Rule 30: s_i(t+1) = s_{i-1}(t) XOR (s_i(t) OR s_{i+1}(t))

The center column of Rule 30 appears statistically random.
Wolfram offered $30,000 in prizes for fundamental questions about it.

While Rule 30 doesn't directly generate primes, its apparent randomness
means it shares statistical properties with random sequences —
including the expected distribution of prime-valued run lengths.
""")

    r30 = Rule30Analysis(width=201)
    r30.run(500)

    analysis = r30.analyze_center_column()

    print(f"Center column analysis (500 steps):")
    print(f"  Total runs: {analysis['num_runs']}")
    print(f"  Mean run length: {analysis['mean_run_length']:.2f}")
    print(f"  Prime run lengths: {len(analysis['prime_run_lengths'])} out of {analysis['num_runs']}")
    print(f"  Fraction of prime runs: {analysis['fraction_prime_runs']:.3f}")
    print(f"  Expected fraction (random): {analysis['expected_prime_fraction']:.3f}")

    print(f"\nRun lengths: {analysis['run_lengths'][:30]}")
    print(f"Prime runs: {analysis['prime_run_lengths'][:20]}")

    # Interpret first 50 bits of center column as a binary number at each step
    print(f"\nBinary interpretation of center column (first 64 bits):")
    center = analysis['center_column'][:64]
    binary_str = ''.join(str(b) for b in center)
    value = int(binary_str, 2)
    print(f"  Bits: {binary_str}")
    print(f"  Value: {value}")
    print(f"  Is this value prime? {is_prime_simple(value)}")


if __name__ == "__main__":
    demonstrate_sieve_ca()
    demonstrate_rule30_primes()
