"""
Rule 90 Cellular Automaton Simulation
======================================
Rule 90: s_i(t+1) = s_{i-1}(t) XOR s_{i+1}(t)

Key connections:
- Generates Sierpiński triangle from single seed
- State at step n, position k = C(n,k) mod 2 (Pascal's triangle mod 2)
- Number of live cells at step n = 2^popcount(n) (Gould's sequence)
- Mersenne numbers n=2^k-1 give maximum live cells = 2^k
"""

import numpy as np
from typing import List, Tuple, Optional
import json


class Rule90Simulation:
    """Simulate Rule 90 cellular automaton on infinite and cyclic grids."""

    def __init__(self, width: int = 256, cyclic: bool = False):
        self.width = width
        self.cyclic = cyclic
        self.history = []

    def step(self, state: np.ndarray) -> np.ndarray:
        """Compute next state using Rule 90 (XOR of neighbors)."""
        if self.cyclic:
            left = np.roll(state, 1)
            right = np.roll(state, -1)
        else:
            left = np.zeros_like(state)
            left[1:] = state[:-1]
            right = np.zeros_like(state)
            right[:-1] = state[1:]
        return (left ^ right).astype(np.uint8)

    def run(self, initial_state: np.ndarray, steps: int) -> List[np.ndarray]:
        """Run simulation for given number of steps."""
        self.history = [initial_state.copy()]
        state = initial_state.copy()
        for _ in range(steps):
            state = self.step(state)
            self.history.append(state.copy())
        return self.history

    @staticmethod
    def single_seed(width: int) -> np.ndarray:
        """Create initial state with single cell in center."""
        state = np.zeros(width, dtype=np.uint8)
        state[width // 2] = 1
        return state

    @staticmethod
    def random_seed(width: int, density: float = 0.5, seed: int = 42) -> np.ndarray:
        """Create random initial state."""
        rng = np.random.RandomState(seed)
        return (rng.random(width) < density).astype(np.uint8)

    def compute_live_cell_counts(self) -> List[int]:
        """Count live cells at each step."""
        return [int(np.sum(state)) for state in self.history]

    def compute_gould_sequence(self, n: int) -> List[int]:
        """
        Compute Gould's sequence: a(n) = 2^popcount(n).
        This equals the number of live cells at step n in Rule 90
        starting from a single seed.
        """
        return [2 ** bin(i).count('1') for i in range(n)]

    def verify_gould_sequence(self) -> List[Tuple[int, int, int, bool]]:
        """
        Verify that Rule 90 live cell counts match Gould's sequence.
        Returns list of (step, actual_count, expected_count, match).
        """
        live_counts = self.compute_live_cell_counts()
        results = []
        for i, count in enumerate(live_counts):
            expected = 2 ** bin(i).count('1')
            results.append((i, count, expected, count == expected))
        return results

    def find_mersenne_steps(self, max_step: int) -> List[Tuple[int, int]]:
        """
        Find steps where n = 2^k - 1 (Mersenne numbers).
        At these steps, live cell count is maximized: 2^k.
        Returns list of (step, live_count) for Mersenne-numbered steps.
        """
        results = []
        k = 1
        while True:
            mersenne_n = 2**k - 1
            if mersenne_n >= len(self.history):
                break
            if mersenne_n > max_step:
                break
            live_count = int(np.sum(self.history[mersenne_n]))
            results.append((mersenne_n, live_count))
            k += 1
        return results

    def analyze_pascal_mod2(self, max_n: int = 50) -> dict:
        """
        Verify that Rule 90 states equal Pascal's triangle mod 2.
        For step n, position k: state[k] = C(n,k) mod 2.
        """
        results = {
            'matches': 0,
            'mismatches': 0,
            'details': []
        }

        for n in range(min(max_n, len(self.history))):
            state = self.history[n]
            offset = (len(state) - n - 1) // 2 if not self.cyclic else 0

            for k in range(min(n + 1, len(state))):
                # Compute C(n,k) mod 2
                pascal_val = binomial_mod2(n, k)
                idx = offset + k if not self.cyclic else k

                if 0 <= idx < len(state):
                    rule90_val = int(state[idx])
                    if rule90_val == pascal_val:
                        results['matches'] += 1
                    else:
                        results['mismatches'] += 1
                        results['details'].append({
                            'step': n, 'pos': k,
                            'rule90': rule90_val,
                            'pascal_mod2': pascal_val
                        })

        return results


def binomial_mod2(n: int, k: int) -> int:
    """
    Compute C(n,k) mod 2 using Lucas' theorem.
    C(n,k) ≡ 1 (mod 2) iff every binary digit of k ≤ corresponding digit of n.
    Equivalently: (k AND (NOT n)) == 0
    """
    if k > n or k < 0:
        return 0
    return 1 if (k & (~n)) == 0 else 0


def demonstrate_rule90_sierpinski(width: int = 129, steps: int = 64):
    """
    Demonstrate Rule 90 generating the Sierpiński triangle.
    Returns the full history as a 2D array.
    """
    sim = Rule90Simulation(width=width, cyclic=False)
    initial = Rule90Simulation.single_seed(width)
    history = sim.run(initial, steps)
    return np.array(history), sim


def demonstrate_mersenne_peak(width: int = 512, steps: int = 128):
    """
    Demonstrate that Mersenne-numbered steps (n=2^k-1) give maximum
    live cell counts in Rule 90.
    """
    sim = Rule90Simulation(width=width, cyclic=False)
    initial = Rule90Simulation.single_seed(width)
    history = sim.run(initial, steps)

    live_counts = sim.compute_live_cell_counts()

    # Find Mersenne steps and their live counts
    mersenne_steps = []
    k = 1
    while 2**k - 1 < len(live_counts):
        n = 2**k - 1
        mersenne_steps.append({
            'k': k,
            'mersenne_n': n,
            'live_count': live_counts[n],
            'expected': 2**k,
            'is_mersenne_prime': is_mersenne_prime(k)
        })
        k += 1

    return live_counts, mersenne_steps, sim


def is_mersenne_prime(p: int) -> bool:
    """Check if 2^p - 1 is a Mersenne prime using known list."""
    known_exponents = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}
    return p in known_exponents


if __name__ == "__main__":
    print("=" * 60)
    print("Rule 90 Cellular Automaton Simulation")
    print("=" * 60)

    # Demo 1: Sierpiński triangle
    print("\n--- Sierpiński Triangle from Rule 90 ---")
    history, sim = demonstrate_rule90_sierpinski(width=65, steps=32)

    # Print small version
    for i, state in enumerate(history[:16]):
        row = ''.join('█' if c else ' ' for c in state[16:49])
        print(f"Step {i:2d}: {row}")

    # Demo 2: Gould's sequence verification
    print("\n--- Gould's Sequence Verification ---")
    print("Step n | Live cells | 2^popcount(n) | Match")
    print("-" * 50)

    sim2 = Rule90Simulation(width=256, cyclic=False)
    initial = Rule90Simulation.single_seed(256)
    sim2.run(initial, 64)

    for i in range(33):
        count = int(np.sum(sim2.history[i]))
        expected = 2 ** bin(i).count('1')
        match = "✓" if count == expected else "✗"
        if i < 20 or count != 2 ** bin(i-1).count('1'):
            print(f"  {i:4d} | {count:10d} | {expected:13d} | {match}")

    # Demo 3: Mersenne steps
    print("\n--- Mersenne-Numbered Steps (n=2^k-1) ---")
    live_counts, mersenne_data, _ = demonstrate_mersenne_peak(width=512, steps=128)

    print("k | n=2^k-1 | Live Cells | Expected 2^k | Mersenne Prime?")
    print("-" * 60)
    for d in mersenne_data:
        mp = "YES ★" if d['is_mersenne_prime'] else "no"
        print(f"{d['k']:2d} | {d['mersenne_n']:7d} | {d['live_count']:10d} | {d['expected']:12d} | {mp}")

    print("\nKey insight: At step n=2^k-1 (Mersenne number), all k bits are 1,")
    print("so popcount(n)=k, giving 2^k live cells — the maximum for that bit-width.")
    print("This is the direct CA manifestation of Mersenne number structure.")
