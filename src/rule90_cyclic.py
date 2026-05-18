"""
Rule 90 on Cyclic Grids — Period Analysis and Mersenne Prime Connection
========================================================================

Key mathematical result:
- Rule 90 on a cyclic grid of size N is equivalent to a Linear Feedback Shift Register (LFSR)
- The transition matrix is a linear operator over GF(2)
- Maximum period for even N=2k is 2^k - 1 (a Mersenne number!)
- The period achieves this maximum iff the characteristic polynomial is primitive over GF(2)
- When 2^k - 1 is a Mersenne PRIME, the cycle structure is most elegant:
  all non-zero states lie on a single cycle of Mersenne prime length

This is the SAME mathematical structure underlying:
- LFSR maximal period sequences (m-sequences)
- The Mersenne Twister PRNG
- Primitive polynomial theory over GF(2)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import time
import json


class CyclicRule90:
    """Rule 90 on a cyclic (periodic boundary) grid."""

    def __init__(self, size: int):
        self.size = size
        self.state = None
        self.history = []

    def set_state(self, state: np.ndarray):
        """Set initial state."""
        assert len(state) == self.size
        self.state = state.copy().astype(np.uint8)
        self.history = [self.state.tobytes()]

    def step(self) -> np.ndarray:
        """Compute next state with cyclic boundary conditions."""
        left = np.roll(self.state, 1)
        right = np.roll(self.state, -1)
        self.state = (left ^ right).astype(np.uint8)
        return self.state.copy()

    def find_period(self, max_steps: int = None) -> Optional[int]:
        """
        Find the period of the current state by detecting when
        the state repeats.
        """
        if max_steps is None:
            max_steps = 2 ** (self.size // 2 + 1)  # Theoretical max

        seen = {self.state.tobytes(): 0}
        for step in range(1, max_steps + 1):
            self.step()
            state_bytes = self.state.tobytes()
            if state_bytes in seen:
                period = step - seen[state_bytes]
                return period
            seen[state_bytes] = step

        return None  # Period exceeds max_steps

    def find_period_fast(self, max_steps: int = None) -> Optional[int]:
        """
        Find period using Floyd's cycle detection (tortoise and hare).
        More memory efficient for large grids.
        """
        if max_steps is None:
            max_steps = 2 ** (self.size // 2 + 1)

        # Reset to initial state
        initial = np.frombuffer(self.history[0], dtype=np.uint8).copy()
        self.state = initial.copy()

        # Tortoise and hare
        tortoise = initial.copy()
        hare = initial.copy()

        def step_state(s):
            left = np.roll(s, 1)
            right = np.roll(s, -1)
            return (left ^ right).astype(np.uint8)

        # Phase 1: Find meeting point
        tortoise = step_state(tortoise)
        hare = step_state(step_state(hare))

        steps = 0
        while not np.array_equal(tortoise, hare) and steps < max_steps:
            tortoise = step_state(tortoise)
            hare = step_state(step_state(hare))
            steps += 1

        if steps >= max_steps:
            return None

        # Phase 2: Find period
        period = 1
        tortoise = initial.copy()
        hare = step_state(tortoise)  # Move hare one step from start

        while not np.array_equal(tortoise, hare):
            tortoise = step_state(tortoise)
            hare = step_state(hare)
            period += 1
            if period > max_steps:
                return None

        return period


def analyze_period_for_grid_sizes(sizes: List[int], max_steps_factor: int = 2) -> Dict:
    """
    Analyze Rule 90 periods for various cyclic grid sizes.

    For even N=2k: max period = 2^k - 1
    For odd N: period divides 2^ord(2,2N+1) - 1

    Returns analysis results.
    """
    results = []

    for size in sizes:
        # Compute theoretical max period
        if size % 2 == 0:
            k = size // 2
            theoretical_max = 2**k - 1
        else:
            theoretical_max = None  # More complex for odd sizes

        max_steps = 2**(size // 2 + 1) if size <= 30 else 2**20

        start_time = time.time()

        # Try multiple initial states to find maximum period
        best_period = 0
        num_trials = min(50, 2**size) if size <= 16 else 20
        rng = np.random.RandomState(42)

        for trial in range(num_trials):
            sim = CyclicRule90(size)
            if trial == 0:
                # Single seed
                initial = np.zeros(size, dtype=np.uint8)
                initial[0] = 1
            elif trial == 1:
                # All ones
                initial = np.ones(size, dtype=np.uint8)
            else:
                # Random
                initial = (rng.random(size) > 0.5).astype(np.uint8)

            sim.set_state(initial)
            period = sim.find_period(max_steps) if size <= 20 else sim.find_period_fast(max_steps)
            if period is not None and period > best_period:
                best_period = period

        elapsed = time.time() - start_time

        is_mersenne = False
        if theoretical_max is not None:
            is_mersenne = is_mersenne_number(theoretical_max)

        result = {
            'grid_size': size,
            'period': best_period,
            'theoretical_max_period': theoretical_max,
            'achieves_max': best_period == theoretical_max if best_period and theoretical_max else None,
            'is_max_mersenne': is_mersenne,
            'computation_time': elapsed
        }
        results.append(result)

        status = ""
        if best_period:
            status = f"max_period_found={best_period}"
            if theoretical_max and best_period == theoretical_max:
                status += " (MAXIMUM!)"
        else:
            status = "period > max_steps"

        print(f"  N={size:3d}: {status}  [theoretical max: {theoretical_max}]  ({elapsed:.3f}s)")

    return results


def is_mersenne_number(n: int) -> bool:
    """Check if n is a Mersenne number (n = 2^k - 1 for some k)."""
    return (n + 1) & n == 0 and n > 0


def multiplicative_order(a: int, n: int) -> int:
    """Compute the multiplicative order of a modulo n."""
    if n == 1:
        return 1
    order = 1
    current = a % n
    while current != 1:
        current = (current * a) % n
        order += 1
        if order > n:
            return -1  # No order exists (a and n not coprime)
    return order


def analyze_mersenne_period_structure(exponents: List[int]) -> Dict:
    """
    For each k, analyze the period structure of Rule 90 on grid size 2k.
    When 2^k - 1 is Mersenne prime, all non-zero states form a single cycle.
    """
    results = []

    known_mersenne_exponents = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127,
                                 521, 607, 1279, 2203, 2281, 3217, 4253, 4423}

    for k in exponents:
        mersenne_num = 2**k - 1
        is_prime = k in known_mersenne_exponents

        # For grid size 2k, max period = 2^k - 1 = Mersenne number
        grid_size = 2 * k

        result = {
            'k': k,
            'grid_size': grid_size,
            'mersenne_number': mersenne_num,
            'is_mersenne_prime': is_prime,
            'max_period': mersenne_num,
            'achieves_max': None  # Would need actual simulation
        }

        # For small k, we can actually verify
        if k <= 12:
            sim = CyclicRule90(grid_size)
            initial = np.zeros(grid_size, dtype=np.uint8)
            initial[0] = 1
            sim.set_state(initial)
            period = sim.find_period(2**(k+1))
            result['actual_period'] = period
            result['achieves_max'] = (period == mersenne_num)

            # Count distinct cycles by trying different initial states
            if k <= 8:
                cycle_info = count_cycles(grid_size, mersenne_num)
                result['cycle_info'] = cycle_info

        results.append(result)

    return results


def count_cycles(grid_size: int, expected_period: int) -> Dict:
    """
    Count the number of distinct cycles and their lengths
    for Rule 90 on a cyclic grid of given size.
    """
    visited = set()
    cycles = []
    total_states = 2**grid_size

    def step_state(s):
        left = np.roll(s, 1)
        right = np.roll(s, -1)
        return (left ^ right).astype(np.uint8)

    # Sample initial states (can't enumerate all for large grids)
    num_samples = min(1000, total_states)
    rng = np.random.RandomState(42)

    for _ in range(num_samples):
        initial = rng.randint(0, 2, size=grid_size).astype(np.uint8)
        initial_bytes = initial.tobytes()

        if initial_bytes in visited:
            continue

        # Find period of this state
        state = initial.copy()
        seen = {initial_bytes: 0}
        for step in range(1, expected_period + 2):
            state = step_state(state)
            sb = state.tobytes()
            if sb in seen:
                period = step - seen[sb]
                cycles.append(period)
                break
            seen[sb] = 0
            visited.add(sb)

    return {
        'num_cycles_found': len(cycles),
        'cycle_lengths': Counter(cycles),
        'max_cycle_length': max(cycles) if cycles else 0,
        'states_sampled': num_samples
    }


def demonstrate_lfsr_equivalence(sizes: List[int]):
    """
    Demonstrate that Rule 90 on cyclic grids is equivalent to an LFSR.
    Show that the period matches the LFSR maximal period when conditions are met.
    """
    print("\n" + "=" * 70)
    print("RULE 90 ↔ LFSR EQUIVALENCE DEMONSTRATION")
    print("=" * 70)

    print("""
Rule 90 on a cyclic grid of size N is mathematically equivalent to a
Linear Feedback Shift Register (LFSR). The key facts:

1. Rule 90 is a LINEAR operator over GF(2) — XOR of neighbors
2. On cyclic grids, the transition matrix has characteristic polynomial
   related to x^N + 1 over GF(2)
3. Maximum LFSR period = 2^k - 1 (Mersenne number!) for register length k
4. Maximum achieved iff feedback polynomial is PRIMITIVE over GF(2)
5. When 2^k - 1 is a MERSENNE PRIME, the cycle structure is optimal:
   ALL non-zero states form a SINGLE cycle of Mersenne prime length

This is WHY the Mersenne Twister PRNG works — it exploits this exact
connection by using M_p = 2^19937 - 1 (a Mersenne prime) as its period.
""")

    for size in sizes:
        k = size // 2
        if size % 2 != 0:
            print(f"  N={size} (odd): period structure more complex, skipping")
            continue

        mersenne = 2**k - 1
        is_mp = k in {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}

        if k <= 12:
            sim = CyclicRule90(size)
            initial = np.zeros(size, dtype=np.uint8)
            initial[0] = 1
            sim.set_state(initial)
            period = sim.find_period(2**(k+2))
            achieves = "YES ★" if period == mersenne else "no"
        else:
            period = "N/A (too large)"
            achieves = "N/A"

        mp_label = "MERSENNE PRIME" if is_mp else "Mersenne number"
        print(f"  N={size:3d} (k={k:2d}): max period = 2^{k}-1 = {mersenne} ({mp_label})")
        print(f"           actual period = {period}, achieves max: {achieves}")


if __name__ == "__main__":
    print("=" * 70)
    print("RULE 90 ON CYCLIC GRIDS — PERIOD & MERSENNE PRIME ANALYSIS")
    print("=" * 70)

    # Analysis 1: Periods for small grid sizes
    print("\n--- Period Analysis for Various Grid Sizes ---")
    print("Even grid sizes N=2k: theoretical max period = 2^k - 1\n")

    sizes = list(range(4, 27, 2))  # Even sizes from 4 to 26
    results = analyze_period_for_grid_sizes(sizes)

    # Analysis 2: Mersenne period structure
    print("\n--- Mersenne Period Structure ---")
    print("Grid size 2k: when 2^k-1 is Mersenne prime, all states on one cycle\n")

    mersenne_results = analyze_mersenne_period_structure([2, 3, 4, 5, 6, 7, 8, 10, 12])
    for r in mersenne_results:
        mp = "MERSENNE PRIME ★" if r['is_mersenne_prime'] else "composite"
        actual = r.get('actual_period', 'N/A')
        achieves = r.get('achieves_max', 'N/A')
        print(f"  k={r['k']:2d}, N={r['grid_size']:3d}: M_{r['k']}={r['mersenne_number']:5d} ({mp})")
        print(f"         actual_period={actual}, achieves_max={achieves}")

    # Analysis 3: LFSR equivalence
    demonstrate_lfsr_equivalence([4, 6, 8, 10, 12, 14, 16, 18, 20])

    # Save results
    output = {
        'period_analysis': [
            {k: v for k, v in r.items() if k != 'cycle_info'}
            for r in results
        ],
        'mersenne_structure': mersenne_results
    }

    with open('/home/z/my-project/life-prime/results/rule90_cyclic_periods.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\nResults saved to results/rule90_cyclic_periods.json")
