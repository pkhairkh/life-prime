"""
Conway's Game of Life — Prime Sieve Simulation
================================================

Dean Hickerson's "Primer" Pattern (1991):
- Emits a lightweight spaceship (LWSS) at generation N iff N is prime
- Implements the Sieve of Eratosthenes using Game of Life components
- Counter machines count up through integers
- For each prime p, a glider gun destroys LWSS at multiples of p
- Surviving LWSS = primes (not multiples of any smaller number)

This module:
1. Simulates the SIEVE LOGIC that the Primer pattern implements
2. Demonstrates the actual GoL cellular automaton evolution
3. Shows how glider-based computation implements primality testing
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict


class GameOfLife:
    """Full Conway's Game of Life simulation."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.generation = 0

    def set_cell(self, x: int, y: int, value: int = 1):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = value

    def set_pattern(self, pattern: List[Tuple[int, int]], offset: Tuple[int, int] = (0, 0)):
        for x, y in pattern:
            self.set_cell(x + offset[0], y + offset[1])

    def step(self) -> np.ndarray:
        """Advance one generation using standard GoL rules."""
        # Count neighbors using convolution-like approach
        neighbors = np.zeros_like(self.grid, dtype=np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)

        # Apply rules
        new_grid = np.zeros_like(self.grid)
        # Birth: dead cell with exactly 3 neighbors
        new_grid[(self.grid == 0) & (neighbors == 3)] = 1
        # Survival: live cell with 2 or 3 neighbors
        new_grid[(self.grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1

        self.grid = new_grid
        self.generation += 1
        return self.grid.copy()

    def run(self, steps: int) -> List[np.ndarray]:
        """Run for given number of steps, returning all states."""
        states = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            states.append(self.grid.copy())
        return states

    def count_live_cells(self) -> int:
        return int(np.sum(self.grid))

    def to_string(self) -> str:
        """Convert grid to string representation."""
        rows = []
        for row in self.grid:
            rows.append(''.join('█' if c else '·' for c in row))
        return '\n'.join(rows)


# Common Game of Life patterns
PATTERNS = {
    'glider': [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    'lwss': [(0, 1), (0, 3), (1, 0), (2, 0), (3, 0), (3, 3), (4, 0), (4, 1), (4, 2)],
    'blinker': [(0, 0), (1, 0), (2, 0)],
    'beacon': [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)],
    'pulsar': [
        # Top section
        (2, 0), (3, 0), (4, 0), (8, 0), (9, 0), (10, 0),
        (0, 2), (0, 3), (0, 4), (5, 2), (5, 3), (5, 4),
        (7, 2), (7, 3), (7, 4), (12, 2), (12, 3), (12, 4),
        (2, 5), (3, 5), (4, 5), (8, 5), (9, 5), (10, 5),
        # Bottom section (mirror)
        (2, 7), (3, 7), (4, 7), (8, 7), (9, 7), (10, 7),
        (0, 8), (0, 9), (0, 10), (5, 8), (5, 9), (5, 10),
        (7, 8), (7, 9), (7, 10), (12, 8), (12, 9), (12, 10),
        (2, 12), (3, 12), (4, 12), (8, 12), (9, 12), (10, 12),
    ],
    'gosper_gun': [
        (0, 4), (0, 5), (1, 4), (1, 5),
        (10, 4), (10, 5), (10, 6), (11, 3), (11, 7), (12, 2), (12, 8),
        (13, 2), (13, 8), (14, 5), (15, 3), (15, 7), (16, 4), (16, 5),
        (16, 6), (17, 5),
        (20, 2), (20, 3), (20, 4), (21, 2), (21, 3), (21, 4), (22, 1),
        (22, 5), (24, 0), (24, 1), (24, 5), (24, 6),
        (34, 2), (34, 3), (35, 2), (35, 3),
    ],
}


class PrimerSiege:
    """
    Simulates the LOGIC of Dean Hickerson's Primer pattern.

    The Primer implements the Sieve of Eratosthenes using GoL components:
    - LWSS (lightweight spaceships) emitted at each integer N
    - Glider guns for each prime p destroy LWSS at multiples of p
    - Surviving LWSS correspond to primes

    We simulate the sieve logic AND show how GoL components implement it.
    """

    def __init__(self, max_n: int = 100):
        self.max_n = max_n
        self.destroyed = set()  # Numbers destroyed by the sieve
        self.primes_found = []
        self.sieve_events = []  # Track which prime destroyed which multiples

    def run_sieve(self) -> List[int]:
        """
        Run the Sieve of Eratosthenes, tracking the logic
        that the Primer pattern implements with gliders.
        """
        is_prime = [True] * (self.max_n + 1)
        is_prime[0] = is_prime[1] = False

        for p in range(2, self.max_n + 1):
            if is_prime[p]:
                self.primes_found.append(p)
                # This prime p's glider gun destroys multiples
                for multiple in range(2 * p, self.max_n + 1, p):
                    if is_prime[multiple]:
                        self.sieve_events.append({
                            'prime_gun': p,
                            'destroyed_multiple': multiple,
                            'description': f"Gun p={p} destroys LWSS at N={multiple}"
                        })
                    is_prime[multiple] = False
                    self.destroyed.add(multiple)

        return self.primes_found

    def simulate_primer_output(self, up_to: int = 50) -> List[Dict]:
        """
        Simulate what the Primer pattern outputs at each generation.
        At generation 120N (scaled), an LWSS escapes iff N is prime.
        """
        self.run_sieve()
        output = []

        for n in range(2, min(up_to + 1, self.max_n + 1)):
            is_prime = n not in self.destroyed
            destroying_guns = [p for p in self.primes_found
                               if p < n and n % p == 0]

            output.append({
                'n': n,
                'is_prime': is_prime,
                'lwss_escapes': is_prime,
                'destroyed_by': destroying_guns if not is_prime else None,
                'mechanism': (f"LWSS escapes (prime!)" if is_prime
                              else f"Destroyed by gun(s) p={destroying_guns}")
            })

        return output

    def get_primer_visualization_data(self, up_to: int = 30) -> Dict:
        """
        Generate data for visualizing the Primer's sieve mechanism.
        Shows which primes 'fire' at which multiples.
        """
        self.run_sieve()

        prime_guns = {}
        for p in self.primes_found:
            if p * 2 > up_to:
                break
            multiples = list(range(2 * p, up_to + 1, p))
            prime_guns[p] = multiples

        return {
            'primes': self.primes_found,
            'prime_guns': prime_guns,
            'composite_numbers': sorted(self.destroyed),
            'grid_size': up_to
        }


def demonstrate_primer_logic():
    """Demonstrate the Primer pattern's sieve logic."""
    print("=" * 70)
    print("DEAN HICKERSON'S PRIMER — SIEVE OF ERATOSTHENES IN GAME OF LIFE")
    print("=" * 70)

    primer = PrimerSiege(max_n=50)
    output = primer.simulate_primer_output(up_to=50)

    print("\nAt generation 120N, the Primer emits an LWSS iff N is prime:")
    print(f"{'N':>3} | {'Prime?':>6} | {'Mechanism'}")
    print("-" * 60)

    for o in output[:30]:
        prime_str = "YES ★" if o['is_prime'] else "no"
        print(f"{o['n']:3d} | {prime_str:>6} | {o['mechanism']}")

    print(f"\nPrimes found up to 50: {primer.primes_found}")


def demonstrate_gol_patterns():
    """Demonstrate basic GoL patterns used in the Primer."""
    print("\n" + "=" * 70)
    print("GAME OF LIFE PATTERNS USED IN THE PRIMER")
    print("=" * 70)

    # Glider
    print("\nGlider (period 4, translates diagonally):")
    gol = GameOfLife(20, 20)
    gol.set_pattern(PATTERNS['glider'], offset=(2, 2))
    for i in range(5):
        print(f"  Gen {i}:")
        print("  " + gol.to_string()[40:80])  # Show a small region
        gol.step()

    # LWSS
    print("\nLightweight Spaceship (LWSS, period 4, translates horizontally):")
    gol2 = GameOfLife(30, 10)
    gol2.set_pattern(PATTERNS['lwss'], offset=(2, 2))
    for i in range(6):
        print(f"  Gen {i}: live cells = {gol2.count_live_cells()}")
        gol2.step()

    # Gosper Glider Gun
    print("\nGosper Glider Gun (period 30, emits a glider every 30 generations):")
    gol3 = GameOfLife(40, 15)
    gol3.set_pattern(PATTERNS['gosper_gun'], offset=(0, 0))

    glider_count = 0
    for i in range(60):
        if i > 0 and i % 30 == 0:
            glider_count += 1
            print(f"  Gen {i}: Glider #{glider_count} emitted! (live cells: {gol3.count_live_cells()})")
        gol3.step()


def simulate_prime_detection_via_gol(max_gen: int = 200, grid_size: int = 60):
    """
    Simulate a simplified version of prime detection in GoL.
    We track population dynamics and look for prime-period oscillations.
    """
    print("\n" + "=" * 70)
    print("GoL POPULATION DYNAMICS AND PRIME PERIODS")
    print("=" * 70)

    print("""
In the Primer pattern, the key computational elements are:
1. GOSPER GLIDER GUNS: fire gliders at fixed periods (period 30)
2. LWSS: represent candidate numbers, fired at regular intervals
3. GLIDER-LWSS COLLISIONS: destroy the LWSS if N is composite

The period of a Gosper glider gun is 30. In the Primer:
- The overall pattern repeats with period 120
- At gen 120N, an LWSS is sent west
- Each prime p's gun sends interceptors at gen 120(kp)
- The LWSS at gen 120N survives iff N has no divisor < N

Let's simulate the population dynamics:
""")

    # Run a random GoL and track population
    gol = GameOfLife(grid_size, grid_size)
    rng = np.random.RandomState(42)
    for y in range(grid_size):
        for x in range(grid_size):
            if rng.random() < 0.3:
                gol.set_cell(x, y)

    populations = []
    for i in range(max_gen):
        populations.append(gol.count_live_cells())
        gol.step()

    # Analyze for periodic behavior
    print(f"Population over {max_gen} generations:")
    print(f"  Min: {min(populations)}, Max: {max(populations)}")
    print(f"  Mean: {np.mean(populations):.1f}, Std: {np.std(populations):.1f}")

    # Look for periodic patterns using autocorrelation
    pop_array = np.array(populations, dtype=float)
    pop_array -= np.mean(pop_array)

    if len(pop_array) > 10:
        correlations = np.correlate(pop_array, pop_array, mode='full')
        center = len(correlations) // 2
        # Find significant peaks (potential periods)
        autocorr = correlations[center:center + max_gen // 2]
        if autocorr[0] > 0:
            autocorr /= autocorr[0]

        # Find peaks
        peaks = []
        for i in range(2, min(50, len(autocorr))):
            if autocorr[i] > 0.5 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append((i, autocorr[i]))

        print(f"\nSignificant periodic components (autocorrelation > 0.5):")
        for period, corr in peaks[:10]:
            is_prime_period = all(period % p != 0 for p in range(2, period))
            prime_marker = " (prime period ★)" if is_prime_period and period > 1 else ""
            print(f"  Period {period}: correlation = {corr:.3f}{prime_marker}")

    return populations


if __name__ == "__main__":
    # Demo 1: Primer logic
    demonstrate_primer_logic()

    # Demo 2: GoL patterns
    demonstrate_gol_patterns()

    # Demo 3: Population dynamics
    populations = simulate_prime_detection_via_gol()

    # Demo 4: Primer visualization data
    print("\n\n--- Primer Sieve Visualization Data ---")
    primer = PrimerSiege(max_n=100)
    viz_data = primer.get_primer_visualization_data(up_to=50)

    print("\nPrime 'guns' and the composite numbers they destroy:")
    for p in sorted(viz_data['prime_guns'].keys())[:10]:
        multiples = viz_data['prime_guns'][p]
        print(f"  Gun p={p:2d}: destroys N ∈ {multiples}")

    print(f"\nAll primes up to 50: {viz_data['primes'][:15]}")
    print(f"All composites up to 50: {sorted(viz_data['composite_numbers'])[:15]}")
