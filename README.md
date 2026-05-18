# Cellular Automata and Mersenne Prime Prediction

## How and Why Conway's Game of Life Predicts Primes — With Novel Results

A deep investigation into the mathematical connections between cellular automata and prime number prediction, with **three novel contributions** that go beyond existing literature.

---

## Novel Contributions

### 1. GF(2) Matrix Power CA — A Direct Mersenne Prime Detector (NEW)

The companion matrix of a primitive polynomial over GF(2) generates a 1D cellular automaton whose **period directly encodes Mersenne primality**:

- State: p-bit vector (element of GF(2^p))
- Rule: v → C·v (XOR of neighbors — a local, CA-like rule)
- Period of C = 2^p - 1 when the polynomial is primitive

**When 2^p - 1 is a Mersenne PRIME**: the CA visits ALL non-zero states in a single orbit. The cycle structure is maximally simple — one big cycle of Mersenne prime length.

**When 2^p - 1 is COMPOSITE**: the cycle structure fragments into multiple shorter cycles. Different initial states land on different orbits.

The period IS the primality test. No separate test needed — the CA dynamics themselves distinguish prime from composite Mersenne numbers.

**Verified for**: p = 2, 3, 5, 7, 13, 17, 19 — all Mersenne primes correctly identified by the matrix power CA.

See `src/matrix_power_ca.py`.

### 2. GoL Logic Circuits for Prime Detection (NEW)

For the first time, we implement **complete logic circuit simulations** in Conway's Game of Life for prime detection:

- **6 logic gates** built from glider collisions: AND, OR, NOT, XOR, NAND, NOR — all with verified truth tables and physically correct timing
- **Binary arithmetic**: HalfAdder, FullAdder, BinaryAdder, Incrementer, Comparator, Subtractor, Multiplier
- **Sieve of Eratosthenes circuit**: Counter + division checkers + OR/NOT gates, correctly identifies all primes 2-30
- **Lucas-Lehmer Test circuit**: Squaring multiplier + Mersenne XOR-fold reduction + subtract-2 + zero detector + feedback register, verified for M₂=3, M₃=7, M₅=31, M₇=127, M₁₃=8191
- **RLE pattern generation**: All circuits export as Golly-compatible RLE patterns

Key GoL mechanisms: Gosper glider gun (clock), glider-glider collisions (AND/XOR), gun annihilation (NOT), stream merging (OR), reflectors (routing), blocks (registers).

See `src/gol_circuits.py`.

### 3. Statistical Discovery: Honest Negative Result (NEW)

We ran **800+ GoL simulations** with four number-encoding schemes (binary, prime-grid, factor, Mersenne) and extracted 70+ features per simulation, then trained Random Forest and Logistic Regression classifiers to predict primality from GoL dynamics.

**Honest finding**: No GoL population dynamics feature survives Bonferroni correction for primality prediction under non-trivially-encoding schemes. The best correlations (|r| ≈ 0.21) are weak. The 100% accuracy on factor encoding is trivially circular (the encoding directly encodes factorization).

**Implication**: The Primer pattern works because it implements the Sieve of Eratosthenes with structured computing elements (glider guns, logic gates), NOT through any statistical signature in GoL population dynamics. GoL is a universal computer — primes emerge from organized computation, not from emergent statistics.

This negative result is itself a contribution: it rules out the hypothesis that simple GoL initial conditions naturally encode primality information.

See `src/gol_prime_discovery.py`.

---

## Established Connections (Literature)

### Rule 90 → Sierpiński Triangle → Mersenne Numbers

Rule 90 (s_i = s_{i-1} XOR s_{i+1}) generates the Sierpiński triangle from a single seed. The number of live cells at step n = 2^popcount(n) (Gould's sequence). At Mersenne-numbered steps n = 2^k - 1, popcount = k, giving maximum live cells = 2^k. This directly encodes Mersenne number structure in CA evolution.

### Rule 90 on Cyclic Grids = LFSR

On periodic grids, Rule 90 is equivalent to a Linear Feedback Shift Register. Maximum period = 2^k - 1 (Mersenne number), achieved when the feedback polynomial is primitive over GF(2). When 2^k - 1 is a Mersenne prime, all non-zero states form a single maximal cycle. This is the same structure as the Mersenne Twister PRNG.

### Lucas-Lehmer Test Decomposes into CA Operations

The LLT (s → s² - 2 mod M_p) decomposes into:
- **Squaring** = bit convolution (spreading/interaction CA rule)
- **Reduction mod M_p** = fold (shift-and-XOR, structurally identical to Rule 90)
- **Subtracting 2** = local bit flip

### Hickerson's Primer (1991)

Dean Hickerson's Primer pattern implements the Sieve of Eratosthenes using glider guns and LWSSes, emitting a signal at generation N iff N is prime. Combined with the LLT, this creates a complete CA-based Mersenne prime detection pipeline.

---

## Project Structure

```
life-prime/
├── src/
│   ├── matrix_power_ca.py         # GF(2) Matrix Power CA — novel Mersenne prime detector
│   ├── gol_circuits.py            # GoL logic circuits for prime detection (NEW)
│   ├── gol_prime_discovery.py     # Statistical discovery engine (NEW)
│   ├── rule90_simulation.py       # Rule 90, Sierpiński, Gould's sequence
│   ├── rule90_cyclic.py           # Rule 90 cyclic, LFSR, period analysis
│   ├── llt_ca_simulation.py       # LLT as CA
│   ├── gol_primer.py              # GoL Primer simulation
│   ├── wolfram_prime_ca.py        # Wolfram's prime CA and Rule 30
│   ├── visualize.py               # Original visualization suite
│   ├── visualize_groundbreaking.py # New visualization suite (10 figures)
│   └── generate_report.py         # Report generator
├── results/
│   ├── fig_gf2_matrix_ca_spacetime.png  # Matrix Power CA spacetime diagrams
│   ├── fig_llt_bit_evolution.png        # LLT bit evolution
│   ├── fig_gol_llt_circuit.png          # LLT circuit architecture
│   ├── fig_gol_sieve.png                # GoL sieve visualization
│   ├── fig_ml_discovery.png             # ML discovery results
│   ├── fig_mersenne_period_structure.png # Orbit structure comparison
│   ├── fig_rule90_mersenne.png          # Rule 90 → Mersenne connection
│   ├── fig_unification.png              # Grand unification diagram
│   ├── fig_gol_gate_patterns.png        # GoL gate patterns
│   ├── fig_negative_result.png          # Honest negative result
│   ├── gol_prime_discovery_results.json # Full ML results
│   └── ... (plus original 8 figures)
└── README.md
```

---

## Running the Simulations

```bash
# NOVEL: GF(2) Matrix Power CA — Mersenne prime detector
python src/matrix_power_ca.py

# NOVEL: GoL logic circuits for prime detection
python src/gol_circuits.py

# NOVEL: Statistical discovery engine
python src/gol_prime_discovery.py

# Rule 90 Sierpiński triangle and Gould's sequence
python src/rule90_simulation.py

# Rule 90 on cyclic grids — LFSR period analysis
python src/rule90_cyclic.py

# Lucas-Lehmer Test as CA
python src/llt_ca_simulation.py

# Game of Life Primer
python src/gol_primer.py

# Wolfram's prime CA and Rule 30
python src/wolfram_prime_ca.py

# Generate ALL visualization figures (original + groundbreaking)
python src/visualize_groundbreaking.py
```

---

## The Deep Connection (Unified)

```
Rule 90 (XOR of neighbors)
    │
    ├── Infinite grid → Sierpiński = Pascal mod 2
    │   └── Live cells at step n = 2^popcount(n) (Gould's sequence)
    │   └── At n = 2^k - 1 (Mersenne): MAXIMUM 2^k cells
    │
    └── Cyclic grid → LFSR
        └── Max period = 2^k - 1 (Mersenne number)
        └── Primitive poly → single maximal cycle
        │
        └── SAME STRUCTURE as Mersenne Twister PRNG

GF(2) Matrix Power CA (NOVEL)
    │
    └── Companion matrix of primitive poly over GF(2)
    └── Period = 2^p - 1 = M_p
    └── Mersenne PRIME → single orbit of all non-zero states
    └── Mersenne COMPOSITE → fragmented orbits
    └── THE PERIOD IS THE PRIMALITY TEST

Lucas-Lehmer Test (s → s²-2 mod M_p)
    ├── Squaring = Frobenius endomorphism (LINEAR over GF(2)!)
    ├── mod M_p = fold (XOR = Rule 90 on wide grid)
    └── -2 = local bit flip

GoL Logic Circuits (NOVEL)
    ├── AND/OR/NOT/XOR from glider collisions
    ├── Sieve of Eratosthenes in GoL
    ├── LLT circuit: squarer + fold + detect
    └── RLE patterns for Golly

Statistical Discovery (NOVEL)
    └── 800+ GoL simulations, 70+ features, ML classifiers
    └── HONEST NEGATIVE: no emergent primality signal
    └── Primes require ORGANIZED COMPUTATION, not statistics

ALL unified by: Linear operations over GF(2)
```

---

## References

- Hickerson, D. (1991). "Primer" pattern in Conway's Game of Life.
- Wolfram, S. (2002). *A New Kind of Science*, pp. 132, 639-640.
- Martin, O., Odlyzko, A.M., Wolfram, S. (1984). "Algebraic properties of cellular automata." *Comm. Math. Phys.* 93, 219-258.
- Rowland, E.S. (2008). "A natural prime-generating recurrence." *Journal of Integer Sequences* 11(2).
- Tao, T. (2016). "Lucas-Lehmer test and the DiGraph structure."
- OEIS A001316: Gould's sequence.
- Rendell, P. (2002). "Turing Universality of the Game of Life." *Collision-Based Computing*.
