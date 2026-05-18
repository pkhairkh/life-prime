# Cellular Automata and Prime/Mersenne Prime Prediction

## How and Why Conway's Game of Life Predicts Primes, Especially Mersenne Primes

A comprehensive simulation-based investigation into the mathematical connections between cellular automata (CA) and prime number prediction, with special focus on Mersenne primes.

---

## Key Findings

1. **Rule 90 → Mersenne Number Detection**: The elementary CA Rule 90 (XOR of neighbors) generates the Sierpiński triangle from a single seed. At Mersenne-numbered steps n = 2^k - 1, the number of live cells reaches its theoretical maximum of 2^k, directly encoding Mersenne number structure in the CA evolution.

2. **Rule 90 on Cyclic Grids = LFSR**: Rule 90 on periodic grids is mathematically equivalent to a Linear Feedback Shift Register. Maximum LFSR period = 2^k - 1 (a Mersenne number), achieved when the feedback polynomial is primitive over GF(2). When 2^k - 1 is a **Mersenne prime**, all non-zero states form a single maximal cycle.

3. **Lucas-Lehmer Test IS a Cellular Automaton**: The LLT for Mersenne primes decomposes into CA operations:
   - **Squaring** = bit convolution (a spreading/interaction CA rule)
   - **Reduction mod M_p** = fold (shift-and-XOR, structurally identical to Rule 90!)
   - **Subtracting 2** = local bit flip

4. **Game of Life Primer**: Dean Hickerson's 1991 Primer pattern implements the Sieve of Eratosthenes using glider guns and LWSSes, emitting a signal at generation N iff N is prime. Combined with the LLT, this creates a complete CA-based Mersenne prime detection pipeline.

5. **Unifying Structure**: All connections arise from the shared algebraic structure of linear operations over GF(2). The Mersenne Twister PRNG, error-correcting codes, and cryptographic stream ciphers all exploit this same structure.

---

## Project Structure

```
life-prime/
├── src/
│   ├── rule90_simulation.py      # Rule 90 on infinite grids, Sierpiński triangle, Gould's sequence
│   ├── rule90_cyclic.py           # Rule 90 on cyclic grids, LFSR equivalence, period analysis
│   ├── llt_ca_simulation.py       # Lucas-Lehmer Test as cellular automaton
│   ├── gol_primer.py              # Game of Life Primer pattern simulation
│   ├── wolfram_prime_ca.py        # Wolfram's 16-color prime CA and Rule 30 analysis
│   ├── visualize.py               # Comprehensive visualization generation
│   └── generate_report.py         # Research report PDF generation
├── results/
│   ├── fig1_rule90_sierpinski.png     # Rule 90 Sierpiński triangle
│   ├── fig2_mersenne_peak.png         # Mersenne number peak analysis
│   ├── fig3_cyclic_periods.png        # Cyclic grid period analysis
│   ├── fig4_llt_ca.png                # LLT as CA visualization
│   ├── fig5_gol_primer.png            # Game of Life Primer mechanism
│   ├── fig6_summary_connections.png   # Complete connection map
│   ├── fig7_llt_results.png           # LLT results table
│   ├── fig8_rule90_patterns.png       # Pattern comparison
│   ├── rule90_cyclic_periods.json     # Raw simulation data
│   └── life_prime_research_report.pdf # Full research report
└── README.md
```

---

## Running the Simulations

```bash
# Rule 90 Sierpiński triangle and Gould's sequence verification
python src/rule90_simulation.py

# Rule 90 on cyclic grids — period and Mersenne prime analysis
python src/rule90_cyclic.py

# Lucas-Lehmer Test as cellular automaton
python src/llt_ca_simulation.py

# Game of Life Primer pattern
python src/gol_primer.py

# Wolfram's prime CA and Rule 30
python src/wolfram_prime_ca.py

# Generate all visualization figures
python src/visualize.py

# Generate the research report PDF
python src/generate_report.py
```

---

## The Deep Connection (Summary)

```
Rule 90 (XOR of neighbors)
    │
    ├── Infinite grid → Sierpiński triangle = Pascal mod 2
    │   └── Live cells at step n = 2^popcount(n) (Gould's sequence)
    │   └── At n = 2^k - 1 (Mersenne): popcount = k → 2^k cells (MAXIMUM!)
    │
    └── Cyclic grid → Equivalent to LFSR
        └── Max period = 2^k - 1 (Mersenne number)
        └── Max achieved when feedback poly is primitive over GF(2)
        └── When 2^k - 1 is MERSENNE PRIME → single maximal cycle
        │
        └── Same structure as Mersenne Twister PRNG

Lucas-Lehmer Test (s → s²-2 mod M_p)
    ├── Squaring = bit convolution CA
    ├── mod M_p = fold (XOR of spatially separated bits = Rule 90!)
    └── -2 = local bit flip

Game of Life Primer (Hickerson 1991)
    └── Sieve of Eratosthenes via glider guns
    └── Emits signal at generation N iff N is prime
    └── Provides prime inputs for LLT → Mersenne prime detection

ALL unified by: Linear operations over GF(2)
```

---

## References

- Hickerson, D. (1991). "Primer" pattern in Conway's Game of Life.
- Wolfram, S. (2002). *A New Kind of Science*, pp. 132, 639-640.
- Martin, O., Odlyzko, A.M., Wolfram, S. (1984). "Algebraic properties of cellular automata." *Comm. Math. Phys.* 93, 219-258.
- Rowland, E.S. (2008). "A natural prime-generating recurrence." *Journal of Integer Sequences* 11(2).
- Tao, T. (2016). "Lucas-Lehmer test and the DiGraph structure." *What's New* blog.
- OEIS A001316: Gould's sequence.
