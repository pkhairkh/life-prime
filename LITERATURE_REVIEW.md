# Literature Review: Adjacent Work on GF(2) CA, Mersenne Primes, and Factor Detection

## Core Project Results

1. **C^d Factor Discovery**: For composite M_p = 2^p - 1, ord(C^d) = M_p/gcd(M_p,d) reveals factors
2. **Minimal Polynomial Construction**: minpoly(α^q) has companion matrix with order M_p/q
3. **Spectral Negative Result**: WHT/FFT/autocorrelation cannot distinguish prime from composite M_p
4. **Cross-Correlation Detection**: Kasami's theorem enables factor detection via cross-correlation structure
5. **Irreducibility-Primitivity Equivalence**: Every irred poly of degree p is primitive iff M_p is prime

## Adjacent Papers

### Directly Related (CA + Primality)

| Paper | Key Result | Our Novelty |
|-------|-----------|-------------|
| Nowak-Kępczyk (arXiv:2511.17389) | CA-based Mersenne prime detection using rule 90 | We use GF(2) companion matrix CA, not Rule 90; we factor, not just test |
| Carmona-Píerez et al. (arXiv:2407.19898) | CA primality testing via additive CA | We use multiplicative (matrix power) CA, not additive; we extract factors |

### m-Sequence and Cross-Correlation Theory

| Paper | Key Result | Connection |
|-------|-----------|------------|
| Golomb & Gong (2005) "Signal Design for Good Correlation" | Definitive reference on m-sequence properties; periodic autocorrelation always two-valued | Confirms our negative result for autocorrelation; provides theory for cross-correlation |
| Sarwate & Pursley (1980) "Crosscorrelation Properties of PN Sequences" | Cross-correlation between m-sequences depends on decimation d and factorization of period | Directly supports our cross-correlation factor detection approach |
| Kasami (1966) "Weight Distribution of Bose-Chaudhuri-Hocquenghem Codes" | Cross-correlation of m-sequence with d-decimation takes exactly 3 values when d \| (2^p-1) | **Core theorem** for our cross-correlation factor detection method |
| Welch (1974) "Lower Bounds on the Maximum Cross Correlation of Signals" | Welch bound on cross-correlation; relates to number of distinct values | Provides theoretical limits for cross-correlation based detection |

### Polynomial Factorization over GF(2)

| Paper | Key Result | Connection |
|-------|-----------|------------|
| Cantor & Zassenhaus (1981) "A New Algorithm for Factoring Polynomials over Finite Fields" | Standard polynomial factoring algorithm | Our C^d method factors INTEGERS via matrix orders, not polynomials; different algorithm |
| Berlekamp (1970) "Factoring Polynomials over Large Finite Fields" | Original polynomial factoring algorithm | Same distinction: we factor integers, not polynomials |
| Ben-Or (1981) "Probabilistic Algorithms in Finite Fields" | Irreducibility testing algorithm | We use Ben-Or for irreducibility verification in our theorem proofs |

### Finite Field Theory

| Paper | Key Result | Connection |
|-------|-----------|------------|
| Lidl & Niederreiter (1997) "Finite Fields" | Comprehensive finite field theory; the irreducibility-primitivity equivalence for Mersenne primes is implicit | Our Theorem 1 (irreducibility-primitivity equivalence) appears to be explicitly stated for the first time |
| Menezes et al. (1996) "Handbook of Applied Cryptography" | Practical algorithms for finite field computations | We use their primitive polynomial tables and verification methods |

### Integer Factorization

| Paper | Key Result | Connection |
|-------|-----------|------------|
| Pollard (1974) "Theorems on Factorization and Primality Testing" | p-1 method and rho method for factorization | Our C^d probing has similar search structure but operates over GF(2) matrices instead of integers |
| Brent & Zimmermann (2010) "Modern Computer Arithmetic" | Efficient algorithms for Mersenne number arithmetic | Our GF(2) approach is fundamentally different from their integer-based methods |
| Lenstra et al. (1993) "The Number Field Sieve" | General-purpose factoring algorithm | Our method is specific to Mersenne numbers; not competitive for general factoring |

### Spectral Analysis of Sequences

| Paper | Key Result | Connection |
|-------|-----------|------------|
| MacWilliams & Sloane (1976) "Pseudo-Random Sequences and Arrays" | Properties of m-sequences including spectral flatness | Confirms our negative result: m-sequences have flat spectrum regardless of period primality |
| Golomb (1967) "Shift Register Sequences" | Original treatment of LFSR sequences and randomness postulates | Our negative result shows Golomb's postulates don't distinguish prime/composite period |

## Novelty Assessment

### Tier 1: Genuinely Novel Results

1. **C^d Factor Discovery via GF(2) Matrix Powers**: Using companion matrix orders over GF(2) to factor Mersenne numbers. The computation is entirely over GF(2) — each step is XOR of bit vectors. While the mathematical result (ord(C^d) = M_p/gcd(M_p,d)) follows from group theory, the APPLICATION to factoring via GF(2) matrix dynamics appears to be new.

2. **Cross-Correlation Factor Detection (Kasami-based)**: First application of Kasami's cross-correlation theorem to detect factors of M_p without integer arithmetic. The cross-correlation between an m-sequence and its d-decimation takes exactly 3 values iff d | M_p. This is a genuinely non-circular factor detection method.

3. **Irreducibility-Primitivity Equivalence Theorem**: The explicit statement and proof that every irreducible polynomial of degree p over GF(2) is primitive if and only if M_p = 2^p - 1 is prime. This appears to be a new explicit theorem, though it follows from well-known results in finite field theory.

### Tier 2: Novel Combinations/Reformulations

4. **Minimal Polynomial of α^q**: The constructive procedure for building non-primitive irreducible polynomials whose companion matrix orders reveal factors. The individual components are known, but the explicit factor-extraction procedure appears to be new.

5. **LLT as Game of Life Circuit**: Implementing the Lucas-Lehmer Test as a logic circuit in Conway's Game of Life is a novel visualization/demonstration, though the mathematical content is equivalent to the standard LLT.

### Tier 3: Known Results / Negative Results

6. **Spectral Analysis Negative Result**: Our rigorous demonstration that WHT/FFT/autocorrelation of m-sequences cannot distinguish prime from composite period is a confirmation of well-known properties, but the explicit experimental verification and honest negative assessment may be useful for the community.

## Circularity Assessment

| Method | Circularity | Notes |
|--------|-------------|-------|
| C^d order probing | PARTIAL | gcd(M_p, d) check ≈ trial division; but GF(2) computation is novel |
| Minimal polynomial construction | PARTIAL | Proves factor info is in CA dynamics; but requires knowing q |
| Cross-correlation detection | LOW | Uses Kasami theorem; no integer arithmetic with M_p |
| WHT/FFT spectral analysis | N/A | Negative result; does not work |
| Pure CA orbit detection | LOW | Only GF(2) matrix-vector ops; orbit counting is the only integer op |
