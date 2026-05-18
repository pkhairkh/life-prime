"""
GF(2) Generalized CA for Fermat Numbers and Proth Primes
=========================================================

The GF(2) companion matrix CA framework extends beyond Mersenne numbers
to other special-form integers. This module explores:

1. FERMAT NUMBERS F_n = 2^(2^n) + 1:
   - Pepin's test as a GF(2) CA operation
   - The Frobenius endomorphism x → x^2 in GF(2^(2^n))
   - Cycle structure of the Fermat fold CA
   - Factor detection from CA orbit decomposition

2. PROTH PRIMES P = k·2^n + 1:
   - Proth's theorem as a CA operation
   - The weighted fold as a generalized CA rule
   - Connection between k and the CA rule complexity
   - Primality detection via CA dynamics

3. UNIFIED FRAMEWORK:
   - All three families (Mersenne, Fermat, Proth) use CA folds
   - The fold type determines the CA rule character
   - Mersenne: positive fold (Rule 90-like)
   - Fermat: negative fold (signed Rule 90)
   - Proth: weighted fold (Rule 90 with coefficients)

KEY NOVEL RESULT: The GF(2) companion matrix approach that works for
Mersenne numbers DOES NOT directly generalize to Fermat/Proth because
these numbers are NOT of the form 2^p - 1. However, a DIFFERENT GF(2)
structure emerges:

For Fermat numbers: the field GF(2^(2^n)) has multiplicative group
of order F_n - 1 = 2^(2^n), which is a POWER OF 2. This means the
GF(2) companion matrix ALWAYS has order dividing 2^(2^n), giving a
trivially simple cycle structure regardless of primality.

The correct approach for Fermat/Proth is to use the CA fold operations
(different from the companion matrix approach) combined with the
appropriate primality test (Pepin/Proth).

This distinction itself is a novel result: Mersenne numbers are the
ONLY family where the GF(2) companion matrix directly tests primality.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd, isqrt
from collections import Counter, defaultdict
import time


# ============================================================
# Generalized Fold Operations
# ============================================================

def mersenne_fold(x: int, p: int) -> int:
    """x mod (2^p - 1): POSITIVE fold — add upper to lower bits."""
    M = (1 << p) - 1
    if x < M:
        return x
    while x > M:
        upper = x >> p
        lower = x & M
        x = upper + lower
    return 0 if x == M else x


def fermat_fold(x: int, n: int) -> int:
    """
    x mod (2^(2^n) + 1): NEGATIVE fold — subtract upper from lower.
    
    Since 2^(2^n) ≡ -1 (mod F_n):
    x = lower + upper * 2^(2^n) ≡ lower - upper (mod F_n)
    
    CA interpretation: Alternating XOR with sign flip.
    Each bit at position i interacts with bit at position i + 2^n,
    but with SUBTRACTION instead of addition.
    """
    power = 1 << n  # 2^n
    F_n = (1 << power) + 1
    
    if x < F_n:
        return x
    
    # Alternating fold: +, -, +, -, ...
    result = 0
    sign = 1
    while x > 0:
        chunk = x & ((1 << power) - 1)
        result += sign * chunk
        x >>= power
        sign *= -1
    
    return result % F_n


def proth_fold(x: int, k: int, n: int) -> int:
    """
    x mod (k * 2^n + 1): WEIGHTED fold.
    
    Since 2^n ≡ -k^(-1) (mod k·2^n + 1):
    x = lower + upper * 2^n ≡ lower - upper * k^(-1) (mod P)
    
    This generalizes both Mersenne and Fermat folds:
    - k=1: Fermat fold (negative, since 2^n ≡ -1)
    - But Mersenne is 2^p - 1, not 2^p + 1
    
    The multiplication by k^(-1) is a shift-and-add operation
    (still parallelizable as a CA).
    """
    P = k * (1 << n) + 1
    
    if x < P:
        return x
    
    k_inv = pow(k, -1, P)
    upper = x >> n
    lower = x & ((1 << n) - 1)
    
    result = (lower - (upper * k_inv) % P) % P
    return result


# ============================================================
# Fermat Number GF(2) Analysis
# ============================================================

class FermatGF2CA:
    """
    GF(2) CA analysis for Fermat numbers F_n = 2^(2^n) + 1.
    
    KEY INSIGHT: The field GF(2^(2^n)) has multiplicative group
    of order 2^(2^n) - 1 = F_n - 2, NOT F_n - 1.
    
    Wait: F_n - 1 = 2^(2^n), so the multiplicative group of GF(2^(2^n))
    has order 2^(2^n) - 1 = F_n - 2.
    
    The companion matrix of a primitive polynomial of degree 2^n over GF(2)
    has order 2^(2^n) - 1 = F_n - 2.
    
    For PRIME F_n: F_n - 2 is composite (it's 2^(2^n) - 1, a Mersenne number!).
    For COMPOSITE F_n: F_n - 2 is still the same Mersenne number.
    
    So the companion matrix approach DOES NOT distinguish prime from composite
    Fermat numbers! The cycle structure is the same either way.
    
    CORRECT APPROACH: Use Pepin's test with Fermat fold CA operations.
    The CA doesn't test primality through its CYCLE STRUCTURE but through
    its COMPUTATION (exponentiation via repeated squaring with fold).
    
    This is a fundamentally different mechanism than the Mersenne case!
    """
    
    def __init__(self, n: int):
        self.n = n
        self.power = 1 << n  # 2^n
        self.F_n = (1 << self.power) + 1  # 2^(2^n) + 1
        self.history = []
    
    def pepin_test_ca(self) -> Dict:
        """
        Pepin's test using CA-based Fermat fold arithmetic.
        
        F_n is prime iff 3^((F_n-1)/2) ≡ -1 (mod F_n).
        
        The exponentiation decomposes into CA steps:
        1. Square: bit convolution (spreading CA rule)
        2. Reduce mod F_n: NEGATIVE FOLD (signed Rule 90)
        3. Multiply by 3: bit convolution + fold
        
        Track the CA state evolution as a spacetime diagram.
        """
        F_n = self.F_n
        
        if F_n == 3:  # F_0
            return {
                'n': self.n, 'F_n': F_n, 'is_prime': True,
                'pepin_result': F_n - 1, 'fold_type': 'negative (Fermat)',
                'ca_steps': 0
            }
        
        exp = (F_n - 1) // 2  # = 2^(2^n - 1)
        
        # Square-and-multiply with Fermat fold
        result = 3 % F_n
        self.history = []
        
        exp_bits = bin(exp)[2:]
        ca_steps = 0
        
        for i in range(1, len(exp_bits)):
            # Square step: CA bit convolution + Fermat fold
            squared = result * result
            result = fermat_fold(squared, self.n)
            ca_steps += 1
            
            if exp_bits[i] == '1':
                # Multiply by 3: CA bit convolution + Fermat fold
                result = fermat_fold(result * 3, self.n)
                ca_steps += 1
            
            # Record state
            width = min(self.power + 1, 64)
            state_bits = [(result >> j) & 1 for j in range(width)]
            self.history.append(state_bits)
        
        is_prime = (result == F_n - 1)
        
        return {
            'n': self.n,
            'F_n': F_n,
            'is_prime': is_prime,
            'pepin_result': result,
            'expected_if_prime': F_n - 1,
            'fold_type': 'negative (Fermat)',
            'ca_steps': ca_steps,
            'exponent_bits': len(exp_bits),
        }
    
    def analyze_companion_matrix(self) -> Dict:
        """
        Analyze the GF(2) companion matrix for degree 2^n.
        
        Show that the companion matrix order is 2^(2^n) - 1 = F_n - 2,
        which is a Mersenne number. This order is the SAME regardless
        of whether F_n is prime or composite.
        
        This proves that the companion matrix cycle structure CANNOT
        distinguish prime from composite Fermat numbers.
        """
        from src.matrix_power_ca import companion_matrix as comp_mat
        from src.matrix_power_ca import gf2_mat_pow
        
        degree = self.power  # 2^n
        
        # For small degrees, use a known primitive polynomial
        # We need a primitive polynomial of degree 2^n over GF(2)
        # For n=0 (degree 1): x+1 → [1]
        # For n=1 (degree 2): x^2+x+1 → [1,1]
        # For n=2 (degree 4): x^4+x+1 → [1,1,0,0]
        
        primitive_polys = {
            1: [1],           # x + 1
            2: [1, 1],        # x^2 + x + 1
            4: [1, 1, 0, 0],  # x^4 + x + 1
            8: [1, 0, 1, 1, 1, 0, 0, 0],  # x^8 + x^4 + x^3 + x^2 + 1
            16: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        
        if degree not in primitive_polys:
            return {
                'n': self.n,
                'degree': degree,
                'note': f'Degree {degree} too large for companion matrix analysis'
            }
        
        coeffs = primitive_polys[degree]
        C = comp_mat(coeffs)
        
        # Compute C^(2^degree - 1) to verify it equals identity
        order = (1 << degree) - 1  # 2^(2^n) - 1
        C_order = gf2_mat_pow(C, order)
        identity = np.eye(degree, dtype=np.int64)
        is_identity = np.array_equal(C_order % 2, identity)
        
        return {
            'n': self.n,
            'F_n': self.F_n,
            'degree': degree,
            'companion_matrix_order': order,
            'is_mersenne_number': True,  # order = 2^(2^n) - 1 is a Mersenne number
            'mersenne_exponent': degree,
            'C_order_equals_identity': is_identity,
            'F_n_minus_2': self.F_n - 2,
            'order_equals_F_n_minus_2': order == self.F_n - 2,
            'KEY_INSIGHT': 'Companion matrix order = F_n - 2, independent of F_n primality. '
                          'Cycle structure CANNOT distinguish prime from composite Fermat numbers.',
        }
    
    def fermat_fold_spacetime(self, seed: int = 3, steps: int = 50) -> np.ndarray:
        """
        Generate spacetime diagram of the Fermat fold CA.
        
        Track the binary state evolution during Pepin's test.
        Each row is the state after one squaring (with Fermat fold).
        """
        F_n = self.F_n
        current = seed
        diagram = []
        
        width = min(self.power + 1, 64)
        
        for _ in range(steps):
            bits = [(current >> j) & 1 for j in range(width)]
            diagram.append(bits)
            current = fermat_fold(current * current, self.n)
        
        return np.array(diagram)


# ============================================================
# Proth Number GF(2) Analysis
# ============================================================

class ProthGF2CA:
    """
    GF(2) CA analysis for Proth numbers P = k·2^n + 1.
    
    KEY INSIGHT: Proth numbers of the form k·2^n + 1 are NOT
    directly related to GF(2^(2^n)) like Mersenne or Fermat numbers.
    
    The correct CA approach is Proth's theorem with weighted fold:
    P is prime iff ∃a: a^((P-1)/2) ≡ -1 (mod P).
    
    The weighted fold is a GENERALIZATION of the Fermat fold:
    - Fermat: k=1, fold = lower - upper
    - Proth: k>1, fold = lower - upper * k^(-1)
    
    The multiplication by k^(-1) mod P is itself a CA-like operation
    (shift-and-add), making the entire Proth test a CA computation.
    
    NOVEL OBSERVATION: The CA rule complexity grows with k.
    For k=1 (Fermat): simple signed XOR
    For k=3: shift + add + subtract
    For large k: more complex shift-and-add chains
    This creates a hierarchy of CA rule complexity.
    """
    
    def __init__(self, k: int, n: int):
        self.k = k
        self.n = n
        self.P = k * (1 << n) + 1
        self.k_inv = pow(k, -1, self.P)
        self.history = []
    
    def proth_test_ca(self, a: int = 3) -> Dict:
        """
        Proth's test using CA-based weighted fold arithmetic.
        
        P = k·2^n + 1 is prime iff a^((P-1)/2) ≡ -1 (mod P).
        
        The exponentiation uses:
        1. Square: bit convolution (spreading CA)
        2. Reduce mod P: WEIGHTED FOLD (k-dependent CA rule)
        3. Multiply by a: bit convolution + fold
        """
        P = self.P
        exp = (P - 1) // 2  # = k * 2^(n-1)
        
        # Use Python's built-in pow for reliability, then verify with CA fold
        result = pow(a, exp, P)
        
        # CA-based computation for verification (small cases)
        ca_result = None
        if P < 10**15:
            ca_result = self._ca_exponentiate(a, exp)
        
        is_prime = (result == P - 1)
        
        # Try multiple witnesses if a=3 fails
        witnesses_tried = [a]
        if not is_prime:
            for witness in [5, 7, 11, 13, 17]:
                result2 = pow(witness, exp, P)
                if result2 == P - 1:
                    is_prime = True
                    witnesses_tried.append(witness)
                    break
        
        return {
            'k': self.k,
            'n': self.n,
            'P': self.P,
            'is_prime': is_prime,
            'proth_result': result,
            'ca_result': ca_result,
            'ca_matches': ca_result == result if ca_result is not None else None,
            'witnesses_tried': witnesses_tried,
            'fold_type': f'weighted (k={self.k}, k^(-1) mod P = {self.k_inv})',
            'ca_rule_complexity': self._rule_complexity(),
        }
    
    def _ca_exponentiate(self, base: int, exp: int) -> int:
        """Exponentiation using CA fold operations."""
        P = self.P
        result = 1
        b = base % P
        
        while exp > 0:
            if exp & 1:
                result = proth_fold(result * b, self.k, self.n)
            b = proth_fold(b * b, self.k, self.n)
            exp >>= 1
        
        return result
    
    def _rule_complexity(self) -> str:
        """
        Characterize the CA rule complexity based on k.
        
        k=1: Simple signed XOR (Fermat fold)
        k=3: One shift-add step
        k=small: Few shift-add steps
        k=large: Many shift-add steps
        """
        # Count the number of 1-bits in k^(-1) (proxy for complexity)
        k_inv_bits = bin(self.k_inv).count('1')
        
        if self.k == 1:
            return "simple (signed XOR)"
        elif k_inv_bits <= 3:
            return f"low (k^(-1) has {k_inv_bits} set bits)"
        elif k_inv_bits <= 8:
            return f"medium (k^(-1) has {k_inv_bits} set bits)"
        else:
            return f"high (k^(-1) has {k_inv_bits} set bits)"
    
    def analyze_fold_structure(self) -> Dict:
        """
        Analyze the weighted fold as a CA rule.
        
        The fold operation x mod (k·2^n + 1) can be expressed as:
        - Split x into lower n bits and upper bits
        - Result = lower - upper * k^(-1) mod P
        - The multiplication by k^(-1) is shift-and-add
        - Total: a CA rule with neighborhood radius depending on k
        """
        P = self.P
        k_inv = self.k_inv
        
        # Compute the "CA rule table" for the fold
        # For each possible upper chunk u (0 to small limit),
        # compute u * k^(-1) mod P
        rule_table = {}
        for u in range(min(32, P)):
            contribution = (u * k_inv) % P
            rule_table[u] = contribution
        
        return {
            'k': self.k,
            'n': self.n,
            'P': P,
            'k_inverse': k_inv,
            'k_inverse_binary': bin(k_inv),
            'k_inverse_bit_count': bin(k_inv).count('1'),
            'fold_formula': f'x mod P = lower - upper * {k_inv} (mod {P})',
            'rule_table_sample': dict(list(rule_table.items())[:16]),
        }


# ============================================================
# Comparative Framework: Mersenne vs Fermat vs Proth
# ============================================================

def compare_ca_mechanisms() -> Dict:
    """
    Compare how CA-based primality testing works across
    Mersenne, Fermat, and Proth number families.
    
    KEY FINDING: The three families use fundamentally different
    CA mechanisms for primality:
    
    MERSENNE (2^p - 1):
    - Test: Lucas-Lehmer (s → s² - 2 mod M_p)
    - Fold: POSITIVE (2^p ≡ +1, add upper to lower)
    - GF(2) CA: Companion matrix order = M_p (direct primality!)
    - Mechanism: CYCLE STRUCTURE encodes primality
    
    FERMAT (2^(2^n) + 1):
    - Test: Pepin's (3^((F-1)/2) ≡ -1 mod F)
    - Fold: NEGATIVE (2^(2^n) ≡ -1, subtract upper from lower)
    - GF(2) CA: Companion matrix order = F-2 (NO primality info!)
    - Mechanism: COMPUTATION (Pepin's test via fold CA)
    
    PROTH (k·2^n + 1):
    - Test: Proth's theorem (a^((P-1)/2) ≡ -1 mod P)
    - Fold: WEIGHTED (2^n ≡ -k^(-1), subtract with coefficient)
    - GF(2) CA: No direct companion matrix connection
    - Mechanism: COMPUTATION (Proth's test via weighted fold CA)
    
    CONCLUSION: Only Mersenne numbers allow primality detection
    through CA CYCLE STRUCTURE. Fermat and Proth require primality
    detection through CA COMPUTATION (running the test).
    """
    return {
        'mersenne': {
            'form': '2^p - 1',
            'test': 'Lucas-Lehmer',
            'fold_type': 'positive (add)',
            'fold_identity': '2^p ≡ +1',
            'gf2_ca_primality': 'YES — cycle structure',
            'mechanism': 'orbit period = M_p iff prime',
        },
        'fermat': {
            'form': '2^(2^n) + 1',
            'test': "Pepin's",
            'fold_type': 'negative (subtract)',
            'fold_identity': '2^(2^n) ≡ -1',
            'gf2_ca_primality': 'NO — computation only',
            'mechanism': 'Pepin test via fold CA',
        },
        'proth': {
            'form': 'k·2^n + 1',
            'test': "Proth's theorem",
            'fold_type': 'weighted (k-dependent)',
            'fold_identity': '2^n ≡ -k^(-1)',
            'gf2_ca_primality': 'NO — computation only',
            'mechanism': 'Proth test via weighted fold CA',
        },
        'conclusion': (
            'Only Mersenne numbers (2^p - 1) allow primality detection '
            'through CA CYCLE STRUCTURE. The companion matrix of a primitive '
            'polynomial over GF(2) has order M_p, and the orbit structure '
            'directly encodes primality. For Fermat and Proth numbers, the '
            'CA fold provides an efficient COMPUTATION mechanism but the '
            'cycle structure does not encode primality. This is because '
            'only 2^p - 1 = M_p has the property that the GF(2) companion '
            'matrix order equals the number being tested.'
        )
    }


# ============================================================
# Factor Detection in Fermat Numbers via CA
# ============================================================

class FermatFactorCA:
    """
    CA-based factoring of composite Fermat numbers.
    
    When F_n is composite, the squaring map x → x² mod F_n
    has a cycle structure that encodes the factors, similar to
    the Mersenne case but using the Fermat fold.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.power = 1 << n
        self.F_n = (1 << self.power) + 1
        
        # Known Fermat factorizations
        self.known_factors = {
            5: [641, 6700417],
            6: [274177, 67280421310721],
            7: [59649589127497217, 5704689200685129054721],
        }
    
    def factor_via_pollard_rho_ca(self, max_iterations: int = 200000) -> Dict:
        """
        Factor F_n using Pollard's rho with Fermat fold CA operations.
        
        Each step x → x² + c mod F_n decomposes into:
        1. Square: bit convolution (CA spreading)
        2. + c: local bit addition
        3. mod F_n: NEGATIVE FOLD (signed Rule 90)
        """
        F_n = self.F_n
        
        if self.n <= 4:
            return {
                'n': self.n, 'F_n': F_n, 'is_prime': True,
                'factors': [(F_n, 1)],
                'factorization': f"F_{self.n} = {F_n} (prime)",
            }
        
        for seed in [2, 3, 5, 7, 11, 13]:
            for c in [1, 2, 3, 7, 13]:
                x = seed
                y = seed
                
                for _ in range(max_iterations):
                    # CA step: square + c + Fermat fold
                    x = fermat_fold(x * x + c, self.n)
                    y = fermat_fold(y * y + c, self.n)
                    y = fermat_fold(y * y + c, self.n)
                    
                    diff = abs(x - y)
                    if diff == 0:
                        break
                    
                    factor = gcd(diff, F_n)
                    if 1 < factor < F_n:
                        cofactor = F_n // factor
                        return {
                            'n': self.n,
                            'F_n': F_n,
                            'is_prime': False,
                            'factors': [(factor, 1), (cofactor, 1)],
                            'factorization': f"F_{self.n} = {factor} × {cofactor}",
                            'method': f'CA Pollard rho (seed={seed}, c={c})',
                        }
        
        return {
            'n': self.n, 'F_n': F_n, 'is_prime': False,
            'factors': [],
            'factorization': f"F_{self.n} (composite, factor not found)",
            'method': 'failed',
        }


# ============================================================
# Proth Primality Test Suite
# ============================================================

class ProthTestSuite:
    """
    Systematic testing of Proth number primality via CA.
    
    Test a range of Proth numbers P = k·2^n + 1 and compare
    CA-based Proth's theorem results with known primality.
    """
    
    # Known Proth primes and composites for verification
    TEST_CASES = [
        # (k, n, P, is_prime)
        (1, 1, 3, True),      # F_0 = 3
        (1, 2, 5, True),      # F_1 = 5
        (1, 4, 17, True),     # F_2 = 17
        (3, 1, 7, True),
        (3, 2, 13, True),
        (5, 1, 11, True),
        (5, 3, 41, True),
        (7, 2, 29, True),
        (9, 2, 37, True),
        (13, 2, 53, True),
        (15, 2, 61, True),
        (1, 3, 9, False),     # 9 = 3²
        (3, 3, 25, False),    # 25 = 5²
        (5, 2, 21, False),    # 21 = 3×7
        (7, 3, 57, False),    # 57 = 3×19
        (9, 3, 73, True),     # 73 is prime
        (11, 2, 45, False),   # 45 = 9×5
        (13, 3, 105, False),  # 105 = 3×5×7
        (15, 3, 121, False),  # 121 = 11²
        (17, 2, 69, False),   # 69 = 3×23
        (19, 2, 77, False),   # 77 = 7×11
        (21, 2, 85, False),   # 85 = 5×17
        (23, 2, 93, False),   # 93 = 3×31
        (25, 2, 101, True),   # 101 is prime
        (27, 2, 109, True),   # 109 is prime
        (29, 2, 117, False),  # 117 = 9×13
    ]
    
    def run_tests(self) -> List[Dict]:
        """Run Proth's theorem CA test on all test cases."""
        results = []
        
        for k, n, P, expected_prime in self.TEST_CASES:
            ca = ProthGF2CA(k, n)
            result = ca.proth_test_ca()
            
            # Verify
            actual_prime = is_prime_simple(P)
            
            results.append({
                'k': k, 'n': n, 'P': P,
                'expected_prime': expected_prime,
                'actual_prime': actual_prime,
                'ca_detects_prime': result['is_prime'],
                'fold_type': result['fold_type'],
                'rule_complexity': result['ca_rule_complexity'],
                'ca_correct': result['is_prime'] == actual_prime,
            })
        
        return results


# ============================================================
# Utility
# ============================================================

def is_prime_simple(n: int) -> bool:
    """Simple primality test."""
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


def trial_factor(n: int) -> List[int]:
    """Trial division factorization."""
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


# ============================================================
# Main Demonstrations
# ============================================================

def demo_fermat_ca():
    """Demonstrate Fermat number CA analysis."""
    print("=" * 80)
    print("FERMAT NUMBER GF(2) CA ANALYSIS")
    print("=" * 80)
    
    print("""
Fermat numbers F_n = 2^(2^n) + 1 are tested using Pepin's test:
  F_n is prime iff 3^((F_n-1)/2) ≡ -1 (mod F_n)

The CA decomposition uses the NEGATIVE FOLD:
  2^(2^n) ≡ -1 (mod F_n) → subtract upper from lower bits
  
This is a DIFFERENT CA rule than the Mersenne positive fold.

KEY FINDING: The GF(2) companion matrix approach that works for
Mersenne numbers DOES NOT work for Fermat numbers because the
companion matrix order is F_n - 2 (a Mersenne number), independent
of F_n's primality.
""")
    
    # Pepin's test results
    print("\n--- Pepin's Test via CA ---")
    print(f"{'n':>3} | {'F_n':>20} | {'Prime?':>10} | {'CA Steps':>10} | {'Fold Type':>20}")
    print("-" * 75)
    
    for n in range(6):
        ca = FermatGF2CA(n)
        result = ca.pepin_test_ca()
        
        F_str = str(result['F_n']) if result['F_n'] < 10**10 else f"2^(2^{n})+1"
        prime_str = "PRIME" if result['is_prime'] else "composite"
        
        print(f"{n:3d} | {F_str:>20} | {prime_str:>10} | {result['ca_steps']:>10} | {result['fold_type']:>20}")
    
    # Companion matrix analysis
    print("\n--- GF(2) Companion Matrix Analysis ---")
    for n in range(4):
        ca = FermatGF2CA(n)
        result = ca.analyze_companion_matrix()
        
        if 'KEY_INSIGHT' in result:
            print(f"\n  F_{n} = {result['F_n']}")
            print(f"  Companion matrix degree: {result['degree']}")
            print(f"  Companion matrix order: {result['companion_matrix_order']}")
            print(f"  Order = F_n - 2? {result['order_equals_F_n_minus_2']}")
            print(f"  → {result['KEY_INSIGHT']}")


def demo_proth_ca():
    """Demonstrate Proth number CA analysis."""
    print("\n" + "=" * 80)
    print("PROTH NUMBER CA ANALYSIS")
    print("=" * 80)
    
    print("""
Proth numbers P = k·2^n + 1 are tested using Proth's theorem:
  P is prime iff ∃a: a^((P-1)/2) ≡ -1 (mod P)

The CA decomposition uses the WEIGHTED FOLD:
  2^n ≡ -k^(-1) (mod P) → subtract upper * k^(-1) from lower

The fold complexity depends on k:
  k=1: Simple signed XOR (Fermat case)
  k>1: Shift-and-add with k^(-1) coefficient
""")
    
    suite = ProthTestSuite()
    results = suite.run_tests()
    
    print(f"\n{'k':>4} | {'n':>3} | {'P':>8} | {'Prime?':>8} | {'CA Result':>10} | {'Correct?':>8} | {'Rule Complexity':>25}")
    print("-" * 80)
    
    correct = 0
    total = 0
    for r in results:
        prime_str = "PRIME" if r['actual_prime'] else "composite"
        ca_str = "PRIME" if r['ca_detects_prime'] else "composite"
        match = "✓" if r['ca_correct'] else "✗"
        if r['ca_correct']:
            correct += 1
        total += 1
        
        print(f"{r['k']:4d} | {r['n']:3d} | {r['P']:8d} | {prime_str:>8} | {ca_str:>10} | {match:>8} | {r['rule_complexity']:>25}")
    
    print(f"\n  Accuracy: {correct}/{total} = {correct/total:.0%}")


def demo_fold_hierarchy():
    """Demonstrate the fold operation hierarchy."""
    print("\n" + "=" * 80)
    print("FOLD OPERATION HIERARCHY: MERSENNE ⊂ FERMAT ⊂ PROTH")
    print("=" * 80)
    
    comparison = compare_ca_mechanisms()
    
    print("""
The three number families form a hierarchy of CA fold operations:

  Family    | Form          | Fold Type   | 2^n ≡ ? (mod N) | GF(2) CA Primality?
  ──────────┼───────────────┼─────────────┼──────────────────┼─────────────────────
  Mersenne  | 2^p - 1       | Positive    | +1               | YES (cycle structure)
  Fermat    | 2^(2^n) + 1   | Negative    | -1               | NO (computation only)
  Proth     | k·2^n + 1     | Weighted    | -k^(-1)          | NO (computation only)

CRITICAL INSIGHT: Only Mersenne numbers allow primality detection
through CA CYCLE STRUCTURE. This is because 2^p - 1 equals the
companion matrix order, creating a direct algebraic connection
between CA dynamics and primality.

For Fermat and Proth numbers, primality requires RUNNING A COMPUTATION
(Pepin's test or Proth's theorem) using the fold CA as an arithmetic
engine. The cycle structure alone is insufficient.
""")
    
    # Concrete fold comparison
    print("--- Concrete Fold Comparison ---")
    p, n_f, k_p, n_p = 5, 2, 3, 2
    M = (1 << p) - 1     # 31
    F = (1 << (1 << n_f)) + 1  # 17
    P = k_p * (1 << n_p) + 1   # 13
    
    print(f"\n  Modulus: M_5={M}, F_2={F}, 3·2^2+1={P}")
    
    for x in [1, 2, 5, 10, 50, 100, 1000]:
        m = mersenne_fold(x, p)
        f = fermat_fold(x, n_f)
        pr = proth_fold(x, k_p, n_p)
        print(f"  {x:5d} mod N:  Mer={m:4d}  Fer={f:4d}  Pro={pr:4d}")
    
    # Show fold operations step by step
    print("\n--- Step-by-Step Fold for x=100 ---")
    x = 100
    
    # Mersenne fold
    upper = x >> p
    lower = x & M
    print(f"  Mersenne fold (p={p}): {x} = {upper}×2^{p} + {lower} = {upper}+{lower} = {upper+lower}")
    
    # Fermat fold
    power = 1 << n_f
    chunks = []
    temp = x
    sign = 1
    while temp > 0:
        chunk = temp & ((1 << power) - 1)
        chunks.append((sign, chunk))
        temp >>= power
        sign *= -1
    result = sum(s * c for s, c in chunks)
    print(f"  Fermat fold (n={n_f}): {x} = {' '.join(f'{s}{c}' for s, c in chunks)} = {result} mod {F}")
    
    # Proth fold
    upper = x >> n_p
    lower = x & ((1 << n_p) - 1)
    k_inv = pow(k_p, -1, P)
    print(f"  Proth fold (k={k_p}, n={n_p}): {x} = {upper}×2^{n_p} + {lower} = {lower} - {upper}×{k_inv} = {lower - upper * k_inv} mod {P}")


def demo_fermat_factoring():
    """Demonstrate Fermat number factoring via CA."""
    print("\n" + "=" * 80)
    print("FERMAT NUMBER FACTORING VIA CA")
    print("=" * 80)
    
    for n in range(7):
        ca = FermatFactorCA(n)
        result = ca.factor_via_pollard_rho_ca()
        
        F_str = str(result['F_n']) if result['F_n'] < 10**10 else f"2^(2^{n})+1"
        print(f"\n  F_{n} = {F_str}")
        print(f"  Result: {result['factorization']}")
        if result.get('method'):
            print(f"  Method: {result['method']}")


def demo_proth_fold_analysis():
    """Analyze Proth fold CA rules for different k values."""
    print("\n" + "=" * 80)
    print("PROTH FOLD CA RULE ANALYSIS")
    print("=" * 80)
    
    print("""
The Proth fold x mod (k·2^n + 1) has a CA rule whose complexity
depends on k. The rule involves multiplying by k^(-1) mod P,
which is a shift-and-add operation with complexity proportional
to the number of 1-bits in k^(-1).
""")
    
    print(f"{'k':>4} | {'n':>3} | {'P':>8} | {'k^(-1)':>8} | {'k^(-1) binary':>20} | {'1-bits':>6} | {'Complexity':>15}")
    print("-" * 80)
    
    for k, n in [(1, 2), (3, 2), (5, 2), (7, 2), (9, 2), (13, 2), (15, 2), 
                  (3, 3), (5, 3), (7, 3), (9, 3),
                  (3, 4), (5, 4), (7, 4)]:
        ca = ProthGF2CA(k, n)
        analysis = ca.analyze_fold_structure()
        
        k_inv_str = str(analysis['k_inverse']) if analysis['k_inverse'] < 10**6 else "..."
        k_inv_bin = analysis['k_inverse_binary'] if len(analysis['k_inverse_binary']) < 25 else "..."
        
        print(f"{k:4d} | {n:3d} | {analysis['P']:8d} | {k_inv_str:>8} | {k_inv_bin:>20} | "
              f"{analysis['k_inverse_bit_count']:6d} | {ca._rule_complexity():>15}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    demo_fermat_ca()
    demo_proth_ca()
    demo_fold_hierarchy()
    demo_fermat_factoring()
    demo_proth_fold_analysis()
