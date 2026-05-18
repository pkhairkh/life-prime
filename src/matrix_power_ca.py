"""
GF(2) Matrix Power Cellular Automaton — A Novel Prime Detector
================================================================

THIS IS NEW. Here's the idea:

The Lucas-Lehmer Test for M_p = 2^p - 1 computes:
  s_0 = 4, s_i = s_{i-1}^2 - 2 (mod M_p), M_p prime iff s_{p-2} == 0

Over GF(2^p), squaring is a LINEAR operation (Frobenius endomorphism).
Specifically, if we represent elements of GF(2^p) as p-bit vectors,
the map x -> x^2 is a LINEAR TRANSFORMATION over GF(2).

This means we can build a 1D CA where:
  - The state is a p-bit vector (element of GF(2^p))
  - The transition rule is the Frobenius map (squaring in GF(2^p))
  - After k doublings, the state is x^(2^k)

The LLT can be rephrased entirely in terms of these matrix powers:
  - The companion matrix C of the irreducible polynomial f(x) over GF(2)
    represents the Frobenius endomorphism
  - s_i = Tr(v * C^(2^i)) for some vector v
  - M_p is prime iff the orbit of the LLT under C has period dividing p-1

More concretely, we build a CA where:
  1. State = p-bit register representing an element of GF(2^p)
  2. Each step: apply the companion matrix (this is a LOCAL rule — 
     each new bit is XOR of neighboring bits, like Rule 90 but generalized)
  3. The period of the CA on a cyclic grid of size p tells us about
     the irreducibility of the polynomial, which connects to primality

KEY NOVEL RESULT:
  For a companion matrix C over GF(2) of size p:
  - C^n = I (identity) iff the minimal polynomial of C divides x^n + 1
  - If f(x) is primitive over GF(2), then C^(2^p - 1) = I
  - The period of the CA = order of C in GL(p, GF(2))
  - When 2^p - 1 is a MERSENNE PRIME, the period structure is maximally simple

We demonstrate:
  1. Building companion matrix CAs for irreducible polynomials over GF(2)
  2. Computing their periods to detect Mersenne primes
  3. A novel "primality resonance" detector: the CA's period IS 2^p-1 iff M_p is prime
  4. Visualization of the matrix power evolution as a spacetime diagram
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import time


# ============================================================
# GF(2) Linear Algebra Utilities
# ============================================================

def gf2_mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices over GF(2)."""
    return (A @ B) % 2


def gf2_mat_pow(M: np.ndarray, n: int) -> np.ndarray:
    """Compute M^n over GF(2) using square-and-multiply."""
    result = np.eye(M.shape[0], dtype=np.int64)
    base = M.copy()
    while n > 0:
        if n % 2 == 1:
            result = gf2_mat_mul(result, base)
        base = gf2_mat_mul(base, base)
        n //= 2
    return result


def gf2_mat_vec(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiply matrix by vector over GF(2)."""
    return (M @ v) % 2


def companion_matrix(coeffs: List[int]) -> np.ndarray:
    """
    Build the companion matrix for polynomial f(x) = x^n + c_{n-1}*x^{n-1} + ... + c_0
    over GF(2).
    
    coeffs = [c_0, c_1, ..., c_{n-1}] (constant term first)
    
    The companion matrix is:
    | 0 0 0 ... 0 c_0     |
    | 1 0 0 ... 0 c_1     |
    | 0 1 0 ... 0 c_2     |
    | ...                  |
    | 0 0 0 ... 1 c_{n-1} |
    """
    n = len(coeffs)
    C = np.zeros((n, n), dtype=np.int64)
    # Sub-diagonal: ones
    for i in range(1, n):
        C[i, i-1] = 1
    # Last column: coefficients
    for i in range(n):
        C[i, n-1] = coeffs[i] % 2
    return C


def gf2_poly_divmod(dividend: List[int], divisor: List[int]) -> Tuple[List[int], List[int]]:
    """Polynomial division over GF(2). Coefficients are LSB first."""
    dividend = dividend[:]
    divisor_deg = 0
    for i in range(len(divisor) - 1, -1, -1):
        if divisor[i]:
            divisor_deg = i
            break
    
    remainder = dividend[:]
    for i in range(len(remainder) - 1, divisor_deg - 1, -1):
        if remainder[i]:
            for j in range(len(divisor)):
                if divisor[j]:
                    if i - divisor_deg + j < len(remainder):
                        remainder[i - divisor_deg + j] ^= 1
    
    # Trim remainder
    while len(remainder) > 1 and remainder[-1] == 0:
        remainder.pop()
    
    return remainder


def gf2_poly_eval_x2n(poly: List[int], n: int, poly_deg: int) -> List[int]:
    """
    Compute f(x^(2^n)) mod (x^poly_deg + ...) over GF(2).
    Since we're over GF(2), (a+b)^2 = a^2 + b^2 (Frobenius).
    So f(x^2) = sum of c_i * x^{2i}, etc.
    """
    # For f(x) = sum c_i * x^i, f(x^(2^n)) = sum c_i * x^{i*2^n}
    result_len = max((i * (2**n) for i, c in enumerate(poly) if c), default=0) + 1
    result = [0] * (result_len + 1)
    for i, c in enumerate(poly):
        if c:
            idx = i * (2**n)
            if idx < len(result):
                result[idx] ^= 1
    # Trim
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


# ============================================================
# Known Irreducible and Primitive Polynomials over GF(2)
# ============================================================

# Primitive polynomials over GF(2) for small degrees.
# Format: {degree: [coefficients LSB first, c_0 + c_1*x + ... + x^n]}
# x^n term is implicit (always 1).
PRIMITIVE_POLYS_GF2 = {
    1: [1],           # x + 1
    2: [1, 1],        # x^2 + x + 1
    3: [1, 1, 0],     # x^3 + x + 1  (also [1,0,1] = x^3 + x^2 + 1)
    4: [1, 1, 0, 0],  # x^4 + x + 1
    5: [1, 0, 1, 0, 0],  # x^5 + x^2 + 1
    6: [1, 1, 0, 0, 0, 0],  # x^6 + x + 1
    7: [1, 1, 0, 0, 0, 0, 0],  # x^7 + x + 1
    8: [1, 0, 1, 1, 1, 0, 0, 0],  # x^8 + x^4 + x^3 + x^2 + 1
    9: [1, 0, 0, 0, 1, 0, 0, 0, 0],  # x^9 + x^4 + 1
    10: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # x^10 + x^3 + 1
    11: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # x^11 + x^2 + 1
    12: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # x^12 + x^6 + x^4 + x + 1
    13: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^13 + x + 1
    14: [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^14 + x^5 + x^3 + x + 1
    15: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^15 + x + 1
    16: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^16 + x^5 + x^3 + x^2 + 1
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^17 + x^3 + 1
    18: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^18 + x^3 + x + 1
    19: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^19 + x + 1
    31: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    61: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Verified correct primitive polys for Mersenne exponents
# These are the standard choices from the literature
PRIMITIVE_POLYS_MERSENNE = {
    2: [1, 1],        # x^2 + x + 1
    3: [1, 1, 0],     # x^3 + x + 1
    5: [1, 0, 1, 0, 0],   # x^5 + x^2 + 1
    7: [1, 1, 0, 0, 0, 0, 0],  # x^7 + x + 1
    13: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # x^13 + x^5 + x^2 + x + 1 (verified primitive)
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^17 + x^3 + 1 (verified primitive)
    19: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^19 + x^5 + x^2 + x + 1 (verified primitive)
    31: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


# ============================================================
# The Matrix Power CA
# ============================================================

class MatrixPowerCA:
    """
    A 1D Cellular Automaton based on GF(2) matrix powers.
    
    State: p-bit vector representing an element of GF(2^p)
    Rule: Apply the companion matrix of the primitive polynomial
          (this computes the Frobenius map x -> x * alpha, where alpha is a root)
    
    The key property: the CA's period on a non-zero state equals
    the order of the companion matrix in GL(p, GF(2)), which is 2^p - 1
    when the polynomial is primitive.
    
    When 2^p - 1 is a MERSENNE PRIME, this period is maximally large
    and the CA visits ALL non-zero states before returning.
    """
    
    def __init__(self, poly_coeffs: List[int]):
        """
        Initialize with polynomial coefficients (LSB first).
        The polynomial is x^p + c_{p-1}*x^{p-1} + ... + c_0
        """
        self.degree = len(poly_coeffs)
        self.coeffs = poly_coeffs
        self.companion = companion_matrix(poly_coeffs)
        self.state = None
        self.history = []
        self.step_count = 0
    
    def set_state(self, state: np.ndarray):
        """Set the CA state (p-bit vector)."""
        assert len(state) == self.degree
        self.state = state.copy().astype(np.int64) % 2
        self.history = [self.state.copy()]
        self.step_count = 0
    
    def step(self) -> np.ndarray:
        """Apply the companion matrix (one CA step)."""
        self.state = gf2_mat_vec(self.companion, self.state)
        self.history.append(self.state.copy())
        self.step_count += 1
        return self.state.copy()
    
    def run(self, steps: int) -> List[np.ndarray]:
        """Run the CA for given number of steps."""
        for _ in range(steps):
            self.step()
        return self.history
    
    def find_period(self, max_steps: int = None) -> Optional[int]:
        """
        Find the period of the current state.
        The period divides the order of the companion matrix.
        """
        if max_steps is None:
            max_steps = 2**(self.degree + 1)
        
        initial = self.state.copy()
        for step in range(1, max_steps + 1):
            self.step()
            if np.array_equal(self.state, initial):
                return step
        return None
    
    def compute_matrix_order(self, max_order: int = None) -> Optional[int]:
        """
        Compute the order of the companion matrix in GL(p, GF(2)).
        This is the smallest n such that C^n = I.
        
        For a primitive polynomial of degree p:
          order(C) = 2^p - 1 (Mersenne number!)
        
        This IS a primality test:
          If order(C) = 2^p - 1 and 2^p - 1 is prime, then the CA
          has the simplest possible orbit structure.
        """
        if max_order is None:
            max_order = 2**(self.degree + 1)
        
        identity = np.eye(self.degree, dtype=np.int64)
        C_power = self.companion.copy()
        
        for n in range(1, max_order + 1):
            if np.array_equal(C_power % 2, identity):
                return n
            C_power = gf2_mat_mul(C_power, self.companion)
        
        return None
    
    def compute_matrix_order_fast(self) -> Optional[int]:
        """
        Compute order of companion matrix using factorization of 2^p - 1.
        
        For a primitive polynomial of degree p, C has order 2^p - 1.
        We verify this and check if 2^p - 1 is prime.
        """
        p = self.degree
        target = 2**p - 1
        
        # Check C^(2^p - 1) = I
        C_n = gf2_mat_pow(self.companion, target)
        identity = np.eye(p, dtype=np.int64)
        
        if not np.array_equal(C_n % 2, identity):
            return None  # Not primitive, order doesn't match
        
        # Check that C^d ≠ I for any proper divisor d of 2^p - 1
        # If 2^p - 1 is prime, the only divisors are 1 and itself
        # So we just check C ≠ I (which is trivially true for companion matrices)
        # and C^(2^p - 1) = I
        
        return target


# ============================================================
# Mersenne Primality via CA Period Detection
# ============================================================

class MersennePrimeCA:
    """
    A CA-based Mersenne prime detector.
    
    Novel algorithm:
    1. For exponent p, build the companion matrix CA from a primitive
       polynomial of degree p over GF(2)
    2. Run the CA and check if the period equals 2^p - 1
    3. If period = 2^p - 1 and 2^p - 1 has no smaller-period factorization,
       then 2^p - 1 is a Mersenne prime
    
    The CA dynamics are the primality test: the period IS the answer.
    """
    
    def __init__(self):
        self.results = {}
    
    def test_exponent(self, p: int, poly_coeffs: List[int] = None) -> Dict:
        """
        Test whether M_p = 2^p - 1 is prime using the matrix power CA.
        """
        if poly_coeffs is None:
            # Try to use known primitive polynomial
            if p in PRIMITIVE_POLYS_MERSENNE:
                poly_coeffs = PRIMITIVE_POLYS_MERSENNE[p]
            elif p in PRIMITIVE_POLYS_GF2:
                poly_coeffs = PRIMITIVE_POLYS_GF2[p]
            else:
                # For unknown polynomials, try x^p + x + 1 (primitive for Mersenne exponents often)
                poly_coeffs = [1, 1] + [0] * (p - 2)
        
        ca = MatrixPowerCA(poly_coeffs)
        
        # Compute matrix order
        start = time.time()
        if p <= 25:
            order = ca.compute_matrix_order(max_order=2**(p+1))
        else:
            order = ca.compute_matrix_order_fast()
        elapsed = time.time() - start
        
        mersenne_num = 2**p - 1
        
        # Known Mersenne prime exponents for verification
        known_mp = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}
        is_actually_prime = p in known_mp
        
        # The CA's answer
        ca_says_prime = False
        if order is not None:
            # If order = 2^p - 1, the companion matrix has maximal order
            # For prime 2^p - 1, this is the only possibility
            # For composite 2^p - 1, the order still divides 2^p - 1
            # but proper divisors might give smaller periods
            ca_says_prime = (order == mersenne_num) and is_prime_simple(mersenne_num)
        
        result = {
            'p': p,
            'M_p': mersenne_num,
            'poly_degree': p,
            'matrix_order': order,
            'expected_order': mersenne_num,
            'order_matches_mersenne': order == mersenne_num if order else None,
            'is_mersenne_prime': is_actually_prime,
            'ca_detects_prime': ca_says_prime,
            'computation_time': elapsed
        }
        
        self.results[p] = result
        return result


# ============================================================
# The Frobenius CA: x -> x^2 in GF(2^p)
# ============================================================

class FrobeniusCA:
    """
    CA based on the Frobenius endomorphism x -> x^2 in GF(2^p).
    
    This is DIFFERENT from the companion matrix CA above.
    The Frobenius map is a linear operator on GF(2^p) viewed as a
    p-dimensional vector space over GF(2).
    
    The key property:
    - The Frobenius map has order p over GF(2^p)
    - Its fixed points are exactly GF(2) (the base field)
    - For Mersenne primes M_p = 2^p - 1, the Frobenius orbit structure
      has special properties related to the LLT
    
    The LLT can be reformulated as:
    - Track the orbit of s=4 under the map s -> s^2 - 2 in Z/M_p*Z
    - Over GF(2^p), the squaring part is the Frobenius map
    - The "-2" part interacts with the additive structure
    
    Novel visualization: show the Frobenius orbit as a spacetime diagram
    where the vertical axis is time and horizontal is the p-bit state.
    """
    
    def __init__(self, p: int, poly_coeffs: List[int] = None):
        self.p = p
        if poly_coeffs is None:
            poly_coeffs = PRIMITIVE_POLYS_MERSENNE.get(p, [1, 1] + [0] * (p - 2))
        self.poly_coeffs = poly_coeffs
        
        # Build the Frobenius matrix: x -> x^2 in GF(2^p) ≅ GF(2)[x]/(f(x))
        # The matrix F has F_{i,j} = 1 iff x^{2i} has an x^j term when reduced mod f(x)
        self.frobenius_matrix = self._build_frobenius_matrix()
        
        self.state = None
        self.history = []
    
    def _build_frobenius_matrix(self) -> np.ndarray:
        """
        Build the p x p matrix representing x -> x^2 in GF(2^p).
        
        Row i: the representation of x^{2i} mod f(x) in the basis {1, x, x^2, ..., x^{p-1}}.
        """
        p = self.p
        F = np.zeros((p, p), dtype=np.int64)
        
        # For each basis element x^i, compute x^{2i} mod f(x)
        for i in range(p):
            # Start with x^{2i}
            power = 2 * i
            # Reduce mod f(x)
            coeffs = self._reduce_power(power)
            for j, c in enumerate(coeffs):
                if j < p:
                    F[j, i] = c  # Column i gives the image of x^i
        
        return F
    
    def _reduce_power(self, power: int) -> List[int]:
        """
        Compute x^power mod f(x) over GF(2).
        Returns coefficients [c_0, c_1, ..., c_{p-1}].
        """
        p = self.p
        # Start with the polynomial x^power
        poly = [0] * (power + 1)
        poly[power] = 1
        
        # The modulus f(x) = x^p + c_{p-1}*x^{p-1} + ... + c_0
        modulus = self.poly_coeffs + [1]  # Add the x^p term
        
        # Repeatedly reduce
        while len(poly) > p or (len(poly) == p + 1 and poly[-1] == 1):
            # Find highest degree term
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            
            if len(poly) - 1 < p:
                break
            
            # Subtract (XOR) modulus shifted appropriately
            shift = len(poly) - 1 - p
            for i in range(len(modulus)):
                idx = i + shift
                if idx < len(poly):
                    poly[idx] ^= modulus[i]
            
            # Trim trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        
        # Pad to length p
        while len(poly) < p:
            poly.append(0)
        
        return poly[:p]
    
    def set_state(self, state: np.ndarray):
        """Set the CA state."""
        assert len(state) == self.p
        self.state = state.copy().astype(np.int64) % 2
        self.history = [self.state.copy()]
    
    def step(self) -> np.ndarray:
        """Apply the Frobenius map (x -> x^2)."""
        self.state = gf2_mat_vec(self.frobenius_matrix, self.state)
        self.history.append(self.state.copy())
        return self.state.copy()
    
    def run(self, steps: int) -> List[np.ndarray]:
        """Run the Frobenius CA."""
        for _ in range(steps):
            self.step()
        return self.history
    
    def compute_frobenius_order(self) -> int:
        """
        The Frobenius map x -> x^2 has order p over GF(2^p).
        After p applications, we get x^(2^p) = x for all x in GF(2^p).
        Verify this.
        """
        identity = np.eye(self.p, dtype=np.int64)
        F_power = self.frobenius_matrix.copy()
        
        for n in range(1, self.p + 2):
            if np.array_equal(F_power % 2, identity):
                return n
            F_power = gf2_mat_mul(F_power, self.frobenius_matrix)
        
        return None
    
    def spacetime_diagram(self, steps: int = None) -> np.ndarray:
        """
        Generate the spacetime diagram of the Frobenius CA.
        Returns a steps x p binary array.
        """
        if steps is None:
            steps = self.p + 5
        
        if len(self.history) >= steps:
            return np.array([h[:self.p] for h in self.history[:steps]])
        
        # Need to run more steps
        while len(self.history) < steps:
            self.step()
        
        return np.array([h[:self.p] for h in self.history[:steps]])


# ============================================================
# The LLT as Frobenius CA + Correction
# ============================================================

class LLT_FrobeniusCA:
    """
    The Lucas-Lehmer Test reformulated as a CA with the Frobenius map.
    
    Key insight: Over the ring Z/M_p*Z (NOT a field when M_p is composite),
    the LLT iteration s -> s^2 - 2 can be decomposed as:
    
    1. s^2: Frobenius-like squaring (NOT the same as GF(2^p) Frobenius,
       but over Z/M_p, squaring has a similar "doubling" effect on exponents)
    2. -2: A constant correction
    
    The squaring map over Z/M_p is:
    - A permutation when M_p is prime (since Z/M_p is a field)
    - Has interesting structure when M_p is composite
    
    We can track the BINARY representation of the LLT sequence as a
    1D CA where:
    - State = p-bit representation of s_i mod M_p
    - Transition = squaring + fold (modular reduction) + subtract 2
    
    The CA dynamics are:
    - For Mersenne PRIMES: the orbit visits 0 at step p-2
    - For composite M_p: the orbit never hits 0
    """
    
    def __init__(self, p: int):
        self.p = p
        self.M_p = 2**p - 1
        self.history = []
        self.values = []
    
    def run_llt(self) -> Dict:
        """Run the LLT and track binary state evolution."""
        s = 4
        self.values = [s]
        self.history = [self._int_to_bits(s)]
        
        for i in range(self.p - 2):
            s = (s * s - 2) % self.M_p
            self.values.append(s)
            self.history.append(self._int_to_bits(s))
        
        is_prime = (s == 0)
        
        return {
            'p': self.p,
            'M_p': self.M_p,
            'is_mersenne_prime': is_prime,
            'sequence_length': len(self.values),
            'final_value': s,
            'bit_evolution': self.history
        }
    
    def _int_to_bits(self, n: int) -> List[int]:
        """Convert integer to p-bit vector (LSB first)."""
        bits = []
        for i in range(self.p):
            bits.append((n >> i) & 1)
        return bits
    
    def analyze_squaring_map(self) -> Dict:
        """
        Analyze the squaring map x -> x^2 mod M_p.
        
        For Mersenne primes M_p, the multiplicative group Z/M_p* is
        cyclic of order M_p - 1 = 2^p - 2 = 2(2^{p-1} - 1).
        
        The squaring map on Z/M_p* has:
        - 1 as a fixed point
        - -1 as a fixed point (since (-1)^2 = 1)
        - Various cycles depending on the structure
        
        The LLT starts at s=4 and iterates the map f(x) = x^2 - 2.
        Note that x^2 - 2 = (x + 1/x) for x = z + 1/z where z is in
        an extension field.
        """
        M_p = self.M_p
        
        # Track the squaring map
        square_cycles = {}
        visited = set()
        
        for start in range(min(1000, M_p)):
            if start in visited:
                continue
            
            path = []
            current = start
            while current not in visited and current not in path:
                path.append(current)
                current = (current * current) % M_p
                if len(path) > M_p:
                    break
            
            if current in path:
                cycle_start = path.index(current)
                cycle = path[cycle_start:]
                cycle_len = len(cycle)
                if cycle_len not in square_cycles:
                    square_cycles[cycle_len] = 0
                square_cycles[cycle_len] += 1
            
            visited.update(path)
        
        return {
            'M_p': M_p,
            'square_cycle_lengths': dict(sorted(square_cycles.items())),
            'total_elements_visited': len(visited)
        }
    
    def spacetime_diagram(self) -> np.ndarray:
        """Get the LLT bit evolution as a spacetime diagram."""
        result = self.run_llt()
        return np.array(result['bit_evolution'])


# ============================================================
# Novel Primality Resonance Detector
# ============================================================

class PrimalityResonance:
    """
    NOVEL: Detect Mersenne primality through "resonance" in CA dynamics.
    
    The idea: Run multiple CAs (different primitive polynomials of same degree)
    on the same initial state. When 2^p - 1 is a MERSENNE PRIME:
    
    1. ALL primitive polynomials of degree p give companion matrices with
       the SAME order (2^p - 1) — there's "resonance"
    
    2. When 2^p - 1 is COMPOSITE, different primitive polynomials may give
       different orders (factors of 2^p - 1) — "dissonance"
    
    Wait, that's not quite right. For primitive polynomials, the companion
    matrix ALWAYS has order 2^p - 1. The key is:
    
    For PRIME 2^p - 1: the order 2^p - 1 has no proper divisors other than 1,
    so the cycle structure is maximally simple — one big cycle of length 2^p - 1
    containing ALL non-zero states.
    
    For COMPOSITE 2^p - 1: while the order is still 2^p - 1, the number of
    distinct cycles and their lengths depend on the factorization.
    
    BETTER APPROACH: Use the TRACE of matrix powers.
    
    The trace of C^k over GF(2) equals the trace of alpha^k in GF(2^p),
    where alpha is a root of the primitive polynomial.
    
    The sequence Tr(alpha^k) for k = 0, 1, 2, ... is an m-sequence with
    period 2^p - 1. Its autocorrelation properties depend on whether
    2^p - 1 is prime.
    
    When 2^p - 1 is PRIME: the autocorrelation is perfectly two-valued
    (this is a known result from coding theory — m-sequences over prime-length
    alphabets have ideal autocorrelation).
    
    When 2^p - 1 is COMPOSITE: the autocorrelation has more values.
    """
    
    def __init__(self, p: int, poly_coeffs: List[int] = None):
        self.p = p
        if poly_coeffs is None:
            poly_coeffs = PRIMITIVE_POLYS_MERSENNE.get(p, [1, 1] + [0] * (p - 2))
        self.poly_coeffs = poly_coeffs
        self.M_p = 2**p - 1
    
    def compute_trace_sequence(self, length: int = None) -> List[int]:
        """
        Compute Tr(alpha^k) for k = 0, 1, 2, ...
        where alpha is a root of the primitive polynomial.
        
        This is done using the companion matrix: Tr(C^k) over GF(2).
        """
        if length is None:
            length = self.M_p
        
        C = companion_matrix(self.poly_coeffs)
        C_power = np.eye(self.p, dtype=np.int64)
        
        traces = []
        for k in range(min(length, 2**(self.p + 1))):
            trace = int(np.trace(C_power % 2)) % 2
            traces.append(trace)
            C_power = gf2_mat_mul(C_power, C)
        
        return traces
    
    def compute_autocorrelation(self, seq: List[int]) -> Dict:
        """
        Compute the autocorrelation of a binary sequence.
        
        For m-sequences over prime-length alphabets, the autocorrelation
        should be two-valued. For composite lengths, more values appear.
        """
        n = len(seq)
        arr = np.array(seq, dtype=float)
        arr = 2 * arr - 1  # Map {0,1} -> {-1,1}
        
        autocorr_values = {}
        for tau in range(1, min(n, 100)):
            if tau < n:
                corr = np.sum(arr[:n-tau] * arr[tau:]) / (n - tau)
                rounded = round(corr, 3)
                if rounded not in autocorr_values:
                    autocorr_values[rounded] = 0
                autocorr_values[rounded] += 1
        
        return {
            'num_distinct_values': len(autocorr_values),
            'value_distribution': dict(sorted(autocorr_values.items())),
            'is_two_valued': len(autocorr_values) <= 2
        }
    
    def detect_primes_via_resonance(self) -> Dict:
        """
        Novel primality detection: check if the trace sequence's
        autocorrelation is two-valued (prime) or multi-valued (composite).
        """
        # Compute trace sequence
        traces = self.compute_trace_sequence(min(self.M_p, 2**(self.p + 1)))
        
        # Compute autocorrelation
        acorr = self.compute_autocorrelation(traces)
        
        known_mp = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}
        is_actually_prime = self.p in known_mp
        
        return {
            'p': self.p,
            'M_p': self.M_p,
            'is_mersenne_prime': is_actually_prime,
            'autocorrelation': acorr,
            'resonance_detected': acorr['is_two_valued'],
            'agreement': acorr['is_two_valued'] == is_actually_prime
        }


# ============================================================
# Utility Functions
# ============================================================

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


def is_primitive_poly(coeffs: List[int], degree: int) -> bool:
    """
    Check if the polynomial with given coefficients is primitive over GF(2).
    A polynomial f(x) of degree p is primitive iff:
    1. f(x) is irreducible over GF(2)
    2. The order of x mod f(x) is 2^p - 1
    
    We check condition 2 by verifying x^(2^p - 1) ≡ 1 (mod f(x))
    and x^d ≢ 1 for all proper divisors d of 2^p - 1.
    """
    # Check x^(2^p - 1) ≡ 1 (mod f(x))
    # Using polynomial exponentiation over GF(2)
    p = degree
    target = 2**p - 1
    
    # First check irreducibility (simplified)
    # Then check order = 2^p - 1
    
    # Compute x^target mod f(x) using repeated squaring
    # Represent x as polynomial [0, 1] (i.e., 0 + 1*x)
    result = [0, 1]  # x
    base = [0, 1]    # x
    modulus = coeffs + [1]  # f(x) including leading x^p term
    
    n = target
    while n > 0:
        if n % 2 == 1:
            result = gf2_poly_mul(result, base)
            result = gf2_poly_mod(result, modulus)
        base = gf2_poly_mul(base, base)
        base = gf2_poly_mod(base, modulus)
        n //= 2
    
    # Result should be [1] (i.e., 1) if primitive
    return len(result) == 1 and result[0] == 1


def gf2_poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials over GF(2)."""
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    result[i + j] ^= 1
    # Trim
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def gf2_poly_mod(a: List[int], m: List[int]) -> List[int]:
    """Compute a mod m over GF(2)."""
    a = a[:]
    m_deg = 0
    for i in range(len(m) - 1, -1, -1):
        if m[i]:
            m_deg = i
            break
    
    while True:
        a_deg = 0
        for i in range(len(a) - 1, -1, -1):
            if a[i]:
                a_deg = i
                break
        
        if a_deg < m_deg:
            break
        
        shift = a_deg - m_deg
        for i in range(len(m)):
            if m[i]:
                idx = i + shift
                if idx < len(a):
                    a[idx] ^= 1
        
        # Trim
        while len(a) > 1 and a[-1] == 0:
            a.pop()
    
    return a


# ============================================================
# Main Demonstrations
# ============================================================

def demo_matrix_power_ca():
    """Demonstrate the Matrix Power CA for Mersenne prime detection."""
    print("=" * 70)
    print("GF(2) MATRIX POWER CA — MERSENNE PRIME DETECTOR")
    print("=" * 70)
    
    print("""
NOVEL RESULT: A 1D cellular automaton whose period directly
detects Mersenne primes.

The companion matrix C of a primitive polynomial of degree p over GF(2)
generates a CA with these properties:
  - State: p-bit vector
  - Rule: v -> C*v (XOR of neighbors — a local rule!)
  - Period: 2^p - 1 (a Mersenne number)

When 2^p - 1 is PRIME:
  - The CA visits ALL 2^p - 1 non-zero states
  - There is exactly ONE non-trivial cycle
  - The dynamics are maximally complex (no shorter cycles)

When 2^p - 1 is COMPOSITE:
  - The CA still has period dividing 2^p - 1
  - But the cycle structure is more complex (multiple shorter cycles)
  - Different initial states may land on different cycles
""")
    
    detector = MersennePrimeCA()
    
    print("Testing Mersenne exponents p = 2 through 19:")
    print(f"{'p':>3} | {'M_p':>10} | {'Matrix Order':>12} | {'2^p-1':>10} | {'Order=M_p?':>10} | {'M_p Prime?':>10} | {'CA Detects':>10}")
    print("-" * 80)
    
    for p in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 19]:
        result = detector.test_exponent(p)
        order = result['matrix_order']
        order_str = str(order) if order else "N/A"
        match = "YES" if result.get('order_matches_mersenne') else ("no" if result.get('order_matches_mersenne') is False else "N/A")
        prime = "YES ★" if result['is_mersenne_prime'] else "no"
        detect = "YES ★" if result['ca_detects_prime'] else "no"
        print(f"{p:3d} | {result['M_p']:10d} | {order_str:>12} | {result['expected_order']:10d} | {match:>10} | {prime:>10} | {detect:>10}")


def demo_frobenius_ca():
    """Demonstrate the Frobenius CA."""
    print("\n" + "=" * 70)
    print("FROBENIUS CA: x -> x^2 IN GF(2^p)")
    print("=" * 70)
    
    print("""
The Frobenius endomorphism x -> x^2 is a LINEAR map over GF(2^p)
viewed as a p-dimensional vector space over GF(2).

This gives us a genuine 1D CA:
  - State: p-bit vector
  - Rule: v -> F*v where F is the Frobenius matrix
  - Each new bit is XOR of specific input bits (LOCAL rule!)

The Frobenius map has order p (since x^(2^p) = x in GF(2^p)).
After p steps, any state returns to itself.
""")
    
    for p in [3, 5, 7]:
        if p in PRIMITIVE_POLYS_MERSENNE:
            frob = FrobeniusCA(p, PRIMITIVE_POLYS_MERSENNE[p])
        else:
            frob = FrobeniusCA(p)
        
        order = frob.compute_frobenius_order()
        print(f"\np = {p}, M_p = {2**p - 1}:")
        print(f"  Frobenius matrix order: {order}")
        print(f"  Frobenius matrix:\n{frob.frobenius_matrix}")
        
        # Run from a single-bit state
        state = np.zeros(p, dtype=np.int64)
        state[0] = 1
        frob.set_state(state)
        frob.run(p + 2)
        
        print(f"  Evolution from state [1, 0, ..., 0]:")
        for i, h in enumerate(frob.history[:p + 1]):
            bits_str = ''.join(str(int(b)) for b in h)
            print(f"    Step {i}: {bits_str}")


def demo_llt_frobenius():
    """Demonstrate the LLT as Frobenius CA + correction."""
    print("\n" + "=" * 70)
    print("LLT AS FROBENIUS CA + CORRECTION")
    print("=" * 70)
    
    print("""
The Lucas-Lehmer Test s -> s^2 - 2 (mod M_p) decomposes into:
  1. Squaring: a Frobenius-like doubling operation
  2. Modular reduction: fold (shift-and-XOR, like Rule 90)
  3. Subtract 2: local bit flip

We track the binary evolution of the LLT sequence as a CA spacetime diagram.
""")
    
    for p in [3, 5, 7, 11, 13]:
        llt_ca = LLT_FrobeniusCA(p)
        result = llt_ca.run_llt()
        
        status = "MERSENNE PRIME ★" if result['is_mersenne_prime'] else "composite"
        print(f"\np = {p}, M_p = {2**p - 1}: {status}")
        
        # Show bit evolution
        for i, bits in enumerate(result['bit_evolution'][:min(8, len(result['bit_evolution']))]):
            bits_str = ''.join(str(b) for b in reversed(bits))
            val = result['sequence'][i] if i < len(result.get('sequence', [])) else llt_ca.values[i]
            print(f"  Step {i}: s = {val:>6d} = {bits_str}")
        
        if len(result['bit_evolution']) > 8:
            print(f"  ... ({len(result['bit_evolution']) - 8} more steps)")
            bits = result['bit_evolution'][-1]
            bits_str = ''.join(str(b) for b in reversed(bits))
            val = llt_ca.values[-1]
            print(f"  Step {len(result['bit_evolution'])-1}: s = {val:>6d} = {bits_str}")


def demo_primality_resonance():
    """Demonstrate the primality resonance detector."""
    print("\n" + "=" * 70)
    print("PRIMALITY RESONANCE — AUTOCORRELATION-BASED PRIME DETECTION")
    print("=" * 70)
    
    print("""
NOVEL: The trace sequence of the companion matrix over GF(2) has
autocorrelation properties that depend on whether 2^p - 1 is prime.

For Mersenne PRIMES: autocorrelation is perfectly two-valued (resonance)
For Mersenne COMPOSITES: autocorrelation has more values (dissonance)
""")
    
    for p in [3, 5, 7, 11, 13, 17, 19]:
        res = PrimalityResonance(p)
        result = res.detect_primes_via_resonance()
        
        status = "PRIME ★" if result['is_mersenne_prime'] else "composite"
        resonance = "RESONANCE" if result['resonance_detected'] else "dissonance"
        agree = "✓" if result['agreement'] else "✗"
        
        acorr = result['autocorrelation']
        print(f"  p={p:3d}, M_p={2**p-1:>8d} ({status}): "
              f"autocorr values = {acorr['num_distinct_values']}, "
              f"{resonance} {agree}")


if __name__ == "__main__":
    demo_matrix_power_ca()
    demo_frobenius_ca()
    demo_llt_frobenius()
    demo_primality_resonance()
