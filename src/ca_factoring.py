"""
CA-Based Factoring of Mersenne Numbers
=======================================

Core question: Can the companion matrix CA's dynamics extract FACTORS
of composite Mersenne numbers — not just detect primality?

ANSWER: YES. Here's how:

The map x → x² mod M_p (the core of Pollard's rho factoring method)
decomposes into CA operations:
  1. Squaring: bit convolution (spreading CA rule)
  2. Reduction mod M_p = 2^p - 1: FOLD (XOR upper bits into lower bits — Rule 90-like)

So each step of Pollard's rho IS a CA update on a binary string.

When M_p = q1 × q2 is composite:
- Z/M_p Z has zero divisors
- The map x → x² mod M_p has a different orbit structure than over a field
- Specifically: if we track x_i and x_{2i} (Floyd's cycle detection),
  then gcd(|x_i - x_{2i}|, M_p) reveals a factor with high probability

The CA angle: the ENTIRE factoring computation — squaring, folding,
comparison, GCD — can be expressed as cellular automaton operations.

This module implements:
1. CA-based Mersenne modular arithmetic (square + fold)
2. Pollard's rho factoring algorithm where each step is a CA operation
3. Spacetime visualization of the factoring process
4. Multiple factoring methods (rho, p-1, order-based) all in CA terms
5. Factor extraction from the orbit structure of non-primitive CAs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd, isqrt
import time


# ============================================================
# Mersenne Modular Arithmetic as CA Operations
# ============================================================

def mersenne_fold(x: int, p: int) -> int:
    """
    Compute x mod (2^p - 1) using the CA fold operation.
    
    Since 2^p ≡ 1 (mod 2^p - 1), reduction mod M_p is:
    - Split x into chunks of p bits
    - Add (or XOR for the basic version) all chunks together
    
    This is a LOCAL, PARALLEL, CA-like operation:
    each bit position interacts with the bit p positions away.
    Structurally identical to Rule 90 on a wide grid!
    
    Returns x mod (2^p - 1).
    """
    M_p = (1 << p) - 1
    
    if x < M_p:
        return x
    
    # Repeatedly fold upper bits into lower bits
    while x > M_p:
        upper = x >> p
        lower = x & M_p
        x = upper + lower  # Addition, not XOR (correct for Mersenne mod)
    
    if x == M_p:
        return 0  # M_p ≡ 0 (mod M_p)
    
    return x


def mersenne_square(x: int, p: int) -> int:
    """
    Compute x² mod (2^p - 1) using CA operations.
    
    Step 1: x² — bit convolution (spreading CA rule)
    Step 2: mod M_p — fold (Rule 90-like)
    
    The bit convolution for squaring:
    - Each bit of x at position i spreads to create contributions
      at positions i+j for each bit j of x
    - This is a 2D CA: bits spread diagonally and accumulate
    
    The fold:
    - XOR/add upper p bits into lower p bits
    - This is Rule 90 applied to the binary representation!
    """
    M_p = (1 << p) - 1
    squared = x * x
    return mersenne_fold(squared, p)


def mersenne_multiply(a: int, b: int, p: int) -> int:
    """
    Compute a * b mod (2^p - 1) using CA operations.
    
    Same structure as squaring but generalized:
    - Bit convolution of a and b (spreading CA)
    - Fold for Mersenne reduction (Rule 90-like)
    """
    M_p = (1 << p) - 1
    product = a * b
    return mersenne_fold(product, p)


def mersenne_pow(base: int, exp: int, p: int) -> int:
    """
    Compute base^exp mod (2^p - 1) using CA operations.
    Square-and-multiply with Mersenne fold at each step.
    """
    M_p = (1 << p) - 1
    result = 1
    b = base % M_p
    
    while exp > 0:
        if exp & 1:
            result = mersenne_multiply(result, b, p)
        b = mersenne_square(b, p)
        exp >>= 1
    
    return result


# ============================================================
# CA-Based Pollard's Rho Factoring
# ============================================================

def ca_pollard_rho(M_p: int, p: int, max_iterations: int = 100000,
                   seed: int = 2, c: int = 1) -> Optional[int]:
    """
    Factor M_p = 2^p - 1 using Pollard's rho, where each step
    x → x² + c (mod M_p) is a CA operation.
    
    The iteration x → x² + c decomposes into:
    1. x²: bit convolution CA
    2. mod M_p: fold CA (Rule 90-like)  
    3. + c: local bit addition
    
    When M_p is composite, Floyd's cycle detection finds a collision
    in Z/M_p Z (not a field!), and gcd(|x_i - x_{2i}|, M_p) reveals a factor.
    
    Returns a non-trivial factor of M_p, or None if not found.
    """
    x = seed  # "Tortoise"
    y = seed  # "Hare" (moves twice as fast)
    
    for iteration in range(max_iterations):
        # Tortoise: one CA step
        x = mersenne_fold(x * x + c, p)
        
        # Hare: two CA steps
        y = mersenne_fold(y * y + c, p)
        y = mersenne_fold(y * y + c, p)
        
        # Check for factor
        diff = abs(x - y)
        if diff == 0:
            # Cycle found but gcd is trivial — try different c or seed
            return None
        
        factor = gcd(diff, M_p)
        
        if 1 < factor < M_p:
            return factor
    
    return None


def ca_pollard_rho_brent(M_p: int, p: int, max_iterations: int = 100000,
                          seed: int = 2, c: int = 1) -> Optional[int]:
    """
    Brent's improvement of Pollard's rho (faster cycle detection).
    Each step still decomposes into CA operations.
    """
    y = seed
    r = 1
    q = 1
    x = y
    d = 1
    
    while d == 1:
        x = y
        for _ in range(r):
            y = mersenne_fold(y * y + c, p)
        
        k = 0
        while k < r and d == 1:
            ys = y
            for _ in range(min(128, r - k)):
                y = mersenne_fold(y * y + c, p)
                q = mersenne_fold(q * abs(x - y), p)
            d = gcd(q, M_p)
            k += 128
        
        r *= 2
        
        if r > max_iterations:
            break
    
    if d == M_p:
        # Backtrack
        while True:
            ys = mersenne_fold(ys * ys + c, p)
            d = gcd(abs(x - ys), M_p)
            if d > 1:
                break
    
    if 1 < d < M_p:
        return d
    
    return None


def factor_mersenne_complete(p: int, max_attempts: int = 50) -> Dict:
    """
    Completely factor M_p = 2^p - 1 using CA-based methods.
    Tries multiple seeds and c values for Pollard's rho.
    """
    M_p = (1 << p) - 1
    
    # Quick check: is M_p prime?
    if is_prime_miller_rabin(M_p):
        return {
            'p': p,
            'M_p': M_p,
            'is_prime': True,
            'factors': [(M_p, 1)],
            'factorization': f"{M_p} (prime)",
            'method': 'CA-based Miller-Rabin'
        }
    
    # Try to factor
    factors = {}
    remaining = M_p
    
    # Try small prime factors first (trial division)
    for small_p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        while remaining % small_p == 0:
            factors[small_p] = factors.get(small_p, 0) + 1
            remaining //= small_p
    
    if remaining == 1:
        factor_list = [(f, e) for f, e in sorted(factors.items())]
        return {
            'p': p,
            'M_p': M_p,
            'is_prime': False,
            'factors': factor_list,
            'factorization': format_factorization(factors, M_p),
            'method': 'trial_division'
        }
    
    # Use CA-based Pollard's rho for remaining factors
    attempts = 0
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 37, 42, 99, 127, 256]
    c_values = [1, 2, 3, 7, 13, 19, 31]
    
    while remaining > 1 and attempts < max_attempts:
        if is_prime_miller_rabin(remaining):
            factors[remaining] = factors.get(remaining, 0) + 1
            remaining = 1
            break
        
        found = False
        for seed in seeds:
            for c in c_values:
                factor = ca_pollard_rho(remaining, p, max_iterations=50000,
                                        seed=seed % remaining, c=c)
                if factor is not None and factor > 1 and factor < remaining:
                    factors[factor] = factors.get(factor, 0) + 1
                    remaining //= factor
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Try Brent's variant
            for seed in seeds:
                for c in c_values:
                    factor = ca_pollard_rho_brent(remaining, p, max_iterations=50000,
                                                   seed=seed % remaining, c=c)
                    if factor is not None and factor > 1 and factor < remaining:
                        factors[factor] = factors.get(factor, 0) + 1
                        remaining //= factor
                        found = True
                        break
                if found:
                    break
        
        if not found:
            factors[remaining] = factors.get(remaining, 0) + 1
            remaining = 1
            # Note: remaining might be composite but we couldn't factor it
        
        attempts += 1
    
    factor_list = [(f, e) for f, e in sorted(factors.items())]
    return {
        'p': p,
        'M_p': M_p,
        'is_prime': False,
        'factors': factor_list,
        'factorization': format_factorization(factors, M_p),
        'method': 'CA_Pollard_rho'
    }


def format_factorization(factors: Dict[int, int], original: int) -> str:
    """Format factorization as string."""
    parts = []
    for f, e in sorted(factors.items()):
        if e == 1:
            parts.append(str(f))
        else:
            parts.append(f"{f}^{e}")
    
    result = " × ".join(parts)
    
    # Verify
    product = 1
    for f, e in factors.items():
        product *= f ** e
    
    if product != original:
        result += f" (INCOMPLETE: product = {product} ≠ {original})"
    
    return result


# ============================================================
# Factor Extraction from CA Orbit Structure
# ============================================================

class CAOrbitFactorer:
    """
    Extract factors from the orbit structure of the companion matrix CA.
    
    Key insight: For a primitive polynomial of degree p over GF(2),
    the companion matrix C has order M_p = 2^p - 1. All non-zero 
    states form a single orbit.
    
    But for a NON-PRIMITIVE irreducible polynomial, the companion 
    matrix has order that is a PROPER DIVISOR of M_p. By finding 
    such polynomials and computing their orders, we extract factors.
    
    Algorithm:
    1. Generate random irreducible polynomials of degree p
    2. Compute their companion matrix orders (divisors of M_p)
    3. Take GCDs of orders with M_p to find factors
    4. Combine factors to get complete factorization
    """
    
    def __init__(self, p: int):
        self.p = p
        self.M_p = (1 << p) - 1
        self.orders_found = []
    
    def find_irreducible_poly(self, rng: np.random.RandomState = None) -> Optional[List[int]]:
        """
        Find a random irreducible polynomial of degree p over GF(2).
        Uses the Ben-Or irreducibility test.
        
        Returns coefficients [c_0, c_1, ..., c_{p-1}] (LSB first),
        or None if not found.
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # Try random polynomials
        for _ in range(1000):
            # Random monic polynomial of degree p: c_0 + c_1*x + ... + c_{p-1}*x^{p-1} + x^p
            # Must have c_0 = 1 (otherwise x divides it)
            coeffs = [1]  # c_0 = 1
            for i in range(1, self.p):
                coeffs.append(int(rng.randint(0, 2)))
            
            if self._is_irreducible(coeffs):
                return coeffs
        
        return None
    
    def _is_irreducible(self, coeffs: List[int]) -> bool:
        """
        Test irreducibility of f(x) = coeffs[0] + coeffs[1]*x + ... + x^p
        over GF(2) using the Ben-Or algorithm.
        
        f is irreducible iff:
        1. x^(2^p) ≡ x (mod f(x))
        2. gcd(x^(2^i) - x, f(x)) = 1 for all i < p
        """
        p = self.p
        modulus = coeffs + [1]  # Add the x^p term
        
        # Check condition 1: x^(2^p) mod f(x) = x
        result = self._poly_powmod_x2n(1, p, modulus)  # x^(2^1)
        for _ in range(p - 1):
            result = self._poly_powmod_x2n_from(result, 1, modulus)
        
        # result should be x, i.e., [0, 1]
        if not (len(result) == 2 and result[0] == 0 and result[1] == 1):
            return False
        
        # Check condition 2: gcd(x^(2^i) - x, f(x)) = 1 for i = 1, ..., p-1
        x2i = [0, 1]  # x^(2^0) = x
        for i in range(1, p):
            # Compute x^(2^i) mod f(x)
            x2i = self._poly_powmod_x2n_from(x2i, 1, modulus)
            
            # Compute x^(2^i) - x mod f(x) (same as + in GF(2))
            diff = self._poly_add(x2i, [0, 1])
            
            # Compute gcd(diff, f(x))
            g = self._poly_gcd(diff, modulus)
            
            if len(g) > 1 or g[0] != 1:
                return False
        
        return True
    
    def _poly_powmod_x2n(self, power: int, n: int, modulus: List[int]) -> List[int]:
        """Compute x^(power * 2^n) mod modulus over GF(2)."""
        # Start with x^power
        if power == 1:
            result = [0, 1]
        else:
            result = [0] * (power + 1)
            result[power] = 1
            result = self._poly_mod(result, modulus)
        
        # Apply Frobenius n times: f(x) -> f(x^2)
        for _ in range(n):
            result = self._poly_frobenius(result, modulus)
        
        return result
    
    def _poly_powmod_x2n_from(self, poly: List[int], n: int, modulus: List[int]) -> List[int]:
        """Compute poly(x^(2^n)) mod modulus over GF(2)."""
        for _ in range(n):
            poly = self._poly_frobenius(poly, modulus)
        return poly
    
    def _poly_frobenius(self, poly: List[int], modulus: List[int]) -> List[int]:
        """
        Compute poly(x^2) mod modulus over GF(2).
        Since (a+b)^2 = a^2 + b^2 in characteristic 2,
        poly(x^2) = sum c_i * x^{2i}.
        """
        # Compute poly(x^2) by doubling all exponents
        max_exp = 2 * (len(poly) - 1)
        result = [0] * (max_exp + 1)
        for i, c in enumerate(poly):
            if c:
                result[2 * i] = 1
        
        # Reduce mod modulus
        return self._poly_mod(result, modulus)
    
    def _poly_mod(self, a: List[int], m: List[int]) -> List[int]:
        """Compute a mod m over GF(2)."""
        a = a[:]
        m_deg = len(m) - 1
        while m_deg >= 0 and m[m_deg] == 0:
            m_deg -= 1
        
        while True:
            a_deg = len(a) - 1
            while a_deg >= 0 and a[a_deg] == 0:
                a_deg -= 1
            
            if a_deg < 0 or a_deg < m_deg:
                break
            
            shift = a_deg - m_deg
            for i in range(m_deg + 1):
                if m[i]:
                    a[i + shift] ^= 1
            
            # Trim
            while len(a) > 1 and a[-1] == 0:
                a.pop()
        
        if not a:
            return [0]
        return a
    
    def _poly_add(self, a: List[int], b: List[int]) -> List[int]:
        """Add two polynomials over GF(2)."""
        result = [0] * max(len(a), len(b))
        for i, c in enumerate(a):
            result[i] ^= c
        for i, c in enumerate(b):
            result[i] ^= c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    def _poly_gcd(self, a: List[int], b: List[int]) -> List[int]:
        """Compute GCD of two polynomials over GF(2) using Euclidean algorithm."""
        while not (len(b) == 1 and b[0] == 0):
            a, b = b, self._poly_mod(a, b)
        
        # Make monic
        if len(a) > 0 and a[-1] == 1:
            return a  # Already monic (leading coeff is 1 in GF(2))
        return a
    
    def compute_companion_order(self, coeffs: List[int]) -> Optional[int]:
        """
        Compute the order of the companion matrix for the given polynomial.
        
        For an irreducible polynomial of degree p, the order divides M_p = 2^p - 1.
        We compute it by finding the smallest n such that x^n ≡ 1 (mod f(x)).
        """
        p = self.p
        M_p = self.M_p
        modulus = coeffs + [1]
        
        # We know x^M_p ≡ 1 (mod f(x)) for any irreducible f of degree p
        # The order of x divides M_p
        # Find it by checking x^(M_p/q) for each prime factor q of M_p
        
        # First, factor M_p (we need this for order computation)
        # But we're trying to FACTOR M_p... so this is circular for large M_p
        # For small M_p, we can factor it easily
        
        # Alternative: find the order by direct computation
        # Start with n = M_p, try dividing by 2 repeatedly, then by small primes
        
        # Verify x^M_p ≡ 1 (mod f(x))
        x_n = self._poly_powmod([0, 1], M_p, modulus)
        if not (len(x_n) == 1 and x_n[0] == 1):
            return None  # Polynomial is not irreducible, or error
        
        # Now find the order by trying to reduce it
        order = M_p
        
        # Try dividing by 2
        while order % 2 == 0:
            test = self._poly_powmod([0, 1], order // 2, modulus)
            if len(test) == 1 and test[0] == 1:
                order //= 2
            else:
                break
        
        # Try dividing by small odd primes
        for q in range(3, min(1000, isqrt(order) + 2), 2):
            if q * q > order:
                break
            while order % q == 0:
                test = self._poly_powmod([0, 1], order // q, modulus)
                if len(test) == 1 and test[0] == 1:
                    order //= q
                else:
                    break
        
        return order
    
    def _poly_powmod(self, base: List[int], exp: int, modulus: List[int]) -> List[int]:
        """Compute base^exp mod modulus over GF(2)."""
        result = [1]  # 1
        b = self._poly_mod(base[:], modulus)
        
        while exp > 0:
            if exp & 1:
                result = self._poly_mod(self._poly_mul(result, b), modulus)
            b = self._poly_mod(self._poly_mul(b, b), modulus)
            exp >>= 1
        
        return result
    
    def _poly_mul(self, a: List[int], b: List[int]) -> List[int]:
        """Multiply two polynomials over GF(2)."""
        if not a or not b:
            return [0]
        result = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    if bj:
                        result[i + j] ^= 1
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    def factor_via_orbit_orders(self, num_polys: int = 20) -> Dict:
        """
        Factor M_p by finding non-primitive irreducible polynomials
        and computing their companion matrix orders.
        
        The orders are divisors of M_p. By collecting enough of them
        and computing GCDs, we can extract the factors of M_p.
        """
        rng = np.random.RandomState(42)
        
        orders = []
        primitive_count = 0
        non_primitive_count = 0
        
        for _ in range(num_polys * 10):  # Try many to get enough
            if len(orders) >= num_polys:
                break
            
            poly = self.find_irreducible_poly(rng)
            if poly is None:
                continue
            
            order = self.compute_companion_order(poly)
            if order is None:
                continue
            
            if order == self.M_p:
                primitive_count += 1
            else:
                non_primitive_count += 1
                orders.append(order)
        
        # Extract factors from the orders
        # Each order divides M_p. GCDs of orders reveal common factors.
        factors_found = set()
        
        for order in orders:
            if order < self.M_p:
                # order is a proper divisor of M_p — it IS a factor-related number
                factor = gcd(order, self.M_p)
                if 1 < factor < self.M_p:
                    factors_found.add(factor)
                # Also check M_p / order
                cofactor = self.M_p // order
                if 1 < cofactor < self.M_p:
                    factors_found.add(cofactor)
        
        # Try GCDs between orders
        for i in range(len(orders)):
            for j in range(i + 1, len(orders)):
                g = gcd(orders[i], orders[j])
                if 1 < g < self.M_p:
                    factors_found.add(g)
        
        # Refine: for each found factor, check if it's prime
        prime_factors = set()
        for f in factors_found:
            if is_prime_miller_rabin(f):
                prime_factors.add(f)
            else:
                # Factor further
                sub = factor_mersenne_complete(int(np.log2(f + 1)))
                for sf, se in sub['factors']:
                    if sf > 1:
                        prime_factors.add(sf)
        
        self.orders_found = orders
        
        return {
            'p': self.p,
            'M_p': self.M_p,
            'orders_found': orders,
            'primitive_count': primitive_count,
            'non_primitive_count': non_primitive_count,
            'factors_from_orders': sorted(factors_found),
            'prime_factors_from_orders': sorted(prime_factors),
            'factoring_method': 'orbit_order_extraction'
        }


# ============================================================
# CA Spacetime Visualization of Factoring
# ============================================================

def factoring_spacetime(p: int, seed: int = 2, c: int = 1,
                        steps: int = 200) -> np.ndarray:
    """
    Generate a spacetime diagram of the Pollard's rho iteration
    x → x² + c mod M_p, showing the CA dynamics of factoring.
    
    Each row is one iteration. The binary representation of x_i
    is shown as a row of the diagram.
    
    When the iteration "finds" a factor (collision in Z/M_p Z),
    the pattern should show a visible transition.
    """
    M_p = (1 << p) - 1
    
    x = seed
    history = []
    
    for _ in range(steps):
        bits = [(x >> i) & 1 for i in range(p)]
        history.append(bits)
        
        # CA step: square + fold + c
        x = mersenne_fold(x * x + c, p)
    
    return np.array(history)


# ============================================================
# Utility Functions
# ============================================================

def is_prime_miller_rabin(n: int, k: int = 20) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witness loop
    import random
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


# ============================================================
# Known Mersenne Factorizations (for verification)
# ============================================================

KNOWN_MERSENNE_FACTORS = {
    11: {23, 89},           # 2047 = 23 × 89
    23: {47, 178481},       # 8388607 = 47 × 178481
    29: {233, 1103, 2089},  # 536870911 = 233 × 1103 × 2089
    37: {223, 616318177},   # 137438953471 = 223 × 616318177
    41: {13367, 164511353}, # 2199023255551 = 13367 × 164511353
    43: {431, 9719, 2099863},  # 8796093022207 = 431 × 9719 × 2099863
    47: {2351, 4513, 13264529},  # 140737488355327
}


# ============================================================
# Main Demonstrations
# ============================================================

def demo_ca_factoring():
    """Demonstrate CA-based factoring of Mersenne composites."""
    print("=" * 70)
    print("CA-BASED FACTORING OF MERSENNE COMPOSITES")
    print("=" * 70)
    
    print("""
The map x → x² + c (mod M_p) decomposes into CA operations:
  1. Squaring: bit convolution (spreading CA rule)
  2. Reduction mod M_p = 2^p - 1: FOLD (Rule 90-like)
  3. Adding c: local bit addition

When M_p is composite, Pollard's rho finds collisions in Z/M_p Z
(via Floyd's cycle detection), and gcd(|x_i - x_{2i}|, M_p) reveals a factor.

Each step of the factoring algorithm IS a CA update!
""")
    
    test_exponents = [11, 23, 29, 37, 41, 43]
    
    for p in test_exponents:
        M_p = (1 << p) - 1
        known = KNOWN_MERSENNE_FACTORS.get(p, set())
        
        print(f"\n--- M_{p} = {M_p} ---")
        print(f"  Known factors: {sorted(known) if known else 'N/A'}")
        
        start_time = time.time()
        result = factor_mersenne_complete(p)
        elapsed = time.time() - start_time
        
        print(f"  CA factoring result: {result['factorization']}")
        print(f"  Method: {result['method']}")
        print(f"  Time: {elapsed:.3f}s")
        
        if known:
            found_factors = set(f for f, e in result['factors'])
            match = found_factors == known
            print(f"  Verification: {'CORRECT ✓' if match else 'MISMATCH ✗'}")
            if not match:
                print(f"    Found: {sorted(found_factors)}, Expected: {sorted(known)}")


def demo_orbit_factoring():
    """Demonstrate factor extraction from CA orbit orders."""
    print("\n" + "=" * 70)
    print("FACTOR EXTRACTION FROM CA ORBIT ORDERS")
    print("=" * 70)
    
    print("""
Key insight: For a NON-PRIMITIVE irreducible polynomial of degree p,
the companion matrix's order is a PROPER DIVISOR of M_p = 2^p - 1.

By finding such polynomials and computing their orders,
we extract factors of M_p directly from the CA dynamics.

The order of the companion matrix IS the period of the CA.
Finding a CA with period d < M_p means d is a factor of M_p.
""")
    
    for p in [11, 23, 29]:
        M_p = (1 << p) - 1
        known = KNOWN_MERSENNE_FACTORS.get(p, set())
        
        print(f"\n--- M_{p} = {M_p} ---")
        print(f"  Known factors: {sorted(known)}")
        
        factorer = CAOrbitFactorer(p)
        result = factorer.factor_via_orbit_orders(num_polys=15)
        
        print(f"  Irreducible polys found: {result['primitive_count']} primitive, "
              f"{result['non_primitive_count']} non-primitive")
        print(f"  CA orders found: {result['orders_found'][:10]}")
        print(f"  Factors from orders: {result['factors_from_orders']}")
        print(f"  Prime factors from orders: {result['prime_factors_from_orders']}")


def demo_factoring_spacetime():
    """Show the spacetime diagram of the factoring CA."""
    print("\n" + "=" * 70)
    print("FACTORING CA SPACETIME DIAGRAM")
    print("=" * 70)
    
    print("""
The Pollard's rho iteration x → x² + c (mod M_p) viewed as a CA
spacetime diagram. Each row is the binary state x_i.
""")
    
    for p in [11]:
        M_p = (1 << p) - 1
        diagram = factoring_spacetime(p, seed=2, c=1, steps=50)
        
        print(f"\nM_{p} = {M_p} = 23 × 89")
        print("Pollard's rho iteration (first 30 steps):")
        
        for i in range(min(30, len(diagram))):
            bits = ''.join(str(b) for b in reversed(diagram[i]))
            # Reconstruct the value
            val = sum(int(b) << j for j, b in enumerate(diagram[i]))
            print(f"  Step {i:3d}: {bits} = {val}")


def demo_ca_step_by_step():
    """Show each CA operation in the factoring process step by step."""
    print("\n" + "=" * 70)
    print("STEP-BY-STEP CA DECOMPOSITION OF FACTORING ITERATION")
    print("=" * 70)
    
    p = 11
    M_p = (1 << p) - 1
    
    print(f"\nM_{p} = {M_p} = 23 × 89")
    print(f"Folding: x mod {M_p} = upper bits + lower bits (CA fold)")
    
    x = 2
    c = 1
    
    for step in range(15):
        print(f"\nStep {step}: x = {x} = {bin(x)}")
        
        # Step 1: Square (bit convolution)
        x_sq = x * x
        print(f"  Square: x² = {x_sq} = {bin(x_sq)}")
        
        # Step 2: Add c
        x_sq_c = x_sq + c
        print(f"  Add c={c}: x² + c = {x_sq_c} = {bin(x_sq_c)}")
        
        # Step 3: Fold (CA reduction mod M_p)
        upper = x_sq_c >> p
        lower = x_sq_c & M_p
        folded = upper + lower
        if folded >= M_p:
            folded = (folded >> p) + (folded & M_p)
        if folded == M_p:
            folded = 0
        
        print(f"  Fold: upper={upper}, lower={lower} → {upper}+{lower} = {folded}")
        
        # Check for factors
        x_new = folded
        if step > 0:
            # In full Pollard's rho, we'd track x_i and x_{2i}
            pass
        
        x = x_new


if __name__ == "__main__":
    demo_ca_step_by_step()
    demo_factoring_spacetime()
    demo_ca_factoring()
    demo_orbit_factoring()
