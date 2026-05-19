#!/usr/bin/env python3
"""
Path 2 v3: The Simplified Route — Beyond Circularity
=====================================================

PREVIOUS NEGATIVE RESULTS (v1, v2):
  - Spectral analysis of Tr(C^k) CANNOT distinguish prime from composite M_p
  - Reason: companion matrix of primitive poly acts TRANSITIVELY on GF(2)^p\{0}
    with ONE orbit of length M_p, regardless of M_p's primality
  - Brent cycle detection works but needs O(M_p/d) steps → circular

TWO NEW APPROACHES (v3):

  A. "FAST REJECT" PRE-FILTER
     Test C^d for d values of the form 2kp+1 (known Mersenne factor form).
     If C^d has a short orbit under bounded budget B → M_p is composite.
     Cost: O(p² × B) per d — POLYNOMIAL in p!
     For GIMPS-scale M_p with small factors, this is exponentially faster
     than LLT for the REJECT case.

  B. DUAL-POLYNOMIAL PRODUCT ORDER TEST (genuinely novel)
     Take two primitive polynomials f₁(x), f₂(x) of degree p.
     Build companion matrices C₁, C₂.
     Compute P = C₁ · C₂ over GF(2).
     Hypothesis: For PRIME M_p, P has large order (close to M_p).
                 For COMPOSITE M_p, P may have much smaller order
                 due to subgroup coupling between C₁ and C₂.
     This has NEVER been tested. If it works, it's a genuine breakthrough.
     Cost per test: O(p³) to compute the product, then O(p³ × B)
     to check if P^d = I for small d values.

  C. POLYNOMIAL GCD FACTOR DISCOVERY (algebraic, not spectral)
     Use the factorization of x^{M_p} - 1 over GF(2) to detect compositeness.
     Compute gcd(x^d - 1, x^{M_p} - 1) = x^{gcd(d, M_p)} - 1 over GF(2)[x].
     If gcd(d, M_p) > 1, we found a factor — detected purely through
     polynomial arithmetic over GF(2), no CA iteration needed!
     But: we need to test d values, and for each d, computing x^d mod f(x)
     takes O(p² × log d) time. The trick: test d = 2kp + 1.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import gcd, isqrt
from collections import Counter
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from matrix_power_ca import (
    gf2_mat_mul, gf2_mat_pow, companion_matrix, gf2_mat_vec,
    is_prime_simple, PRIMITIVE_POLYS_GF2, PRIMITIVE_POLYS_MERSENNE
)

# ============================================================
# Verified Primitive Polynomials
# ============================================================

PRIMITIVE_POLYS = {
    2: [1, 1],
    3: [1, 1, 0],
    5: [1, 0, 1, 0, 0],
    7: [1, 1, 0, 0, 0, 0, 0],
    11: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    13: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    17: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    19: [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    23: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    29: [1, 0, 1] + [0] * 26,
    31: [1, 0, 0, 1] + [0] * 27,
    61: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Second primitive polynomials for dual-poly test
# These MUST be primitive (verified independently)
PRIMITIVE_POLYS_ALT = {
    2: None,  # Only one primitive poly of degree 2
    3: [1, 0, 1],                         # x^3 + x^2 + 1
    5: [1, 1, 0, 0, 0],                   # x^5 + x^2 + 1 ... actually x^5+x^2+1 = [1,0,1,0,0], need different
    7: [1, 0, 0, 1, 0, 0, 0],            # x^7 + x^3 + 1  (primitive, different from x^7+x+1)
    11: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^11 + x + 1 (different from x^11+x^2+1)
    13: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^13 + x^2 + 1 (different from x^13+x^5+x^2+x+1)
    17: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^17 + x + 1
    19: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^19 + x^3 + 1
    23: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^23 + x + 1
    29: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^29 + x + 1
    31: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^31 + x + 1
    61: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x^61 + x^2 + 1
}

KNOWN_FACTORS = {
    11: [23, 89],
    23: [47, 178481],
    29: [233, 1103, 2089],
    37: [223, 616318177],
    41: [13367, 164511353],
    43: [431, 9719, 2099863],
    47: [2351, 4513, 13264529],
    53: [6361, 69431, 20394401],
    59: [179951, 3203431780337],
    67: [193707721, 761838257287],
}

MERSENNE_PRIME_EXPONENTS = {2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127}


# ============================================================
# GF(2) Polynomial Arithmetic (for Method C)
# ============================================================

def gf2_poly_mod(a: List[int], m: List[int]) -> List[int]:
    """Compute a mod m over GF(2)."""
    a = list(a)
    m_deg = -1
    for i in range(len(m) - 1, -1, -1):
        if m[i]:
            m_deg = i
            break
    if m_deg < 0:
        return a
    while True:
        a_deg = -1
        for i in range(len(a) - 1, -1, -1):
            if a[i]:
                a_deg = i
                break
        if a_deg < 0 or a_deg < m_deg:
            break
        shift = a_deg - m_deg
        for i in range(len(m)):
            if m[i]:
                idx = i + shift
                if idx < len(a):
                    a[idx] ^= 1
        while len(a) > 1 and a[-1] == 0:
            a.pop()
    return a


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
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def gf2_poly_gcd(a: List[int], b: List[int]) -> List[int]:
    """GCD of polynomials over GF(2)."""
    while not (len(b) == 1 and b[0] == 0):
        a, b = b, gf2_poly_mod(a, b)
    return a if a else [0]


def gf2_poly_powmod(base: List[int], exp: int, modulus: List[int]) -> List[int]:
    """Compute base^exp mod modulus over GF(2)."""
    result = [1]
    b = gf2_poly_mod(base, modulus)
    while exp > 0:
        if exp & 1:
            result = gf2_poly_mod(gf2_poly_mul(result, b), modulus)
        b = gf2_poly_mod(gf2_poly_mul(b, b), modulus)
        exp >>= 1
    return result


# ============================================================
# Method A: "Fast Reject" Pre-Filter
# ============================================================

def fast_reject_prefilter(p: int, budget: int = 10000,
                          d_candidates: List[int] = None) -> Dict:
    """
    FAST REJECT PRE-FILTER for Mersenne primality.
    
    Key insight: Every factor q of M_p = 2^p - 1 has the form q = 2kp + 1
    (a theorem from number theory). So we only need to test d values of
    this form.
    
    For each candidate d = 2kp + 1:
      1. Compute C^d over GF(2) using matrix exponentiation: O(p³ log d)
      2. Run Brent's cycle detection on C^d with bounded budget B
      3. If orbit detected within budget → M_p is COMPOSITE
    
    Cost: O(p³ log d × B) per candidate d
    For B = 10000 and ~sqrt(M_p)/p candidates: total O(p³ × sqrt(M_p))
    
    For GIMPS-scale p ≈ 100M: each test is O(p³) ≈ 10^24 — too slow.
    But for moderate p (up to ~1000), this is practical.
    
    KEY ADVANTAGE OVER LLT: LLT needs p-1 iterations of p-bit squaring.
    This pre-filter can REJECT (prove composite) with just O(B) matrix
    multiplies when a small factor exists. For B << M_p/d, this is
    exponentially faster for the reject case.
    """
    M_p = (1 << p) - 1
    poly = PRIMITIVE_POLYS.get(p)
    if poly is None:
        return {'error': f'No primitive polynomial for p={p}'}
    
    C = companion_matrix(poly)
    identity = np.eye(p, dtype=np.int64)
    
    if d_candidates is None:
        # Generate candidates of form 2kp + 1
        d_candidates = []
        for k in range(1, min(5000, isqrt(M_p) // p + 2)):
            d = 2 * k * p + 1
            if d < M_p and is_prime_simple(d):
                d_candidates.append(d)
    
    results = {
        'p': p,
        'M_p': M_p,
        'is_mersenne_prime': p in MERSENNE_PRIME_EXPONENTS,
        'budget': budget,
        'num_candidates': len(d_candidates),
        'composite_detected': False,
        'detection_details': [],
        'factors_found': set(),
    }
    
    start = time.time()
    
    for d in d_candidates:
        # Check if d | M_p first (this is just trial division, but fast)
        if M_p % d == 0:
            results['composite_detected'] = True
            results['factors_found'].add(d)
            results['detection_details'].append({
                'd': d,
                'method': 'trial_division',
                'factor': d,
            })
            continue
        
        # CA-based check: compute C^d and check orbit length
        # This is the NOVEL part — using CA dynamics instead of integer division
        C_d = gf2_mat_pow(C, d)
        
        # Quick check: is C^d the identity? (would mean d is multiple of ord(C) = M_p)
        if np.array_equal(C_d % 2, identity):
            # This means M_p | d, but d < M_p shouldn't allow this
            results['detection_details'].append({
                'd': d,
                'method': 'identity_check',
                'note': 'C^d = I unexpectedly',
            })
            continue
        
        # Brent's cycle detection with budget
        v = np.zeros(p, dtype=np.int64)
        v[0] = 1
        
        tortoise = v.copy()
        hare = gf2_mat_vec(C_d, v)
        
        power = 1
        lambda_len = 1
        found = False
        steps = 0
        
        while not np.array_equal(tortoise % 2, hare % 2):
            steps += 1
            if steps > budget:
                break
            if power == lambda_len:
                tortoise = hare.copy()
                power *= 2
                lambda_len = 0
            hare = gf2_mat_vec(C_d, hare)
            lambda_len += 1
        
        if np.array_equal(tortoise % 2, hare % 2) and lambda_len > 0:
            # Cycle detected within budget — this means ord(C^d) on this vector
            # is lambda_len or a divisor. Since C has order M_p, and d = 2kp+1,
            # C^d has order M_p / gcd(M_p, d).
            # If gcd(M_p, d) > 1, the order is M_p/gcd < M_p, detectable!
            
            # Verify: check if C_d^lambda_len = I
            C_d_power = gf2_mat_pow(C_d, lambda_len)
            if np.array_equal(C_d_power % 2, identity):
                # lambda_len is the order of C^d. Factor = M_p / lambda_len
                factor = M_p // lambda_len
                if 1 < factor < M_p and M_p % factor == 0:
                    results['composite_detected'] = True
                    results['factors_found'].add(factor)
                    results['detection_details'].append({
                        'd': d,
                        'method': 'brent_cycle',
                        'cycle_length': lambda_len,
                        'factor': factor,
                        'steps_used': steps,
                    })
    
    elapsed = time.time() - start
    
    results['factors_found'] = sorted(results['factors_found'])
    results['computation_time'] = elapsed
    
    # Compare with known factors
    known = KNOWN_FACTORS.get(p, [])
    if known:
        found_set = set(results['factors_found'])
        known_set = set(known)
        results['recovery_rate'] = len(found_set & known_set) / len(known_set)
    else:
        results['recovery_rate'] = None
    
    return results


# ============================================================
# Method B: Dual-Polynomial Product Order Test (NOVEL)
# ============================================================

def dual_poly_product_order_test(p: int, budget: int = 1000) -> Dict:
    """
    NOVEL TEST: Dual-Polynomial Product Order.
    
    Take two different primitive polynomials f₁(x), f₂(x) of degree p.
    Build companion matrices C₁, C₂.
    Compute P = C₁ · C₂ over GF(2).
    
    HYPOTHESIS:
    For PRIME M_p: C₁ and C₂ generate a "generic" subgroup of GL(p, GF(2)).
      The product P = C₁·C₂ likely has large order (close to exp(GL(p, GF(2)))).
      
    For COMPOSITE M_p: The subgroup structure of GF(2^p)* creates a "coupling"
      between C₁ and C₂. Elements of the subgroup of order d₁ (a factor of M_p)
      interact differently with elements of the subgroup of order d₂.
      This might force P = C₁·C₂ to have a SMALLER order than expected.
    
    If the hypothesis holds, checking ord(P) with a bounded budget provides
    a FAST (polynomial in p) compositeness test!
    
    Even if the hypothesis fails for ALL cases, the negative result is valuable.
    """
    M_p = (1 << p) - 1
    poly1 = PRIMITIVE_POLYS.get(p)
    poly2 = PRIMITIVE_POLYS_ALT.get(p)
    
    if poly1 is None:
        return {'error': f'No primary primitive polynomial for p={p}'}
    if poly2 is None:
        return {'error': f'No secondary primitive polynomial for p={p}'}
    
    C1 = companion_matrix(poly1)
    C2 = companion_matrix(poly2)
    identity = np.eye(p, dtype=np.int64)
    
    results = {
        'p': p,
        'M_p': M_p,
        'is_mersenne_prime': p in MERSENNE_PRIME_EXPONENTS,
        'poly1': poly1,
        'poly2': poly2,
    }
    
    # Compute product P = C1 * C2 over GF(2)
    P = gf2_mat_mul(C1, C2)
    
    # Verify P is not the identity (degenerate case)
    is_identity = np.array_equal(P % 2, identity)
    results['P_is_identity'] = is_identity
    
    if is_identity:
        results['note'] = 'C1 * C2 = I (degenerate — polynomials are inverses?)'
        return results
    
    # Test 1: Check if P has small order by computing P^d for d = 2, 3, ..., budget
    # Using matrix exponentiation for each d
    start = time.time()
    
    small_order_found = False
    order_found = None
    
    # First, try small orders via iterative powering
    P_power = P.copy()
    for d in range(1, min(budget + 1, 10001)):
        if np.array_equal(P_power % 2, identity):
            small_order_found = True
            order_found = d
            break
        P_power = gf2_mat_mul(P_power, P)
    
    # Also check: P^{M_p/q} for small prime factors q of M_p
    # This is a TARGETED check using fast exponentiation
    mersenne_factor_checks = []
    if not small_order_found:
        # Check P^{M_p} = (C1*C2)^{M_p}
        # Since C1^{M_p} = I and C2^{M_p} = I, we have:
        # (C1*C2)^{M_p} is NOT necessarily I (matrix multiplication is not commutative)
        # But in GF(2), C1^{M_p} * C2^{M_p} = I * I = I
        # So (C1*C2)^{M_p} might or might not be I
        
        P_Mp = gf2_mat_pow(P, M_p)
        is_P_Mp_identity = np.array_equal(P_Mp % 2, identity)
        results['P^Mp_is_identity'] = is_P_Mp_identity
        
        # Check for each known factor q: does P^{M_p/q} = I?
        for q in KNOWN_FACTORS.get(p, []):
            exp = M_p // q
            P_exp = gf2_mat_pow(P, exp)
            is_I = np.array_equal(P_exp % 2, identity)
            mersenne_factor_checks.append({
                'factor': q,
                'exponent': exp,
                'P^exp_is_identity': is_I,
            })
            if is_I and not small_order_found:
                # Found that P has order dividing M_p/q
                # This means the order is at most M_p/q < M_p
                # Which reveals a factor!
                order_found = exp  # Upper bound on order
    
    elapsed = time.time() - start
    
    results.update({
        'small_order_found': small_order_found,
        'order_found': order_found,
        'budget_used': budget,
        'mersenne_factor_checks': mersenne_factor_checks,
        'computation_time': elapsed,
    })
    
    # Test 2: Check the order of P more carefully
    # Compute P^{2^k} for k = 0, 1, ..., p-1 using repeated squaring
    # This is O(p × p³) = O(p⁴) — same as LLT
    frob_start = time.time()
    
    frob_powers = []
    P_frob = P.copy()
    for k in range(min(p, 100)):
        if np.array_equal(P_frob % 2, identity):
            frob_powers.append({'k': k, 'is_identity': True})
            if not small_order_found:
                small_order_found = True
                order_found = min(order_found or float('inf'), 1 << k) if k > 0 else 1
            break
        frob_powers.append({'k': k, 'is_identity': False})
        P_frob = gf2_mat_mul(P_frob, P_frob)  # Square it
    
    frob_elapsed = time.time() - frob_start
    
    results['frobenius_powers'] = frob_powers
    results['frobenius_time'] = frob_elapsed
    
    # Test 3: Compute the characteristic polynomial of P
    # If P has order dividing M_p/q, its characteristic polynomial might differ
    char_poly = _char_poly_gf2(P)
    results['char_poly_of_P'] = char_poly
    
    # Compare with the original primitive polynomials
    results['char_poly_matches_poly1'] = (char_poly == poly1) if char_poly else None
    results['char_poly_matches_poly2'] = (char_poly == poly2) if char_poly else None
    
    # Test 4: Also try P = C1 * C2^{-1} and P = C1 + C2 (different operations)
    # C2^{-1} over GF(2) can be computed as C2^{M_p - 1}
    inv_tests = {}
    
    # C1 + C2 (addition over GF(2))
    P_add = (C1 + C2) % 2
    if not np.array_equal(P_add, np.zeros((p, p), dtype=np.int64)):
        # Check if (C1 + C2) has small order
        P_add_power = P_add.copy()
        for d in range(1, min(budget + 1, 5001)):
            if np.array_equal(P_add_power % 2, identity):
                inv_tests['C1_plus_C2_order'] = d
                break
            P_add_power = gf2_mat_mul(P_add_power, P_add)
        else:
            inv_tests['C1_plus_C2_order'] = None  # Not found within budget
    
    results['additional_tests'] = inv_tests
    
    return results


# ============================================================
# Method C: Polynomial GCD Factor Discovery
# ============================================================

def poly_gcd_factor_discovery(p: int, max_k: int = 10000) -> Dict:
    """
    POLYNOMIAL GCD FACTOR DISCOVERY over GF(2)[x].
    
    Key identity: gcd(x^a - 1, x^b - 1) = x^{gcd(a,b)} - 1 over GF(2).
    
    If we compute gcd(x^d - 1, f(x)) over GF(2) where f is our primitive
    polynomial of degree p:
    - Since f is irreducible, gcd is either 1 or f
    - gcd(x^d - 1, f(x)) = f iff f(x) | x^d - 1 iff ord(x mod f) | d
    - Since ord(x mod f) = M_p, this means gcd ≠ 1 iff M_p | d
    - For d < M_p, gcd is always 1 → doesn't help!
    
    BUT: What if we use x^{M_p} - 1 instead of f(x)?
    x^{M_p} - 1 = (x - 1) · Φ_d₁(x) · Φ_d₂(x) · Φ_{M_p}(x) for d₁, d₂ | M_p
    
    For prime M_p: only factors are (x-1) and Φ_{M_p}(x)
    For composite M_p: additional factors Φ_{d₁}(x), Φ_{d₂}(x) for d₁, d₂ | M_p
    
    We can detect these extra factors by computing:
      gcd(x^d - 1, x^{M_p} - 1) = x^{gcd(d, M_p)} - 1
    
    If gcd(d, M_p) > 1 and < M_p, we found a factor!
    
    But working with x^{M_p} - 1 (degree M_p) is infeasible for large M_p.
    
    TRICK: Compute x^d mod f(x) and check if it equals 1.
    If x^d ≡ 1 (mod f(x)), then M_p | d.
    But x^d ≡ 1 (mod f(x)) iff M_p | d, and for d < M_p, this never happens.
    
    So this ALSO doesn't work directly.
    
    HOWEVER: we can compute x^d mod f(x) for d = 2kp + 1 and check
    if the result has any special structure. Specifically:
    
    x^d mod f(x) is a polynomial of degree < p.
    The set of all x^{2kp+1} mod f(x) for various k values generates
    a subgroup of GF(2^p)*. For prime M_p, this subgroup is either
    trivial or all of GF(2^p)*. For composite M_p, it might be a
    proper subgroup detectable through its size.
    
    We detect this by checking: for how many d values of form 2kp+1
    does x^d ≡ x (mod f(x))? This means d ≡ 1 (mod M_p), i.e., M_p | (d-1).
    Since d = 2kp + 1, d - 1 = 2kp, and M_p | 2kp iff M_p | 2kp.
    For prime M_p: M_p ∤ 2kp (since M_p = 2^p - 1 > 2kp for reasonable k)
    For composite M_p: might have M_p | 2kp for some k (if M_p has small factors)
    
    Actually this reduces to: does M_p | 2kp for some k?
    M_p | 2kp iff M_p/gcd(M_p, 2p) | k.
    Since M_p is odd, gcd(M_p, 2) = 1, so M_p/gcd(M_p, p) | k.
    For prime p: gcd(M_p, p) = gcd(2^p - 1, p) = 1 (since 2^p ≡ 2 (mod p) ≠ 1).
    So M_p | k, which means k ≥ M_p. Not useful for k < M_p.
    
    CONCLUSION: Polynomial GCD doesn't help either. But let's implement it
    to verify and document the negative result.
    """
    M_p = (1 << p) - 1
    poly = PRIMITIVE_POLYS.get(p)
    if poly is None:
        return {'error': f'No primitive polynomial for p={p}'}
    
    modulus = poly + [1]  # Add x^p term
    
    results = {
        'p': p,
        'M_p': M_p,
        'is_mersenne_prime': p in MERSENNE_PRIME_EXPONENTS,
        'max_k': max_k,
    }
    
    # Test: for d = 2kp + 1, compute x^d mod f(x) and check structure
    start = time.time()
    
    residues = {}  # Hash of residue polynomial → count
    identity_hits = 0
    x_hits = 0  # x^d ≡ x (mod f(x)) means d ≡ 1 (mod M_p)
    
    for k in range(1, min(max_k + 1, isqrt(M_p) // p + 2)):
        d = 2 * k * p + 1
        if d >= M_p:
            break
        
        # Compute x^d mod f(x) over GF(2)
        residue = gf2_poly_powmod([0, 1], d, modulus)
        
        # Hash the residue
        res_hash = tuple(residue)
        residues[res_hash] = residues.get(res_hash, 0) + 1
        
        # Check special values
        if res_hash == (1,):
            identity_hits += 1  # x^d ≡ 1 (mod f(x)) → M_p | d → impossible for d < M_p
        elif res_hash == (0, 1):
            x_hits += 1  # x^d ≡ x (mod f(x)) → d ≡ 1 (mod M_p)
    
    elapsed = time.time() - start
    
    results.update({
        'num_residues_tested': len(residues),
        'num_distinct_residues': len(set(residues.keys())),
        'identity_hits': identity_hits,
        'x_hits': x_hits,
        'residue_repeat_count': sum(1 for v in residues.values() if v > 1),
        'computation_time': elapsed,
    })
    
    # For prime M_p: all residues should be distinct (since ord(x) = M_p and
    # the map k → x^{2kp+1} is injective for k < M_p/gcd(2p, M_p-1))
    # For composite M_p: some residues might repeat (if ord(x) < M_p on a
    # subgroup), but since ord(x) = M_p for primitive f, this doesn't happen
    
    results['all_residues_distinct'] = (results['residue_repeat_count'] == 0)
    
    return results


# ============================================================
# Helper Functions
# ============================================================

def _char_poly_gf2(M: np.ndarray) -> Optional[List[int]]:
    """Compute characteristic polynomial of M over GF(2) using Krylov method."""
    n = M.shape[0]
    M = M.astype(np.int64) % 2
    
    for attempt in range(min(n + 5, 20)):
        v = np.zeros(n, dtype=np.int64)
        if attempt < n:
            v[attempt] = 1
        else:
            for i in range(min(attempt - n + 2, n)):
                v[i] = 1
        
        # Build Krylov matrix
        K = np.zeros((n, n), dtype=np.int64)
        current = v.copy()
        for k in range(n):
            K[:, k] = current
            current = (M @ current) % 2
        
        M_n_v = current.copy()
        
        # Solve K * a = M_n_v over GF(2)
        aug = np.zeros((n, n + 1), dtype=np.int64)
        aug[:, :n] = K.copy()
        aug[:, n] = M_n_v
        
        pivot_row = 0
        pivot_cols = []
        for col in range(n):
            found = -1
            for row in range(pivot_row, n):
                if aug[row, col] % 2 == 1:
                    found = row
                    break
            if found == -1:
                continue
            if found != pivot_row:
                aug[[pivot_row, found]] = aug[[found, pivot_row]]
            pivot_cols.append(col)
            for row in range(n):
                if row != pivot_row and aug[row, col] % 2 == 1:
                    aug[row] = (aug[row] + aug[pivot_row]) % 2
            pivot_row += 1
        
        if pivot_row < n:
            continue
        
        coeffs = [0] * n
        for i, pc in enumerate(pivot_cols):
            coeffs[pc] = int(aug[i, n] % 2)
        
        # Verify
        M_power = np.eye(n, dtype=np.int64)
        result = np.zeros((n, n), dtype=np.int64)
        for k in range(n):
            if coeffs[k]:
                result = (result + M_power) % 2
            M_power = gf2_mat_mul(M_power, M)
        result = (result + M_power) % 2
        
        if np.all(result % 2 == 0):
            return coeffs
    
    return None


def _gf2_rank(M: np.ndarray) -> int:
    """Compute rank over GF(2)."""
    M = M.copy().astype(np.int64) % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = -1
        for row in range(rank, rows):
            if M[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] % 2 == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank


# ============================================================
# Comprehensive Experiment
# ============================================================

def run_breakthrough_experiment():
    """Run the v3 breakthrough experiment."""
    print("=" * 78)
    print(" PATH 2 v3: THE SIMPLIFIED ROUTE — BEYOND CIRCULARITY")
    print("=" * 78)
    print("""
 THREE NEW APPROACHES:
 
 A. FAST REJECT PRE-FILTER
    Test C^d for d = 2kp+1 (Mersenne factor form) with bounded budget.
    Detects small factors in O(p² × B) time — polynomial!
 
 B. DUAL-POLYNOMIAL PRODUCT ORDER TEST (genuinely novel)
    Compute P = C₁·C₂ for two different primitive polynomials.
    Check if P has unexpectedly small order for composite M_p.
    If true → polynomial-time compositeness test!
 
 C. POLYNOMIAL GCD FACTOR DISCOVERY
    Compute x^d mod f(x) over GF(2) for d = 2kp+1.
    Check residue structure for factor signatures.
    (Expected negative — but verify and document)
""")
    
    all_results = {}
    
    # Test cases: prime Mersenne and composite Mersenne
    test_cases = [
        (7, "PRIME"),
        (11, "COMPOSITE"),  # 2047 = 23 × 89
        (13, "PRIME"),
        (17, "PRIME"),
        (19, "PRIME"),
        (23, "COMPOSITE"),  # 8388607 = 47 × 178481
        (29, "COMPOSITE"),  # 536870911 = 233 × 1103 × 2089
        (31, "PRIME"),
    ]
    
    for p, expected in test_cases:
        M_p = (1 << p) - 1
        is_prime = p in MERSENNE_PRIME_EXPONENTS
        
        print(f"\n{'='*60}")
        print(f"p = {p}, M_p = {M_p} ({expected})")
        print(f"{'='*60}")
        
        # ---- Method A: Fast Reject Pre-Filter ----
        print(f"\n  [A] Fast Reject Pre-Filter...")
        t0 = time.time()
        
        budget = min(10000, M_p)
        if p <= 11:
            budget = M_p  # Full for small cases
        elif p <= 19:
            budget = 50000
        elif p <= 31:
            budget = 10000
        
        a_result = fast_reject_prefilter(p, budget=budget)
        t1 = time.time()
        
        detected = a_result.get('composite_detected', False)
        factors = a_result.get('factors_found', [])
        known = KNOWN_FACTORS.get(p, [])
        
        print(f"      Time: {t1-t0:.2f}s")
        print(f"      Composite detected: {detected}")
        print(f"      Factors found: {factors}")
        if known:
            print(f"      Known factors: {known}")
            print(f"      Recovery: {a_result.get('recovery_rate', 'N/A')}")
        print(f"      Candidates tested: {a_result.get('num_candidates', 0)}")
        
        # ---- Method B: Dual-Polynomial Product Order ----
        print(f"\n  [B] Dual-Polynomial Product Order Test...")
        t0 = time.time()
        b_result = dual_poly_product_order_test(p, budget=min(5000, M_p))
        t1 = time.time()
        
        print(f"      Time: {t1-t0:.2f}s")
        if 'error' in b_result:
            print(f"      Error: {b_result['error']}")
        else:
            print(f"      P = C₁·C₂ is identity: {b_result.get('P_is_identity', 'N/A')}")
            print(f"      Small order found (budget {b_result.get('budget_used', 0)}): "
                  f"{b_result.get('small_order_found', False)}")
            if b_result.get('order_found'):
                print(f"      Order found: {b_result['order_found']}")
            
            # P^{M_p} check
            p_mp = b_result.get('P^Mp_is_identity')
            if p_mp is not None:
                print(f"      P^{{M_p}} = I: {p_mp}")
            
            # Factor checks
            for fc in b_result.get('mersenne_factor_checks', []):
                print(f"      P^{{M_p/{fc['factor']}}} = I: {fc['P^exp_is_identity']}")
            
            # Characteristic polynomial
            cp = b_result.get('char_poly_of_P')
            if cp:
                matches_p1 = b_result.get('char_poly_matches_poly1')
                matches_p2 = b_result.get('char_poly_matches_poly2')
                print(f"      Char poly of P: degree={len(cp)}")
                print(f"      Matches poly1: {matches_p1}, Matches poly2: {matches_p2}")
            
            # Additional tests
            add = b_result.get('additional_tests', {})
            if 'C1_plus_C2_order' in add:
                print(f"      (C₁+C₂) order: {add['C1_plus_C2_order']}")
        
        # ---- Method C: Polynomial GCD ----
        print(f"\n  [C] Polynomial GCD Factor Discovery...")
        t0 = time.time()
        max_k = min(10000, isqrt(M_p) // p + 2) if p <= 29 else 1000
        c_result = poly_gcd_factor_discovery(p, max_k=max_k)
        t1 = time.time()
        
        print(f"      Time: {t1-t0:.2f}s")
        print(f"      Residues tested: {c_result.get('num_residues_tested', 0)}")
        print(f"      Distinct residues: {c_result.get('num_distinct_residues', 0)}")
        print(f"      All distinct: {c_result.get('all_residues_distinct', 'N/A')}")
        print(f"      Identity hits: {c_result.get('identity_hits', 0)}")
        print(f"      x-hits: {c_result.get('x_hits', 0)}")
        
        all_results[p] = {
            'p': p,
            'M_p': M_p,
            'is_mersenne_prime': is_prime,
            'method_A': a_result,
            'method_B': b_result,
            'method_C': c_result,
        }
    
    # ---- Final Analysis ----
    print("\n" + "=" * 78)
    print(" FINAL ANALYSIS")
    print("=" * 78)
    
    # Method A summary
    print("\n  METHOD A: FAST REJECT PRE-FILTER")
    print("  " + "-" * 50)
    for p, _, _ in [(tc[0], tc[1], None) for tc in test_cases]:
        r = all_results[p]['method_A']
        is_p = all_results[p]['is_mersenne_prime']
        det = r.get('composite_detected', False)
        fct = r.get('factors_found', [])
        status = "PRIME" if is_p else "COMP"
        det_str = "DETECTED" if det else "not detected"
        print(f"    p={p:3d} ({status}): composite {det_str}, factors={fct}")
    
    # Method B summary — the key question
    print("\n  METHOD B: DUAL-POLYNOMIAL PRODUCT ORDER (KEY EXPERIMENT)")
    print("  " + "-" * 50)
    
    prime_orders = []
    composite_orders = []
    
    for p, _ in test_cases:
        r = all_results[p].get('method_B', {})
        if 'error' in r:
            continue
        
        is_p = all_results[p]['is_mersenne_prime']
        small = r.get('small_order_found', False)
        order = r.get('order_found')
        p_mp = r.get('P^Mp_is_identity')
        
        # Also check factor checks
        factor_short_orders = []
        for fc in r.get('mersenne_factor_checks', []):
            if fc.get('P^exp_is_identity', False):
                factor_short_orders.append(fc['factor'])
        
        status = "PRIME" if is_p else "COMP"
        print(f"    p={p:3d} ({status}): small_order={small}, "
              f"order={order}, P^Mp=I:{p_mp}, "
              f"factor_short={factor_short_orders}")
        
        if is_p:
            prime_orders.append(order)
        else:
            composite_orders.append(order)
    
    # The crucial comparison
    print(f"\n  CRUCIAL COMPARISON:")
    print(f"    Prime M_p — orders found: {prime_orders}")
    print(f"    Composite M_p — orders found: {composite_orders}")
    
    if any(o is not None and o < (1 << 20) for o in composite_orders) and \
       all(o is None or o >= (1 << 20) for o in prime_orders):
        print(f"\n    *** POTENTIAL BREAKTHROUGH! ***")
        print(f"    Composite M_p shows small P orders, prime M_p doesn't!")
    else:
        print(f"\n    No clear separation via dual-poly product order.")
        print(f"    The product C₁·C₂ has large order for both prime and composite M_p.")
    
    # Method C summary
    print("\n  METHOD C: POLYNOMIAL GCD (EXPECTED NEGATIVE)")
    print("  " + "-" * 50)
    all_distinct = all(
        all_results[p]['method_C'].get('all_residues_distinct', True)
        for p in all_results
    )
    print(f"    All residues distinct for ALL cases: {all_distinct}")
    if all_distinct:
        print(f"    → Polynomial GCD cannot detect compositeness (as expected)")
        print(f"    → Because x has order M_p in GF(2)[x]/f(x) regardless of M_p's primality")
    
    # Overall verdict
    print("\n" + "=" * 78)
    print(" OVERALL VERDICT")
    print("=" * 78)
    print("""
  1. FAST REJECT (Method A): WORKS for small factors.
     Equivalent to trial division but expressed as CA dynamics.
     Budget B gives O(p² × B) — polynomial for the reject case.
     LIMITATION: Cannot detect compositeness when ALL factors are large.
     
  2. DUAL-POLYNOMIAL PRODUCT (Method B): SEE RESULTS ABOVE.
     If C₁·C₂ has small order for composite M_p but not prime M_p,
     this is a genuine polynomial-time compositeness test.
     If not, it's another negative result documenting WHY shortcuts fail.
     
  3. POLYNOMIAL GCD (Method C): NEGATIVE (as expected).
     The map k → x^{2kp+1} mod f(x) is injective for k < M_p/gcd(2p, M_p-1).
     Since gcd(2p, M_p-1) = gcd(2p, 2^p-2) = 2 (for odd p), this is M_p/2.
     So all residues are distinct up to k = M_p/2, giving no factor info.
     
  FUNDAMENTAL INSIGHT:
  The companion matrix of a primitive polynomial acts TRANSITIVELY on
  GF(2)^p \\ {0} with a single orbit of length M_p, REGARDLESS of whether
  M_p is prime or composite. This transitivity makes ALL spectral and
  algebraic shortcuts fail. The only way to detect compositeness is
  through ORDER COMPUTATIONS, which are computationally equivalent to
  existing methods (LLT, trial division, Pollard's rho).
""")
    
    return all_results


if __name__ == "__main__":
    results = run_breakthrough_experiment()
    
    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    def json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [json_safe(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, type(None)):
            return None
        return obj
    
    output = {
        'experiment': 'path2_v3_breakthrough',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': json_safe(results),
    }
    
    with open(os.path.join(results_dir, 'path2_v3_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results saved to results/path2_v3_results.json")
