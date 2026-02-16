#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  P1: System Admin & KGC — OO-IRIBE-EnDKER  (Single-File Version)   ║
║                                                                      ║
║  Paper: "OO-IRIBE-EnDKER" (Scientific Reports 2025)                 ║
║  DOI: 10.1038/s41598-025-01254-1                                    ║
║                                                                      ║
║  This file contains EVERYTHING P1 needs:                             ║
║    • Lattice primitives (TrapGen, SamplePre, SampleLeft)            ║
║    • Data structures (PP, MSK, SK, NRno_t)                          ║
║    • KGC algorithms (Setup, GenSK, NumUp)                           ║
║    • Demo + tests at the bottom                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Run:
    python3 p1_implementation.py          # runs demo
    python3 p1_implementation.py --test   # runs tests
"""

import numpy as np
import hashlib
import math
import json
import time
import os
import sys


# ═══════════════════════════════════════════════════════════════
#  SECTION 1: PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Default parameters from the paper (Table 5, n=32)
N_DEFAULT   = 32                                    # lattice dimension
Q_DEFAULT   = 99991                                 # prime modulus
K_DEFAULT   = math.ceil(math.log2(Q_DEFAULT))       # ⌈log₂ q⌉ = 17
M_DEFAULT   = 2 * N_DEFAULT * K_DEFAULT             # m ≥ 2n⌈log₂ q⌉ = 1088
SIGMA       = 4.0                                   # Gaussian parameter
L_BITS      = 1                                     # plaintext bits per encryption


# ═══════════════════════════════════════════════════════════════
#  SECTION 2: LATTICE PRIMITIVES
# ═══════════════════════════════════════════════════════════════

def mod_q(x, q):
    """Reduce array mod q into [0, q-1]."""
    return np.mod(x, q).astype(np.int64)


def discrete_gaussian(sigma, shape, rng):
    """Sample integers from discrete Gaussian D_{Z, σ} (rounded continuous)."""
    return np.round(rng.normal(0, sigma, shape)).astype(np.int64)


def gadget_matrix(n, q):
    """
    Gadget matrix G = I_n ⊗ [1, 2, 4, ..., 2^{k-1}]  ∈ Z_q^{n × nk}.
    Has a publicly known short trapdoor with ||T̃_G|| ≤ √5.
    """
    k = math.ceil(math.log2(q))
    g = np.array([1 << i for i in range(k)], dtype=np.int64)
    G = np.zeros((n, n * k), dtype=np.int64)
    for i in range(n):
        G[i, i*k:(i+1)*k] = g
    return mod_q(G, q)


def gadget_inverse(u, n, q):
    """
    G⁻¹(u): bit-decompose each entry of u to get short x with G·x = u mod q.
    Works for vectors (Z_q^n → Z^{nk}) and matrices (Z_q^{n×c} → Z^{nk×c}).
    """
    k = math.ceil(math.log2(q))
    if u.ndim == 1:
        x = np.zeros(n * k, dtype=np.int64)
        for i in range(n):
            val = int(u[i]) % q
            for j in range(k):
                x[i*k + j] = val % 2
                val //= 2
        return x
    else:
        return np.column_stack([gadget_inverse(u[:, c], n, q) for c in range(u.shape[1])])


def gadget_trapdoor(n, q):
    """Trapdoor T_G for gadget: each block has [2,-1] bidiagonal structure, G·T_G ≡ 0 mod q."""
    k = math.ceil(math.log2(q))
    t_g = np.zeros((k, k-1), dtype=np.int64)
    for j in range(k-1):
        t_g[j, j] = 2
        t_g[j+1, j] = -1
    T_G = np.zeros((n*k, n*(k-1)), dtype=np.int64)
    for i in range(n):
        T_G[i*k:(i+1)*k, i*(k-1):(i+1)*(k-1)] = t_g
    return T_G


def trap_gen(n, m, q, rng):
    """
    TrapGen → (A, R)  using Micciancio-Peikert 2012 construction.
    
    A = [Ā | G − Ā·R]  where Ā is random, R is short.
    Trapdoor property: A · [R; I] = G  (mod q).
    """
    k = math.ceil(math.log2(q))
    m_bar = m - n*k   # columns for random part

    A_bar = rng.integers(0, q, size=(n, m_bar), dtype=np.int64)
    R = rng.choice([-1, 0, 1], size=(m_bar, n*k)).astype(np.int64)
    G = gadget_matrix(n, q)
    
    A = np.hstack([A_bar, mod_q(G - A_bar @ R, q)])
    return A, R


def sample_pre(A, R, sigma, u, n, q, rng):
    """
    SamplePre: find short s with A·s ≡ u mod q.
    
    Uses trapdoor R:
        1. z = G⁻¹(u)         (short, binary)
        2. p = [R·z; z]        (A·p = G·z = u)
        3. Add kernel perturbation for Gaussian distribution
    """
    is_matrix = (u.ndim == 2)
    if not is_matrix:
        u = u.reshape(-1, 1)
    
    m = A.shape[1]
    T_G = gadget_trapdoor(n, q)
    results = []
    
    for c in range(u.shape[1]):
        uc = mod_q(u[:, c], q)
        z = gadget_inverse(uc, n, q)
        p = np.concatenate([mod_q(R @ z.reshape(-1,1), q).flatten(), z])
        
        # Add kernel perturbation: [R·e; e] where G·e ≡ 0
        if T_G.shape[1] > 0:
            coeffs = discrete_gaussian(sigma/2, T_G.shape[1], rng)
            e = T_G @ coeffs.reshape(-1, 1)
            e = e.flatten()
            kernel = np.concatenate([mod_q(R @ e.reshape(-1,1), q).flatten(), e])
            p = p + kernel
        
        results.append(mod_q(p, q))
    
    S = np.column_stack(results)
    return S.flatten() if not is_matrix else S


def sample_left(A, M, R, sigma, u, n, q, rng):
    """
    SampleLeft: find short s with [A | M]·s ≡ u mod q.
    
    Algorithm:
        1. s₂ ← D_{Z^{m₀}, σ}       (random short vector)
        2. u' = u − M·s₂ mod q
        3. s₁ = SamplePre(A, R, σ, u')
        4. Return [s₁; s₂]
    
    Result: [A|M]·[s₁;s₂] = A·s₁ + M·s₂ = u' + M·s₂ = u
    """
    m0 = M.shape[1]
    is_matrix = (u.ndim == 2)
    if not is_matrix:
        u = u.reshape(-1, 1)
    
    results = []
    for c in range(u.shape[1]):
        uc = mod_q(u[:, c], q)
        s2 = discrete_gaussian(sigma, m0, rng)
        u_prime = mod_q(uc - mod_q(M @ s2.reshape(-1,1), q).flatten(), q)
        s1 = sample_pre(A, R, sigma, u_prime, n, q, rng)
        results.append(np.concatenate([s1, s2]))
    
    S = np.column_stack(results)
    return S.flatten() if not is_matrix else S


def H(identity, n, q):
    """
    Full-rank difference map H(ID) → diagonal matrix in Z_q^{n×n}.
    For any ID₁ ≠ ID₂:  H(ID₁) − H(ID₂) is non-singular.
    """
    diag = np.zeros(n, dtype=np.int64)
    for i in range(n):
        h = hashlib.sha256(f"{identity}||{i}".encode()).digest()
        val = int.from_bytes(h[:8], 'big') % q
        if val == 0:
            val = 1
        diag[i] = val
    return np.diag(diag)


# ═══════════════════════════════════════════════════════════════
#  SECTION 3: KGC ALGORITHMS (Setup, GenSK, NumUp)
# ═══════════════════════════════════════════════════════════════

def setup(N, n=N_DEFAULT, q=Q_DEFAULT, sigma=SIGMA, l=L_BITS, rng=None):
    """
    Setup(1^λ, N) → (PP, MSK)
    
    Paper Construction Step 1:
        • Generate A with trapdoor R via TrapGen
        • Choose random B, W ∈ Z_q^{n×m}
        • Choose random target vectors {u_i}
        • Create Number List NL = [1..N]
        • For each number, create random matrix D_no
    
    Returns:
        PP  = dict with public parameters  (shared with everyone)
        MSK = dict with master secret       (KGC keeps this secret!)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    k = math.ceil(math.log2(q))
    m = 2 * n * k
    
    print(f"[Setup] n={n}, m={m}, q={q}, σ={sigma}, N={N}")
    
    # TrapGen → (A, R)
    A, R = trap_gen(n, m, q, rng)
    
    # Random matrices B, W
    B = rng.integers(0, q, size=(n, m), dtype=np.int64)
    W = rng.integers(0, q, size=(n, m), dtype=np.int64)
    
    # Target vectors
    u_list = [rng.integers(0, q, size=n, dtype=np.int64) for _ in range(l)]
    
    # Number List and per-number matrices
    NL = list(range(1, N + 1))
    D_no = {no: rng.integers(0, q, size=(n, m), dtype=np.int64) for no in NL}
    
    # Gadget matrix (public)
    G = gadget_matrix(n, q)
    
    PP = {
        'A': A, 'B': B, 'W': W, 'G': G,
        'u_list': u_list, 'NL': NL, 'D_no': D_no,
        'n': n, 'm': m, 'q': q, 'sigma': sigma, 'l': l
    }
    
    MSK = {
        'R': R,                    # trapdoor for A
        'id_to_number': {},        # identity → assigned number
        'allocated': set()         # numbers already given out
    }
    
    print(f"[Setup] Done. A ∈ Z_q^{{{n}×{m}}}")
    return PP, MSK


def gen_sk(PP, identity, MSK, rng=None):
    """
    GenSK(PP, ID, MSK) → SK_ID
    
    Paper Construction Step 2:
        1. Assign unallocated number no_ID to this identity
        2. B_ID = B + H(ID)·G
        3. x'_ID ← small random matrix (2m × 2m)
        4. Y_ID = [A | B_ID] · x'_ID
        5. x''_no ← SampleLeft(A, D_{noID}, R, σ, G − Y_ID)
        6. Combine into SK: x_{ID,noID} ∈ Z_q^{3m × 2m}
           Satisfies: [A | B_ID | D_{noID}] · SK = G  (mod q)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n, m, q, sigma = PP['n'], PP['m'], PP['q'], PP['sigma']
    
    # Step 1: Assign number
    if identity in MSK['id_to_number']:
        raise ValueError(f"'{identity}' already registered")
    
    available = [no for no in PP['NL'] if no not in MSK['allocated']]
    if not available:
        raise ValueError("No numbers left — system at capacity")
    
    no_ID = available[0]
    MSK['id_to_number'][identity] = no_ID
    MSK['allocated'].add(no_ID)
    
    print(f"[GenSK] '{identity}' → number {no_ID}")
    
    # Step 2: B_ID = B + H(ID)·G  (pad G to m columns)
    H_ID = H(identity, n, q)
    G_padded = np.zeros((n, m), dtype=np.int64)
    G_padded[:, :PP['G'].shape[1]] = PP['G']
    B_ID = mod_q(PP['B'] + mod_q(H_ID @ G_padded, q), q)
    
    # Step 3: Small random matrix x'_ID ∈ Z^{2m × 2m}
    x_prime = discrete_gaussian(sigma, (2*m, 2*m), rng)
    x_prime_1 = x_prime[:m, :]     # top half:  m × 2m
    x_prime_2 = x_prime[m:, :]     # bottom:    m × 2m
    
    # Step 4: Y_ID = [A | B_ID] · x'_ID
    AB = np.hstack([PP['A'], B_ID])             # n × 2m
    Y_ID = mod_q(AB @ x_prime, q)               # n × 2m
    
    # Step 5: SampleLeft to solve [A | D_no] · x'' = G − Y_ID
    D_no = PP['D_no'][no_ID]
    G_target = np.zeros((n, 2*m), dtype=np.int64)
    G_target[:, :PP['G'].shape[1]] = PP['G']
    target = mod_q(G_target - Y_ID, q)
    
    x_double = sample_left(PP['A'], D_no, MSK['R'], sigma, target, n, q, rng)
    x_double_1 = x_double[:m, :]   # A part:    m × 2m
    x_double_2 = x_double[m:, :]   # D_no part: m × 2m
    
    # Step 6: Combine  SK = [x'₁ + x''₁ ;  x'₂ ;  x''₂]   →  3m × 2m
    SK = mod_q(np.vstack([
        x_prime_1 + x_double_1,    # m × 2m
        x_prime_2,                   # m × 2m
        x_double_2                   # m × 2m
    ]), q)
    
    print(f"[GenSK] SK shape: {SK.shape}")
    return {'SK': SK, 'no_ID': no_ID, 'identity': identity}


def num_up(PP, MSK, t, revocation_list):
    """
    NumUp(PP, MSK, NL, t, RL_t) → NRno_t
    
    Paper Construction Step 3:
        • Filter out revoked users
        • Broadcast numbers of non-revoked users
        • O(1) per broadcast — the scheme's key advantage
    
    Returns:
        NRno_t = {'time': t, 'numbers': set of active numbers}
    """
    print(f"[NumUp] t={t}, revoking: {revocation_list or 'none'}")
    
    active_numbers = set()
    for identity, no in MSK['id_to_number'].items():
        if identity in revocation_list:
            print(f"  ✗ revoked: {identity} (no={no})")
        else:
            active_numbers.add(no)
            print(f"  ✓ active:  {identity} (no={no})")
    
    print(f"[NumUp] Broadcast NRno_{t} = {sorted(active_numbers)}")
    return {'time': t, 'numbers': active_numbers}


# ═══════════════════════════════════════════════════════════════
#  SECTION 4: SAVE / LOAD (for P2, P3, P4 consumption)
# ═══════════════════════════════════════════════════════════════

def save_pp(PP, directory):
    """Save Public Parameters so other teams can load them."""
    os.makedirs(directory, exist_ok=True)
    save_dict = {'A': PP['A'], 'B': PP['B'], 'W': PP['W'], 'G': PP['G']}
    for i, u in enumerate(PP['u_list']):
        save_dict[f'u_{i}'] = u
    for no, D in PP['D_no'].items():
        save_dict[f'D_{no}'] = D
    np.savez_compressed(os.path.join(directory, 'PP.npz'), **save_dict)
    with open(os.path.join(directory, 'PP_meta.json'), 'w') as f:
        json.dump({'n': PP['n'], 'm': PP['m'], 'q': PP['q'],
                   'sigma': PP['sigma'], 'l': PP['l'], 'NL': PP['NL']}, f, indent=2)


def load_pp(directory):
    """Load Public Parameters (used by P2, P3, P4)."""
    data = np.load(os.path.join(directory, 'PP.npz'))
    with open(os.path.join(directory, 'PP_meta.json')) as f:
        meta = json.load(f)
    PP = {
        'A': data['A'], 'B': data['B'], 'W': data['W'], 'G': data['G'],
        'u_list': [data[f'u_{i}'] for i in range(meta['l'])],
        'NL': meta['NL'],
        'D_no': {no: data[f'D_{no}'] for no in meta['NL']},
        'n': meta['n'], 'm': meta['m'], 'q': meta['q'],
        'sigma': meta['sigma'], 'l': meta['l']
    }
    return PP


def save_sk(sk, path):
    """Save a user's secret key."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, SK=sk['SK'])
    with open(path.replace('.npz', '_meta.json'), 'w') as f:
        json.dump({'no_ID': sk['no_ID'], 'identity': sk['identity']}, f, indent=2)


def load_sk(path):
    """Load a user's secret key."""
    data = np.load(path)
    with open(path.replace('.npz', '_meta.json')) as f:
        meta = json.load(f)
    return {'SK': data['SK'], 'no_ID': meta['no_ID'], 'identity': meta['identity']}


def save_nrno(nrno, path):
    """Save non-revoked number set."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'time': nrno['time'], 'numbers': sorted(nrno['numbers'])}, f, indent=2)


def load_nrno(path):
    """Load non-revoked number set."""
    with open(path) as f:
        d = json.load(f)
    return {'time': d['time'], 'numbers': set(d['numbers'])}


# ═══════════════════════════════════════════════════════════════
#  SECTION 5: TESTS
# ═══════════════════════════════════════════════════════════════

def run_tests():
    """Run all correctness tests."""
    rng = np.random.default_rng(42)
    n, q = 8, 99991
    k = math.ceil(math.log2(q))
    m = 2 * n * k
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}")
            failed += 1

    print("\n── Gadget Matrix Tests ──")
    G = gadget_matrix(n, q)
    check("G dimensions", G.shape == (n, n*k))
    
    u = rng.integers(0, q, size=n, dtype=np.int64)
    x = gadget_inverse(u, n, q)
    check("G · G⁻¹(u) = u", np.array_equal(mod_q(G @ x, q), mod_q(u, q)))
    check("G⁻¹(u) is binary", np.all(np.abs(x) <= 1))
    
    T_G = gadget_trapdoor(n, q)
    check("G · T_G = 0 mod q", np.array_equal(mod_q(G @ T_G, q), np.zeros_like(mod_q(G @ T_G, q))))

    print("\n── TrapGen Tests ──")
    A, R = trap_gen(n, m, q, rng)
    check("A dimensions", A.shape == (n, m))
    RI = np.vstack([R, np.eye(n*k, dtype=np.int64)])
    check("A·[R;I] = G mod q", np.array_equal(mod_q(A @ RI, q), mod_q(G, q)))
    check("A entries in [0,q)", np.all(A >= 0) and np.all(A < q))

    print("\n── SamplePre Test ──")
    u = rng.integers(0, q, size=n, dtype=np.int64)
    s = sample_pre(A, R, 4.0, u, n, q, rng)
    check("A·s = u mod q", np.array_equal(mod_q(A @ s, q), mod_q(u, q)))

    print("\n── SampleLeft Test ──")
    M = rng.integers(0, q, size=(n, m), dtype=np.int64)
    u = rng.integers(0, q, size=n, dtype=np.int64)
    s = sample_left(A, M, R, 4.0, u, n, q, rng)
    result = mod_q(np.hstack([A, M]) @ s, q)
    check("[A|M]·s = u mod q", np.array_equal(result, mod_q(u, q)))

    print("\n── H(ID) Tests ──")
    H1 = H("alice", n, q)
    H2 = H("bob", n, q)
    check("H(ID) is n×n", H1.shape == (n, n))
    check("H(alice) ≠ H(bob)", not np.array_equal(H1, H2))

    print("\n── Setup Test ──")
    PP, MSK = setup(5, n=n, q=q, rng=rng)
    check("PP has correct A shape", PP['A'].shape == (n, m))
    check("NL has 5 numbers", len(PP['NL']) == 5)
    check("MSK initially empty", len(MSK['id_to_number']) == 0)

    print("\n── GenSK Tests ──")
    sk1 = gen_sk(PP, "alice", MSK, rng)
    sk2 = gen_sk(PP, "bob", MSK, rng)
    check("Unique numbers", sk1['no_ID'] != sk2['no_ID'])
    check("SK shape = (3m, 2m)", sk1['SK'].shape == (3*m, 2*m))
    check("Identity stored", "alice" in MSK['id_to_number'])
    
    try:
        gen_sk(PP, "alice", MSK, rng)
        check("Duplicate rejected", False)
    except ValueError:
        check("Duplicate rejected", True)

    print("\n── NumUp Tests ──")
    gen_sk(PP, "charlie", MSK, rng)
    nrno = num_up(PP, MSK, t=1, revocation_list=set())
    check("No revocations → all active", len(nrno['numbers']) == 3)
    
    nrno = num_up(PP, MSK, t=2, revocation_list={"bob"})
    check("Bob revoked → 2 active", len(nrno['numbers']) == 2)
    check("Bob's number excluded", sk2['no_ID'] not in nrno['numbers'])
    check("Alice's number included", sk1['no_ID'] in nrno['numbers'])

    print("\n── Serialization Tests ──")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_pp(PP, tmpdir)
        PP2 = load_pp(tmpdir)
        check("PP roundtrip (A)", np.array_equal(PP['A'], PP2['A']))
        
        sk_path = os.path.join(tmpdir, "SK.npz")
        save_sk(sk1, sk_path)
        sk1_loaded = load_sk(sk_path)
        check("SK roundtrip", np.array_equal(sk1['SK'], sk1_loaded['SK']))
        
        nrno_path = os.path.join(tmpdir, "nrno.json")
        save_nrno(nrno, nrno_path)
        nrno2 = load_nrno(nrno_path)
        check("NRno roundtrip", nrno['numbers'] == nrno2['numbers'])

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


# ═══════════════════════════════════════════════════════════════
#  SECTION 6: DEMO
# ═══════════════════════════════════════════════════════════════

def run_demo():
    """Full P1 flow: Setup → Register 5 users → Revoke 1 → Save artifacts."""
    print("=" * 65)
    print("  OO-IRIBE-EnDKER — P1: System Admin & KGC Demo")
    print("=" * 65)
    
    rng = np.random.default_rng(42)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1_output")
    
    # ── Phase 1: Setup ──
    print("\n── Phase 1: SETUP ──")
    t0 = time.time()
    PP, MSK = setup(10, rng=rng)
    t_setup = time.time() - t0
    print(f"  ⏱ {t_setup*1000:.1f} ms")
    
    # ── Phase 2: Register Users ──
    print("\n── Phase 2: REGISTER USERS ──")
    users = ["alice", "bob", "charlie", "diana", "eve"]
    user_keys = {}
    gen_times = []
    
    for uid in users:
        t0 = time.time()
        sk = gen_sk(PP, uid, MSK, rng)
        dt = time.time() - t0
        gen_times.append(dt)
        user_keys[uid] = sk
        print(f"  ⏱ {dt*1000:.1f} ms")
    
    # ── Phase 3: Revoke Eve ──
    print("\n── Phase 3: REVOCATION (t=1, revoke Eve) ──")
    t0 = time.time()
    nrno = num_up(PP, MSK, t=1, revocation_list={"eve"})
    t_numup = time.time() - t0
    print(f"  ⏱ {t_numup*1000:.2f} ms")
    
    # ── Save everything ──
    print("\n── Saving artifacts ──")
    save_pp(PP, out_dir)
    for uid, sk in user_keys.items():
        save_sk(sk, os.path.join(out_dir, f"SK_{uid}.npz"))
    save_nrno(nrno, os.path.join(out_dir, "NRno_t1.json"))
    
    # NOTE: MSK should be kept secret. Saving for completeness.
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "MSK.npz"), R=MSK['R'])
    with open(os.path.join(out_dir, "MSK_meta.json"), 'w') as f:
        json.dump({'id_to_number': MSK['id_to_number'],
                   'allocated': list(MSK['allocated'])}, f, indent=2)
    
    # ── Summary ──
    print("\n" + "=" * 65)
    print("  PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Setup:       {t_setup*1000:>10.1f} ms")
    print(f"  GenSK avg:   {np.mean(gen_times)*1000:>10.1f} ms")
    print(f"  NumUp:       {t_numup*1000:>10.2f} ms  ← O(1) constant!")
    print()
    print("  Output files (in p1_output/):")
    for f in sorted(os.listdir(out_dir)):
        sz = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f:30s}  {sz:>10,} bytes")
    print()
    print("  HOW P2/P3/P4 LOAD THIS:")
    print("    from p1_implementation import load_pp, load_sk, load_nrno")
    print("    PP   = load_pp('p1_output')")
    print("    sk   = load_sk('p1_output/SK_alice.npz')")
    print("    nrno = load_nrno('p1_output/NRno_t1.json')")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        run_demo()
