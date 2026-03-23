"""
╔══════════════════════════════════════════════════════════════════════╗
║  P3: Encryption Engine (Online/Offline) — OO-IRIBE-EnDKER            ║
║                                                                      ║
║  Responsibility: Implement split encryption to optimize performance. ║
║  Context: Generates Intermediate Ciphertext (IT) offline, and        ║
║           Final Ciphertext (CT) online when ID and message arrive.   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
import os
import sys

# Import primitives from P1
try:
    from p1_implementation import mod_q, discrete_gaussian, load_pp, load_nrno, H
except ImportError:
    print("Error: p1_implementation.py not found. Please ensure P1's code is in this directory.")
    sys.exit(1)


class P3EncryptionEngine:
    def __init__(self, pp_directory="p1_output"):
        """Initialize P3 by loading the Public Parameters."""
        if not os.path.exists(pp_directory):
            raise FileNotFoundError(f"Directory {pp_directory} not found. Run P1 Setup first!")
            
        self.pp = load_pp(pp_directory)
        self.n = self.pp['n']
        self.m = self.pp['m']
        self.q = self.pp['q']
        self.sigma = self.pp['sigma']
        self.rng = np.random.default_rng()
        
        # Pre-pad the Gadget matrix G to match m columns for H(ID)*G operations
        # G is naturally n x nk, but B and W are n x m. We pad G with zeros to match m.
        self.G_padded = np.zeros((self.n, self.m), dtype=np.int64)
        self.G_padded[:, :self.pp['G'].shape[1]] = self.pp['G']
        
        print(f"[P3] Initialized Encryption Engine. n={self.n}, m={self.m}, q={self.q}")

    def offline_enc(self, t, nrno_t):
        """
        Offline.Enc(PP, t, NRno_t) -> IT
        Pre-computes ciphertext components that do not depend on the recipient ID or the message.
        """
        t_start = time.time()
        
        # 1. Select random LWE secret s ∈ Z_q^n
        s = self.rng.integers(0, self.q, size=self.n, dtype=np.int64)
        
        # 2. Compute base component c0 = s^T A + e_0
        e_0 = discrete_gaussian(self.sigma, self.m, self.rng)
        c0 = mod_q(s @ self.pp['A'] + e_0, self.q)
        
        # 3. Compute c'_{no} = s^T D_{no} + e'_{no} for all valid numbers in NRno_t
        c_prime_no = {}
        for no in nrno_t['numbers']:
            if no in self.pp['D_no']:
                e_no = discrete_gaussian(self.sigma, self.m, self.rng)
                c_prime_no[no] = mod_q(s @ self.pp['D_no'][no] + e_no, self.q)
        
        # 4. Compute c_t'' = s^T W_t + e''_t
        # W_t is dynamically generated for time t: W_t = W + H(t)G
        H_t = H(str(t), self.n, self.q)
        W_t = mod_q(self.pp['W'] + mod_q(H_t @ self.G_padded, self.q), self.q)
        
        e_t = discrete_gaussian(self.sigma, self.m, self.rng)
        ct_double_prime = mod_q(s @ W_t + e_t, self.q)
        
        t_end = time.time()
        print(f"[P3] Offline Phase complete in {(t_end - t_start)*1000:.2f} ms")
        
        # Return Intermediate Ciphertext (IT). The secret 's' is kept by the encryptor.
        return {
            's': s,
            't': t,
            'c0': c0,
            'c_prime_no': c_prime_no,
            'ct_double_prime': ct_double_prime
        }

    def online_enc(self, target_id, IT, message_bit, u_index=0):
        """
        Online.Enc(PP, ID, IT, Message) -> CT_{ID, t}
        Rapidly binds the recipient ID and encrypts the message bit.
        """
        t_start = time.time()
        
        s = IT['s']
        
        # 1. Compute c_ID = s^T B_ID + e_ID
        # B_ID = B + H(ID)G
        H_ID = H(target_id, self.n, self.q)
        B_ID = mod_q(self.pp['B'] + mod_q(H_ID @ self.G_padded, self.q), self.q)
        
        e_ID = discrete_gaussian(self.sigma, self.m, self.rng)
        c_ID = mod_q(s @ B_ID + e_ID, self.q)
        
        # 2. Compute final component using the message: C_i = s^T u_i + e_i + \mu * ⌊q/2⌋
        u_i = self.pp['u_list'][u_index]
        e_i = int(discrete_gaussian(self.sigma, 1, self.rng)[0])
        
        # Hide the message bit in the most significant bit
        mu_term = message_bit * (self.q // 2)
        C_i = mod_q(int(np.dot(s, u_i)) + e_i + mu_term, self.q)
        
        t_end = time.time()
        print(f"[P3] Online Phase (Target: '{target_id}', Msg: {message_bit}) complete in {(t_end - t_start)*1000:.2f} ms")
        
        # 3. Assemble and return Final Ciphertext
        return {
            'C_i': C_i,
            'c0': IT['c0'],
            'c_ID': c_ID,
            'c_prime_no': IT['c_prime_no'],
            'ct_double_prime': IT['ct_double_prime'],
            'target_id': target_id,
            'time_t': IT['t']
        }

# ═══════════════════════════════════════════════════════════════
#  DEMO / TEST SUITE
# ═══════════════════════════════════════════════════════════════

def run_p3_demo():
    print("=" * 65)
    print("  OO-IRIBE-EnDKER — P3: Encryption Engine Demo")
    print("=" * 65)

    try:
        # Load active numbers from P1 output
        nrno_t = load_nrno("p1_output/NRno_t1.json")
        t = nrno_t['time']
        
        # Initialize Engine
        encryptor = P3EncryptionEngine("p1_output")
        
        print("\n── 1. Executing Offline Phase ──")
        IT = encryptor.offline_enc(t, nrno_t)
        
        print("\n── 2. Executing Online Phase ──")
        target_identity = "alice"
        message_bit = 1 
        
        CT = encryptor.online_enc(target_identity, IT, message_bit)
        
        print("\n── Ciphertext (CT) Artifacts Generated ──")
        print(f"  Target ID: {CT['target_id']}")
        print(f"  Time (t):  {CT['time_t']}")
        print(f"  C_i:       {CT['C_i']} (Scalar)")
        print(f"  c0:        Shape {CT['c0'].shape}")
        print(f"  c_ID:      Shape {CT['c_ID'].shape}")
        print(f"  c_t'':     Shape {CT['ct_double_prime'].shape}")
        print(f"  c'_no:     {len(CT['c_prime_no'])} active numbers included")

        # Save CT so P4 can load it for decryption
        out_path = "p1_output/CT_demo.npz"
        np.savez_compressed(
            out_path, 
            C_i=CT['C_i'], 
            c0=CT['c0'], 
            c_ID=CT['c_ID'], 
            ct_double_prime=CT['ct_double_prime']
        )
        
        # Save the dictionary of c'_no arrays separately to avoid ragged array issues
        c_prime_path = "p1_output/CT_c_prime_no.npz"
        np.savez_compressed(c_prime_path, **{str(k): v for k, v in CT['c_prime_no'].items()})
        print(f"\n  Ciphertext saved to '{out_path}' for P4.")

    except FileNotFoundError as e:
        print(f"    Error: {e}")
        print("     Please run the P1 script first to generate 'p1_output' data.")

if __name__ == "__main__":
    run_p3_demo()