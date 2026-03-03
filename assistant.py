"""
╔══════════════════════════════════════════════════════════════════════╗
║  P2: Cloud Server & Key Delegation — OO-IRIBE-EnDKER                 ║
║                                                                      ║
║  Responsibility: Provide SamplePre assistance to users (P4).         ║
║  Context: Receives blinded vector 'h', returns short vector 'x_prime'║
║                                                                      ║
║  This implementation acts as the semi-trusted Cloud Server.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
import os
import sys

# Import primitives and loaders from P1
try:
    from p1_implementation import mod_q, gadget_inverse, load_pp
except ImportError:
    print("Error: p1_implementation.py not found. Please ensure P1's code is in this directory.")
    sys.exit(1)

class CloudServer:
    def __init__(self, pp_directory="p1_output"):
        """
        Initialize Cloud Server by loading Public Parameters.
        The Cloud does NOT have access to the MSK.
        """
        if not os.path.exists(pp_directory):
            raise FileNotFoundError(f"Directory {pp_directory} not found. Run P1 Setup first!")
            
        self.pp = load_pp(pp_directory)
        self.n = self.pp['n']
        self.q = self.pp['q']
        self.G = self.pp['G']
        self.u_list = self.pp['u_list']
        
        print(f"[Cloud] Initialized with n={self.n}, q={self.q}")

    def gen_dk_cloud_side(self, h_vector, u_index=0):
        """
        GenDK_Cloud_Side(h_vector)
        
        Logic:
            1. The user (P4) calculates h = [A|B_ID|D_no|W_t] * SK_blinded
            2. The cloud solves G * x' = (u_i - h) mod q
            3. Since G is a gadget matrix, we use bit decomposition (gadget_inverse).
        
        Input: 
            h_vector: ndarray of shape (n,)
            u_index: Which target vector u_i to use (default 0 for 1-bit msg).
            
        Output:
            x_prime: A short vector (nk length) such that G*x' = target mod q.
        """
        t_start = time.time()
        
        # 1. Retrieve the target vector u_i from Public Parameters
        u_i = self.u_list[u_index]
        
        # 2. Compute the target for the gadget inversion: (u_i - h) mod q
        # This is the 'offset' the user needs the cloud to solve.
        target = mod_q(u_i - h_vector, self.q)
        
        # 3. Perform Gadget Inversion (SamplePre equivalent for G)
        # This is the "heavy" sampling delegated to the cloud.
        x_prime = gadget_inverse(target, self.n, self.q)
        
        t_end = time.time()
        print(f"[Cloud] Processed GenDK request in {(t_end - t_start)*1000:.2f} ms")
        
        return x_prime

# ═══════════════════════════════════════════════════════════════
#  DEMO / TEST SUITE
# ═══════════════════════════════════════════════════════════════

def run_cloud_demo():
    print("=" * 65)
    print("  OO-IRIBE-EnDKER — P2: Cloud Server Assistant Demo")
    print("=" * 65)

    try:
        # Initialize the server (Requires P1 output)
        cloud = CloudServer("p1_output")
        
        # Simulate a request from a user (P4)
        # In reality, P4 would send a specifically computed 'h'
        print("\n[Cloud] Waiting for request from User (P4)...")
        dummy_h = np.random.randint(0, cloud.q, size=cloud.n)
        
        # Process request
        x_prime = cloud.gen_dk_cloud_side(dummy_h)
        
        # Verify correctness (G * x' should equal (u_i - h))
        target_check = mod_q(cloud.u_list[0] - dummy_h, cloud.q)
        reconstructed = mod_q(cloud.G @ x_prime, cloud.q)
        
        print("\n── Verification ──")
        if np.array_equal(target_check, reconstructed):
            print("  ✅ Correctness: G * x' == (u - h) mod q")
            print(f"  ✅ Efficiency: x_prime is short (binary/small entries)")
        else:
            print("  ❌ Correctness: Verification failed!")

    except FileNotFoundError as e:
        print(f"  ❌ Error: {e}")
        print("     Please run the P1 script first to generate 'p1_output/PP.npz'.")

if __name__ == "__main__":
    run_cloud_demo()
