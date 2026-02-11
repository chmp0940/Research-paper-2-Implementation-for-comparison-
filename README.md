# 📘 Developer Handbook: OO-IRIBE-EnDKER Implementation

## 1. Project Overview
**Objective:** Implement a lattice-based Identity-Based Encryption (IBE) system that supports efficient user revocation and resists key exposure.

**Core Innovation:**
1.  **Integrated Revocation Model:** Replaces complex binary trees with a simple "Number List" ($NL$) to keep KGC workload constant ($O(1)$).
2.  **Online/Offline Encryption:** Splits encryption into a heavy "Offline" phase (pre-computation) and a lightweight "Online" phase to reduce computational cost.
3.  **Cloud-Assisted Decryption:** Uses a semi-trusted cloud server to handle heavy lattice sampling during key generation, reducing user burden.

---

## 2. System Architecture & Data Dictionary
Ensure all team members use these standard data structures.

* **`PP` (Public Parameters):** Contains matrices $A, B, W$, the list of numbers $NL$, and vectors $u_i$.
* **`MSK` (Master Secret Key):** Contains the trapdoor $T_A$ and the full Number List mapping.
* **`NL` (Number List):** A list containing $N$ integers. Each user is assigned one number from this list.
* **`NRno_t`:** The set of numbers corresponding to **non-revoked** users at time $t$.
* **`IT` (Intermediate Ciphertext):** The pre-computed values from the Offline phase.

---

## 3. Team Roles & Task Specifications

### 🧑‍💻 P1: System Admin & KGC (Root Layer)
**Responsibility:** Manage the system setup, user keys, and daily revocation broadcasting using the Number List.

**1. `Setup(N)`**
* **Input:** Security parameter $\lambda$, Number of users $N$.
* **Logic:**
    * Generate LWE matrix $A$ and trapdoor $T_A$ using `TrapGen`.
    * Create a Number List `NL` with at least $N$ numbers.
    * Select random matrices $B, W$ and vectors $u_i$.
* **Output:** `PP` (Public Params), `MSK` (Master Key).

**2. `GenSK(PP, ID, MSK)`**
* **Input:** User identity `ID`.
* **Logic:**
    * Select an unallocated number $no_{ID}$ from `NL` and assign it to `ID`.
    * **Crucial:** Only the KGC knows this mapping.
    * Compute secret key matrix $SK_{ID}$ using `SampleLeft` algorithm.
* **Output:** User Secret Key $SK_{ID}$.

**3. `NumUp(PP, MSK, NL, t, RL)`**
* **Input:** Time $t$, Revocation List `RL` (banned users).
* **Logic:**
    * Identify all users *not* in `RL`.
    * Collect their assigned numbers into a set `NRno_t`.
    * Broadcast `NRno_t`.
* **Output:** `NRno_t`.

---

### 🧑‍💻 P2: Cloud Server & Key Delegation (Assistant Layer)
**Responsibility:** Help users generate decryption keys ($GenDK$) without seeing their secrets.

**1. `GenDK_Cloud_Side(h_vector)`**
* **Context:** The user (P4) sends a blinded vector $h$ to you.
* **Logic:**
    * Run `SamplePre` algorithm to find a short vector $x'$ such that $G x' = u - h$.
    * Send $x'$ back to the user.
* **Note:** You act as a semi-honest server (follow protocol but try to learn info).

---

### 🧑‍💻 P3: Encryption Engine (Online/Offline)
**Responsibility:** Implement the split encryption to optimize performance.

**1. `Offline.Enc(PP, t, NRno_t)`**
* **Input:** Time $t$, Non-revoked number set `NRno_t`.
* **Logic:**
    * Select random LWE secret $s$ and error terms $e, e'$.
    * Compute base ciphertext components **before** knowing the message:
        * $c_0 = s^T A + e'^T$
        * $c'_{no} = s^T D_{no} + e'^T R_{no}$ (for all valid numbers in `NRno_t`)
        * $c''_t = s^T W_t + e'^T S$
* **Output:** Intermediate Ciphertext `IT`.

**2. `Online.Enc(PP, ID, IT, Message)`**
* **Input:** Target `ID`, `IT`, Message $\mu$ (bit).
* **Logic:**
    * Compute the final component using the message:
        * $C_i = s^T u_i + \mu \cdot \lfloor q/2 \rfloor + e_i$.
    * Assemble final ciphertext: $CT_{ID,t} = \{C_i, c_0, c_{ID}, c'_{no}, c''_t\}$.
* **Output:** Final Ciphertext $CT_{ID,t}$.

---

### 🧑‍💻 P4: User & Integrator (Analyst Layer)
**Responsibility:** Coordinate the decryption process and benchmark the system.

**1. `GenDK_User_Side(SK, PP, t)`**
* **Input:** Secret Key $SK_{ID}$, Time $t$.
* **Logic:**
    * Compute vector $h$ and send to P2 (Cloud).
    * Receive $x'$ from Cloud.
    * Combine $x'$ with your $SK$ to form the final Decryption Key $DK_{ID,t}$.
* **Output:** $DK_{ID,t}$.

**2. `Dec(CT, DK)`**
* **Input:** Ciphertext $CT$, Decryption Key $DK$.
* **Logic:**
    * Compute $C'_i = C_i - [c_0 | c_{ID} | c'_{no} | c''_t] \cdot dk$.
    * **Threshold Check:**
        * If $|C'_i - \lfloor q/2 \rfloor| < \lfloor q/4 \rfloor$, output **1**.
        * Otherwise, output **0**.

---

## 4. Implementation Logic Flow
1.  **Start:** P1 runs `Setup` $\rightarrow$ Outputs `PP` (includes `NL`).
2.  **Register:** P1 runs `GenSK` for User Alice $\rightarrow$ Alice gets `SK`.
3.  **Update:** P1 runs `NumUp` for Time $t=1$ $\rightarrow$ Broadcasts `NRno_1`.
4.  **Pre-Encrypt:** P3 runs `Offline.Enc` using `NRno_1` $\rightarrow$ Saves `IT`.
5.  **Encrypt:** P3 runs `Online.Enc` for Alice using `IT` $\rightarrow$ Sends `CT`.
6.  **Key Request:** Alice (P4) calculates $h$, sends to Cloud (P2).
7.  **Cloud Reply:** Cloud (P2) computes $x'$, sends back to Alice.
8.  **Decrypt:** Alice (P4) constructs `DK` and runs `Dec` on `CT`.

## 5. Performance Metrics (For Report)
Measure these to prove the paper's claims:
* **KGC Workload:** Should be constant ($O(1)$), not growing with users $N$.
* **Key Size:** User key size should remain constant.
* **Online Encryption Speed:** Should be negligible (<1ms) compared to Offline phase.
* **Comparisons:** Compare your `GenSK` time against binary tree schemes (it should be faster/constant).
