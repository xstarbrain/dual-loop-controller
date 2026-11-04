"""
Generator for Frontier Advanced Curriculum & Multi-Round 50-Question Examination Pools.
Spanning 5 deep domains exclusive to frontier large models:
1. Distributed Consensus & Formal Invariants
2. Cryptographic Zero-Knowledge & Polynomial Commitments
3. Quantum Information & Topological Stabilizers
4. Continuous Symplectic Neural Dynamics & Adjoint ODEs
5. Higher Categorical Semantics & Monadic Proofs
"""

import json
from pathlib import Path

STUDY_DIR = Path("frontier_study")
STUDY_DIR.mkdir(parents=True, exist_ok=True)

def build_training_curriculum():
    data = [
        # Domain 1: Distributed Consensus & Invariants
        {
            "instruction": "Prove the State Machine Safety property of the Raft consensus algorithm. Explain why a leader can never overwrite or commit an entry at an index where another leader already committed a different entry.",
            "response": (
                "To prove Raft State Machine Safety, we establish the following chain of invariants:\n\n"
                "1. Leader Completeness Invariant: If a log entry is committed in a given term T, then that entry will be present in the logs of the leaders for all higher terms T' > T.\n"
                "Proof: A log entry is committed when it is replicated on a strict majority of nodes (quorum Q1). For any candidate to be elected leader in a higher term T', it must receive votes from a strict majority of nodes (quorum Q2). By the pigeonhole principle, any two majorities intersect in at least one node: Q1 ∩ Q2 ≠ ∅. Let node v ∈ Q1 ∩ Q2. Node v accepted the committed entry in term T. Under Raft's RequestVote RPC restriction, node v only votes for a candidate whose log is at least as up-to-date as its own. Since node v contains the committed entry, any candidate elected in term T' must have a log at least as up-to-date, guaranteeing it already possesses the committed entry.\n\n"
                "2. Log Matching Property: If two logs contain an entry with the same index and term, they are identical in all entries up through that index.\n\n"
                "3. Deduction of State Machine Safety: Because the leader never deletes or overwrites its own log entries (Leader Append-Only), and because every future leader inherits all committed entries by Leader Completeness, no two state machines can ever apply different entries at the same index. Thus, State Machine Safety is preserved."
            )
        },
        {
            "instruction": "Explain how Paxos resolves the split-brain hazard and why ballot number monotonicity is essential to guaranteeing safety in the Synod protocol.",
            "response": (
                "In the Paxos Synod protocol, split-brain hazard (where two distinct proposers believe their distinct values are chosen) is resolved using two-phase quorums and monotonic ballot numbers:\n\n"
                "1. Monotonic Ballot Numbers: Each proposal is assigned a unique, monotonically increasing ballot number n ∈ (N, ProposerID). Acceptors maintain two states: max_promised_ballot (the highest ballot they promised to never accept proposals below) and (max_accepted_ballot, accepted_value).\n\n"
                "2. Phase 1 (Prepare/Promise): A proposer broadcasts Prepare(n). If an acceptor receives n > max_promised_ballot, it replies Promise(max_accepted_ballot, accepted_value) and guarantees it will reject any future proposals with ballot < n.\n\n"
                "3. Quorum Intersection: If proposer P1 achieves a quorum Q1 of promises, it must select the value associated with the highest ballot number among all received promises (or choose its own if all are null). If a value v was already decided by an earlier ballot, at least one acceptor in Q1 accepted v and reports it. Thus, P1 is strictly forced to propose v, preserving consistency.\n\n"
                "4. Resolution of Split-Brain: If two proposers compete concurrently, the acceptor's promise ensures that any proposer with a lower ballot number is rejected in Phase 2 (Accept). Only a proposal supported by a strict majority that adheres to the latest promise can succeed, making simultaneous conflicting commitments impossible."
            )
        },
        {
            "instruction": "Formulate the mathematical definition of a Conflict-Free Replicated Data Type (CRDT) under the state-based (CvRDT) model. What algebraic structure guarantees Strong Eventual Consistency?",
            "response": (
                "A state-based Conflict-Free Replicated Data Type (CvRDT) is mathematically defined as a bounded join-semilattice (S, ⊔, ≤):\n\n"
                "1. Semi-Lattice Axioms: S is a partially ordered set under relation ≤ with a binary join operator ⊔: S × S → S that satisfies three algebraic properties for all x, y, z ∈ S:\n"
                "   - Commutativity: x ⊔ y = y ⊔ x (message delivery order does not matter).\n"
                "   - Associativity: (x ⊔ y) ⊔ z = x ⊔ (y ⊔ z) (message batching or grouping does not matter).\n"
                "   - Idempotence: x ⊔ x = x (duplicate message delivery has no effect).\n\n"
                "2. Monotonic Growth: State transitions across replicas are monotonically increasing with respect to the partial order: if state transitions from s to s', then s ≤ s'.\n\n"
                "3. Strong Eventual Consistency (SEC) Theorem: When replicas exchange their states and apply the merge operator m(s_A, s_B) = s_A ⊔ s_B, any two replicas that have received the same set of updates will reach identical states regardless of transmission order, delays, or retries, without requiring coordination locks."
            )
        },
        {
            "instruction": "In Viewstamped Replication (VR Revisited), analyze what occurs during a View Change and how the new Primary reconstructs the authoritative log.",
            "response": (
                "In Viewstamped Replication, when the primary fails or is suspected, replicas trigger a View Change to maintain safety:\n\n"
                "1. Initiation: A backup replica transitions to VIEW_CHANGE state, increments its view number to v + 1, and broadcasts START_VIEW_CHANGE(v + 1, replica_id).\n\n"
                "2. Quorum of View Changes: When a replica receives ⌊f⌋ START_VIEW_CHANGE messages, it sends DO_VIEW_CHANGE(v + 1, log, op_number, commit_number, latest_normal_view) to the designated primary of view v + 1.\n\n"
                "3. Log Reconstruction by New Primary: The new primary waits for a quorum of f + 1 DO_VIEW_CHANGE messages. It then inspects the received logs and selects the authoritative log according to the following deterministic rule:\n"
                "   - It selects the log that has the highest latest_normal_view number.\n"
                "   - If there is a tie, it selects the log with the largest op_number.\n"
                "4. Authoritative Commitment: The new primary updates its local log to match the authoritative log, sets its op_number to the highest in that log, and broadcasts START_VIEW(v + 1, log, op_number, commit_number). All replicas adopt this log, ensuring no committed operations are ever lost."
            )
        },
        # Domain 2: Cryptographic Zero-Knowledge & Polynomial Commitments
        {
            "instruction": "Derive the reduction of an arithmetic computation into a Rank-1 Constraint System (R1CS) and explain why every constraint must satisfy the bilinear form <A, s> * <B, s> = <C, s>.",
            "response": (
                "An arithmetic circuit over a finite field F consists of addition and multiplication gates. The reduction to Rank-1 Constraint System (R1CS) proceeds as follows:\n\n"
                "1. Witness Vector: Construct a public/private assignment vector s = [1, x_1, x_2, ..., x_n]^T ∈ F^(n+1), where s_0 = 1 (constant wire) and remaining elements represent inputs, intermediate wire values, and outputs.\n\n"
                "2. Bilinear Constraint Form: Every non-linear operation in an arithmetic circuit is a multiplication gate: left_input * right_input = output. Addition gates are linear combinations and do not require distinct constraints; they are folded into linear dot products.\n\n"
                "3. Matrix Representation: For m multiplication gates, we construct three sparse matrices A, B, C ∈ F^(m × (n+1)). For the i-th constraint:\n"
                "   <A_i, s> = linear combination for left input\n"
                "   <B_i, s> = linear combination for right input\n"
                "   <C_i, s> = linear combination for expected output\n"
                "The constraint equation is strictly:\n"
                "   (A · s) ∘ (B · s) = C · s\n"
                "where ∘ denotes Hadamard (element-wise) product. Because rank is 1 per constraint (a single multiplication of two linear forms), any satisfiability witness s verifies the entire computation."
            )
        },
        {
            "instruction": "Explain the mechanics of the Kate-Zaverucha-Goldberg (KZG) polynomial commitment scheme. How does the verifier check that P(z) = y using a single pairing equation?",
            "response": (
                "The KZG polynomial commitment scheme operates over a bilinear pairing group e: G_1 × G_2 → G_T with trusted setup (powers of tau) {[τ^i]_1}_{i=0}^d and [τ]_2:\n\n"
                "1. Commitment: To commit to polynomial P(x) = ∑ c_i x^i, the prover computes commitment C = [P(τ)]_1 = ∑ c_i [τ^i]_1 ∈ G_1.\n\n"
                "2. Evaluation Proof: To prove that P(z) = y, observe that z is a root of P(x) - y. By polynomial division, P(x) - y = Q(x)(x - z) for quotient polynomial Q(x). The prover computes proof π = [Q(τ)]_1 = ∑ q_i [τ^i]_1.\n\n"
                "3. Verification Pairing Equation: The verifier checks whether:\n"
                "   e(C - [y]_1, [1]_2) = e(π, [τ]_2 - [z]_2)\n"
                "Proof of Correctness:\n"
                "   e(C - [y]_1, [1]_2) = e([P(τ) - y]_1, [1]_2) = e([Q(τ)(τ - z)]_1, [1]_2) = e([Q(τ)]_1, [(τ - z)]_2) = e(π, [τ]_2 - [z]_2).\n"
                "Due to the algebraic group model and q-BDH assumption, a malicious prover cannot forge π for y' ≠ P(z) except with negligible probability."
            )
        },
        {
            "instruction": "State the Schwartz-Zippel Lemma and explain its foundational role in Succinct Non-Interactive Arguments of Knowledge (SNARKs).",
            "response": (
                "1. Statement of Schwartz-Zippel Lemma:\n"
                "Let F be a finite field and let P ∈ F[x_1, ..., x_n] be a non-zero multivariate polynomial of total degree d ≥ 0. Let S ⊆ F be a finite subset. If r_1, ..., r_n are chosen uniformly at random from S, then:\n"
                "   Pr[P(r_1, ..., r_n) = 0] ≤ d / |S|\n\n"
                "2. Foundational Role in SNARKs:\n"
                "In SNARK protocols (e.g., PLONK, Groth16), the prover asserts that two high-degree polynomials are identical over the entire domain: A(x)B(x) - C(x) = H(x)Z_H(x). Verifying equality for all points would require O(N) evaluations, violating succinctness. By Schwartz-Zippel, checking equality at a single secret or random challenge point r chosen from a large field (where |F| ≈ 2^256 and d ≈ 2^20) ensures that if the polynomials are not identical, the probability of false acceptance is at most 2^20 / 2^256 ≈ 2^-236 (cryptographically negligible). This enables O(1) verification size and time."
            )
        },
        # Domain 3: Quantum Information & Topological Stabilizers
        {
            "instruction": "Explain the Toric Code as a topological quantum error-correcting code. How do star and plaquette operators define the code space, and why is the ground state 4-fold degenerate on a torus?",
            "response": (
                "The Toric Code (Kitaev, 1997) is a 2D lattice quantum error-correcting code on a torus T^2 with physical qubits on the edges:\n\n"
                "1. Hamiltonian and Stabilizer Generators:\n"
                "   H = - J_s ∑_s A_s - J_p ∑_p B_p\n"
                "   - Star operator A_s = ∏_{j ∈ star(s)} X_j (tensor product of Pauli X on edges incident to vertex s).\n"
                "   - Plaquette operator B_p = ∏_{j ∈ boundary(p)} Z_j (tensor product of Pauli Z on edges bounding face p).\n"
                "All star and plaquette operators commute: [A_s, B_p] = 0, [A_s, A_s'] = 0, [B_p, B_p'] = 0 because any star and plaquette share either 0 or 2 edges (and XZ = -ZX, (-1)^2 = 1).\n\n"
                "2. Code Space Definition:\n"
                "The logical code space is the ground state subspace defined by A_s |ψ⟩ = |ψ⟩ and B_p |ψ⟩ = |ψ⟩ for all s and p.\n\n"
                "3. Four-fold Ground State Degeneracy:\n"
                "On a torus, non-contractible Wilson loops wrap around the two non-trivial topological cycles (horizontal cycle 1 and vertical cycle 2):\n"
                "   Z_L1, Z_L2 (logical Z operators) and X_L1, X_L2 (logical X operators).\n"
                "These non-contractible loops commute with all local stabilizers but anticommute with each other: {Z_L1, X_L2} = 0 and {Z_L2, X_L1} = 0. Therefore, they define two independent logical qubits. The Hilbert space dimension of two logical qubits is 2^2 = 4, yielding exact 4-fold topological ground state degeneracy protected from local perturbations."
            )
        },
        {
            "instruction": "State the Gottesman-Knill Theorem and explain why Clifford circuits can be efficiently simulated on a classical computer, while universal quantum computing requires non-Clifford gates.",
            "response": (
                "1. Gottesman-Knill Theorem:\n"
                "Any quantum circuit consisting solely of:\n"
                "   - Initialization of qubits in computational basis states (|0⟩),\n"
                "   - Clifford group gates: Hadamard (H), Phase (S), and Controlled-NOT (CNOT),\n"
                "   - Measurements in the Pauli basis (X, Y, Z),\n"
                "can be perfectly simulated on a classical computer in polynomial time O(n^2) to O(n^3) using the Stabilizer Formalism.\n\n"
                "2. Proof Mechanism (Stabilizer Tableau):\n"
                "An n-qubit stabilizer state is uniquely characterized by n commuting Pauli operators g_1, ..., g_n such that g_i |ψ⟩ = |ψ⟩. Applying a Clifford gate U maps a Pauli operator P to another Pauli operator U P U^†. Because Clifford operators normalize the Pauli group, we only track the transformation of the n stabilizer generators (storing 2n bits per generator in a 2n × 2n tableau), which requires polynomial classical storage and arithmetic.\n\n"
                "3. Need for Non-Clifford Gates for Universality:\n"
                "The Clifford group is not dense in SU(2^n). To achieve universal quantum computation, an additional non-Clifford gate—such as the T-gate (π/8 phase rotation: diag(1, e^{iπ/4}))—is strictly required. Non-Clifford gates map Pauli stabilizers to linear superpositions of Pauli operators, causing an exponential breakdown of classical polynomial simulation (Magic State Distillation)."
            )
        },
        # Domain 4: Continuous Symplectic Neural Dynamics & Adjoint Solvers
        {
            "instruction": "Derive the Adjoint State Method for Neural Ordinary Differential Equations (Neural ODEs). How does it achieve O(1) memory cost during backpropagation?",
            "response": (
                "In a Neural ODE, the continuous hidden state evolution is governed by dz(t)/dt = f(z(t), t, θ). Given loss L(z(t_1)), training requires computing dL/dθ without caching forward trajectory activations:\n\n"
                "1. Definition of Adjoint State:\n"
                "Define the adjoint state as the Lagrange multiplier a(t) = ∂L/∂z(t).\n\n"
                "2. Dynamics of Adjoint State:\n"
                "By the continuous chain rule over time, the adjoint state satisfies the backward differential equation:\n"
                "   da(t)/dt = - a(t)^T · [∂f(z(t), t, θ) / ∂z(t)]\n"
                "with terminal condition a(t_1) = ∂L/∂z(t_1).\n\n"
                "3. Gradient with Respect to Parameters θ:\n"
                "The total loss gradient with respect to network parameters θ is given by the continuous integral:\n"
                "   dL/dθ = - ∫_{t_1}^{t_0} a(t)^T · [∂f(z(t), t, θ) / ∂θ] dt\n\n"
                "4. O(1) Memory Cost:\n"
                "Standard backpropagation through discrete residual layers requires caching all intermediate activation tensors z(t_i) in GPU memory (O(N) memory). With the Adjoint Method, an ODE solver solves the augmented system [z(t), a(t), dL/dθ] backward in time from t_1 to t_0 simultaneously, reconstructing z(t) continuously in reverse. Thus, peak memory consumption is strictly constant O(1) with respect to integration depth."
            )
        },
        {
            "instruction": "Explain why Symplectic Integrators (such as Störmer-Verlet) are required for simulating Hamiltonian physical neural networks, and what geometric property standard Runge-Kutta integrators violate.",
            "response": (
                "Hamiltonian mechanics governs physical systems where state is parameterized by position q and momentum p with Hamiltonian H(q, p). The equations of motion are dq/dt = ∂H/∂p and dp/dt = -∂H/∂q.\n\n"
                "1. Symplectic Structure & Liouville's Theorem:\n"
                "The phase space flow φ_t: (q(0), p(0)) → (q(t), p(t)) preserves the differential 2-form ω = dq ∧ dp. Equivalently, the Jacobian J of the transformation satisfies J^T J_0 J = J_0 where J_0 = [[0, I], [-I, 0]]. This implies exact phase space volume preservation.\n\n"
                "2. Violation by Standard Runge-Kutta Integrators:\n"
                "Standard numerical integrators (e.g., RK4, explicit Euler) are not symplectic. Over long time horizons, their numerical discretization introduces an artificial damping or pumping of energy, causing non-physical orbital spiral decay or explosive divergence.\n\n"
                "3. Störmer-Verlet / Symplectic Integrators:\n"
                "Symplectic integrators strictly preserve the symplectic 2-form ω at every discrete step. Although they do not conserve the exact Hamiltonian H, they exactly conserve a nearby shadow Hamiltonian H̃ = H + O(Δt^2). Consequently, the global energy error remains bounded without secular drift for millions of integration steps."
            )
        },
        # Domain 5: Higher Categorical Semantics & Monadic Proofs
        {
            "instruction": "State and prove the Yoneda Lemma in Category Theory. Explain its deep conceptual meaning regarding the identity and observability of mathematical objects.",
            "response": (
                "1. Statement of Yoneda Lemma:\n"
                "Let C be a locally small category, and let F: C → Set be a functor. For any object A ∈ C, there is a natural bijection between the set of natural transformations Nat(Hom_C(A, -), F) and the set F(A):\n"
                "   Nat(Hom_C(A, -), F) ≅ F(A)\n"
                "Furthermore, this bijection is natural in both A and F.\n\n"
                "2. Proof:\n"
                "Define mapping Φ: Nat(Hom_C(A, -), F) → F(A) by evaluating the natural transformation α at the identity morphism 1_A ∈ Hom_C(A, A):\n"
                "   Φ(α) = α_A(1_A) ∈ F(A)\n"
                "Conversely, given an element u ∈ F(A), define natural transformation Ψ(u): Hom_C(A, -) ⇒ F whose component at object X is given by:\n"
                "   Ψ(u)_X(f) = F(f)(u) for f ∈ Hom_C(A, X)\n"
                "Checking inverses:\n"
                "- Φ(Ψ(u)) = Ψ(u)_A(1_A) = F(1_A)(u) = id_{F(A)}(u) = u.\n"
                "- For α ∈ Nat(Hom_C(A, -), F) and f: A → X, naturality of α implies α_X ∘ Hom_C(A, f) = F(f) ∘ α_A. Applying this to 1_A yields α_X(f) = F(f)(α_A(1_A)) = Ψ(Φ(α))_X(f).\n"
                "Thus, Φ and Ψ are mutual inverses.\n\n"
                "3. Conceptual Meaning:\n"
                "An object A in a category is completely and uniquely determined by the network of its relationships with all other objects (the functor Hom_C(A, -)). Two objects are isomorphic if and only if they look identical from the perspective of every probe object in the category. Identity is entirely relational."
            )
        },
        {
            "instruction": "Define a Monad in Category Theory using an endofunctor and natural transformations (T, η, μ). Prove the Monad Laws on Kleisli composition.",
            "response": (
                "1. Categorical Definition of a Monad:\n"
                "A Monad on a category C is a triple (T, η, μ) where:\n"
                "- T: C → C is an endofunctor,\n"
                "- η: 1_C ⇒ T is the unit natural transformation,\n"
                "- μ: T^2 ⇒ T is the multiplication natural transformation,\n"
                "satisfying the coherence commutative diagrams:\n"
                "- Associativity: μ ∘ Tμ = μ ∘ μT (as transformations T^3 ⇒ T)\n"
                "- Left/Right Unit: μ ∘ Tη = id_T and μ ∘ ηT = id_T (as transformations T ⇒ T).\n\n"
                "2. Kleisli Composition (Fish Operator >=>):\n"
                "For Kleisli arrows f: A → T(B) and g: B → T(C), Kleisli composition g <=< f : A → T(C) is defined by:\n"
                "   (g <=< f)(a) = μ_C (T(g) (f(a)))\n\n"
                "3. Proof of Monad Laws under Kleisli Composition:\n"
                "- Left Identity (η <=< f = f): (η <=< f)(a) = μ (T(η) (f(a))) = (μ ∘ Tη)(f(a)) = id_T(f(a)) = f(a).\n"
                "- Right Identity (f <=< η = f): (f <=< η)(a) = μ (T(f) (η(a))) = μ (η_{T B} (f(a))) = (μ ∘ ηT)(f(a)) = id_T(f(a)) = f(a).\n"
                "- Associativity ((h <=< g) <=< f = h <=< (g <=< f)): Follows directly from associativity of endofunctor application T and the multiplication law μ ∘ Tμ = μ ∘ μT."
            )
        }
    ]
    return data

# Let's save curriculum data
curr = build_training_curriculum()
with open(STUDY_DIR / "frontier_curriculum.json", "w", encoding="utf-8") as f:
    json.dump(curr, f, indent=2)
print(f"[+] Saved {len(curr)} core frontier training lessons.")
