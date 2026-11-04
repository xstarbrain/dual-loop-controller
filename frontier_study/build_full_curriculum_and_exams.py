"""
Comprehensive generator for Frontier Advanced Training Curriculum
and 3 distinct 50-Question Examination Pools (Round 1, Round 2, Round 3).
"""

import json
from pathlib import Path

STUDY_DIR = Path("frontier_study")
STUDY_DIR.mkdir(parents=True, exist_ok=True)

# 1. EXPANDED CURRICULUM (Covering all 5 core frontier domains)
def get_master_curriculum():
    lessons = [
        # Domain 1: Distributed Consensus & Invariants
        {
            "domain": "Distributed Consensus",
            "instruction": "Prove the State Machine Safety property of the Raft consensus algorithm. Explain why a leader can never overwrite or commit an entry at an index where another leader already committed a different entry.",
            "response": (
                "Raft guarantees State Machine Safety through the Leader Completeness Invariant:\n\n"
                "1. Leader Completeness: If a log entry is committed at term T, that entry is present in all leaders elected at terms T' > T.\n"
                "Proof: An entry is committed only when replicated on a strict majority quorum Q1. Electing a candidate in term T' requires votes from a majority quorum Q2. By the pigeonhole principle, Q1 and Q2 must intersect at some node v. Node v accepted the entry in term T. Under Raft's RequestVote rule, an elector only votes for a candidate whose log is at least as up-to-date as its own. Thus, candidate in term T' must have a log containing the committed entry.\n\n"
                "2. Log Matching Invariant: If two logs contain an entry with the same index and term, they are identical in all preceding entries.\n\n"
                "3. Append-Only Leader: A leader never overwrites or truncates its own log entries.\n"
                "Combining these, no leader will ever propose or commit an alternate entry at that index in any term. State machine safety holds unconditionally."
            )
        },
        {
            "domain": "Distributed Consensus",
            "instruction": "Explain how Paxos resolves the split-brain hazard and why ballot number monotonicity is essential to guaranteeing safety in the Synod protocol.",
            "response": (
                "Paxos resolves split-brain hazards through monotonic ballot numbers and quorum intersections in two phases:\n\n"
                "1. Ballot Monotonicity: Every proposal has a unique, monotonically increasing ballot number n = (round_num, node_id). Acceptors promise to reject any proposal with ballot < n once they promise on ballot n.\n\n"
                "2. Phase 1 (Prepare): Proposer broadcasts Prepare(n). Acceptors return Promise(max_accepted_ballot, accepted_value). If a quorum Q1 responds, the proposer is forced to adopt the value with the highest ballot number reported among promises.\n\n"
                "3. Quorum Intersection: If value v was chosen in an earlier ballot, any future quorum Q2 of promises must intersect Q1 at at least one node that accepted v. That node forces the new proposer to propose v.\n\n"
                "4. Elimination of Split-Brain: No two proposers can commit conflicting values because only one proposer can hold the highest accepted ballot in overlapping majorities."
            )
        },
        {
            "domain": "Distributed Consensus",
            "instruction": "Formulate the mathematical definition of a Conflict-Free Replicated Data Type (CRDT) under the state-based (CvRDT) model. What algebraic structure guarantees Strong Eventual Consistency?",
            "response": (
                "A state-based CRDT (CvRDT) is defined as a join-semilattice (S, ⊔, ≤):\n\n"
                "1. Join-Semilattice Properties:\n"
                "- Commutativity: x ⊔ y = y ⊔ x (order of message delivery is irrelevant).\n"
                "- Associativity: (x ⊔ y) ⊔ z = x ⊔ (y ⊔ z) (message grouping is irrelevant).\n"
                "- Idempotence: x ⊔ x = x (duplicate message delivery has no effect).\n\n"
                "2. Partial Order: x ≤ y if and only if x ⊔ y = y. States grow monotonically over time.\n\n"
                "3. Strong Eventual Consistency Theorem: Any two replicas that receive the same updates will independently apply the merge operator m(s1, s2) = s1 ⊔ s2 and converge to the identical state without distributed coordination locks or consensus rounds."
            )
        },
        {
            "domain": "Distributed Consensus",
            "instruction": "Analyze the primary failure and view change mechanism in Viewstamped Replication (VR Revisited).",
            "response": (
                "In Viewstamped Replication, view changes prevent data loss when a primary crashes:\n\n"
                "1. Trigger: Backups suspect the primary via heartbeats, transition to VIEW_CHANGE state, increment view number to v + 1, and broadcast START_VIEW_CHANGE(v + 1).\n\n"
                "2. DO_VIEW_CHANGE Quorum: Once a replica collects f START_VIEW_CHANGE messages, it sends DO_VIEW_CHANGE(v + 1, log, op_number, commit_number, latest_normal_view) to the designated new primary.\n\n"
                "3. Authoritative Log Selection: The new primary collects f + 1 DO_VIEW_CHANGE messages. It picks the log with the highest latest_normal_view, breaking ties with the highest op_number.\n\n"
                "4. START_VIEW Broadcast: The new primary adopts this log and sends START_VIEW(v + 1, log, op_number, commit_number). Replicas update their state, guaranteeing that every committed operation is preserved."
            )
        },
        {
            "domain": "Distributed Consensus",
            "instruction": "Distinguish Lamport Timestamps from Vector Clocks. Why can Lamport timestamps establish total order but cannot detect concurrent causality violations?",
            "response": (
                "1. Lamport Timestamps: Each process maintains a scalar counter L. On local event, L = L + 1. On message send, attach L. On receive, L_recv = max(L_local, L_msg) + 1. Total order is obtained by tie-breaking (L, process_id). However, if L(a) < L(b), event a did NOT necessarily cause event b; they may be completely concurrent. Lamport clocks establish consistent total order, but cannot determine causality.\n\n"
                "2. Vector Clocks: Each process maintains a vector V of size N. On event on process i, V[i] = V[i] + 1. On receive, V_local[k] = max(V_local[k], V_msg[k]) for all k. Event a causes event b (a → b) if and only if V(a) ≤ V(b) and V(a) ≠ V(b).\n\n"
                "3. Concurrency Detection: If neither V(a) ≤ V(b) nor V(b) ≤ V(a), then a || b (the events are concurrent). Vector clocks provide bidirectional causality isomorphism, whereas Lamport clocks are only unidirectional."
            )
        },

        # Domain 2: Cryptographic Zero-Knowledge & Polynomial Commitments
        {
            "domain": "Zero-Knowledge Cryptography",
            "instruction": "Derive the reduction of an arithmetic computation into a Rank-1 Constraint System (R1CS) and explain why every constraint must satisfy the bilinear form <A, s> * <B, s> = <C, s>.",
            "response": (
                "1. Arithmetic Circuit Representation: A circuit over finite field F consists of addition wires and multiplication gates. Addition is linear and can be absorbed into coefficient vectors without extra constraints.\n\n"
                "2. Witness Vector: Let s = [1, x_1, x_2, ..., x_n]^T ∈ F^(n+1) be the full assignment vector (including public inputs, private inputs, and intermediate wires).\n\n"
                "3. R1CS Matrices: For m multiplication gates, we construct three sparse matrices A, B, C ∈ F^(m × (n+1)). Each row i represents a multiplication gate:\n"
                "   Left Wire: <A_i, s> = ∑ A_{i,k} s_k\n"
                "   Right Wire: <B_i, s> = ∑ B_{i,k} s_k\n"
                "   Output Wire: <C_i, s> = ∑ C_{i,k} s_k\n"
                "The constraint equation is: (A · s) ∘ (B · s) = C · s (where ∘ is Hadamard product). Each constraint represents rank-1 bilinear product, transforming arbitrary algebraic relations into verifiable quadratic equations."
            )
        },
        {
            "domain": "Zero-Knowledge Cryptography",
            "instruction": "Explain the mechanics of the Kate-Zaverucha-Goldberg (KZG) polynomial commitment scheme. How does the verifier check that P(z) = y using a single pairing equation?",
            "response": (
                "KZG operates over bilinear groups (G_1, G_2, G_T, e) with trusted setup {[τ^i]_1}_{i=0}^d and [τ]_2:\n\n"
                "1. Commitment: Prover commits to polynomial P(x) = ∑ c_i x^i as C = [P(τ)]_1 = ∑ c_i [τ^i]_1 ∈ G_1.\n\n"
                "2. Evaluation Proof: To prove P(z) = y, note that P(x) - y has root at x = z. Thus, P(x) - y = Q(x)(x - z). The prover computes quotient commitment π = [Q(τ)]_1.\n\n"
                "3. Pairing Check: The verifier checks:\n"
                "   e(C - [y]_1, [1]_2) = e(π, [τ]_2 - [z]_2)\n"
                "Correctness:\n"
                "   e([P(τ) - y]_1, [1]_2) = e([Q(τ)(τ - z)]_1, [1]_2) = e([Q(τ)]_1, [(τ - z)]_2).\n"
                "Because e is bilinear and non-degenerate, if P(z) ≠ y, finding a valid π requires solving the Discrete Logarithm problem in G_1, which is computationally infeasible."
            )
        },
        {
            "domain": "Zero-Knowledge Cryptography",
            "instruction": "State the Schwartz-Zippel Lemma and explain its foundational role in Succinct Non-Interactive Arguments of Knowledge (SNARKs).",
            "response": (
                "1. Schwartz-Zippel Lemma: Let P ∈ F[x_1, ..., x_n] be a non-zero multivariate polynomial of total degree d over finite field F. For a finite subset S ⊆ F, choosing r_1, ..., r_n uniformly at random from S gives:\n"
                "   Pr[P(r_1, ..., r_n) = 0] ≤ d / |S|\n\n"
                "2. Application to SNARKs: In SNARKs, verifying that a prover's computation satisfies constraints across all domain points H is reformulated as polynomial identity P(x) = H(x)Z(x). Directly testing every point takes O(N) evaluations. Instead, the verifier checks equality at a single secret or random challenge point r. Since |F| ≈ 2^256 and degree d ≈ 2^20, the probability of false witness acceptance is ≤ 2^20 / 2^256 = 2^-236. This reduces polynomial verification to an O(1) random evaluation."
            )
        },
        {
            "domain": "Zero-Knowledge Cryptography",
            "instruction": "Explain how PLONK uses the Grand Product Argument and permutation polynomial Z(x) to enforce copy constraints between circuit wires.",
            "response": (
                "In PLONK, copy constraints ensure that output wires of certain gates connect to input wires of other gates:\n\n"
                "1. Permutation Representation: Wire positions are encoded as indices i ∈ [3n]. A permutation σ: [3n] → [3n] maps positions that must hold equal values. The copy constraint requires f(σ(i)) = f(i).\n\n"
                "2. Multiset Equality via Grand Product: By evaluating randomized accumulators, two multisets {a_i} and {b_i} are identical if ∏ (a_i + β i + γ) = ∏ (b_i + β σ(i) + γ) for random challenges β, γ.\n\n"
                "3. Permutation Polynomial Z(X): Prover constructs polynomial Z(X) satisfying:\n"
                "   Z(g) = 1, and Z(g · x) = Z(x) · [(a(x) + β k_1 x + γ)(b(x) + β k_2 x + γ)(c(x) + β k_3 x + γ)] / [(a(x) + β S_{σ1}(x) + γ)(b(x) + β S_{σ2}(x) + γ)(c(x) + β S_{σ3}(x) + γ)]\n"
                "4. Succinct Check: Verifying Z(g^{n+1}) = 1 and telescoping transition polynomial checks across all roots of unity confirms all wire wiring constraints simultaneously."
            )
        },

        # Domain 3: Quantum Information & Topological Stabilizers
        {
            "domain": "Quantum Stabilizers & Topological Codes",
            "instruction": "Explain the Toric Code as a topological quantum error-correcting code. How do star and plaquette operators define the code space, and why is the ground state 4-fold degenerate on a torus?",
            "response": (
                "The Toric Code (Kitaev) is defined on a 2D square lattice on a torus T^2:\n\n"
                "1. Operators: Physical qubits reside on edges. For each vertex s, star operator A_s = ∏_{i ∈ star(s)} X_i. For each face p, plaquette operator B_p = ∏_{j ∈ boundary(p)} Z_j.\n"
                "All operators commute ([A_s, B_p] = 0) because any star and plaquette share either 0 or 2 edges, and Pauli anti-commutation (-1)^2 = +1.\n\n"
                "2. Code Subspace: The ground state is defined by A_s |ψ⟩ = +1 |ψ⟩ and B_p |ψ⟩ = +1 |ψ⟩ for all s and p.\n\n"
                "3. Four-fold Ground State Degeneracy: Non-contractible loops wrapping around the torus cannot be deformed into contractible stabilizers:\n"
                "   - Logical operators Z_L1, Z_L2 wrap around horizontal and vertical cycles.\n"
                "   - Logical operators X_L1, X_L2 wrap around dual cycles.\n"
                "They commute with all stabilizers but {Z_L1, X_L2} = 0 and {Z_L2, X_L1} = 0. They encode k = 2 logical qubits, yielding a 2^2 = 4-fold degenerate topological code space immune to local perturbations."
            )
        },
        {
            "domain": "Quantum Stabilizers & Topological Codes",
            "instruction": "State the Gottesman-Knill Theorem and explain why Clifford circuits can be efficiently simulated on a classical computer, while universal quantum computing requires non-Clifford gates.",
            "response": (
                "1. Gottesman-Knill Theorem: Any quantum circuit containing only state preparation in computational basis, Clifford gates (Hadamard H, Phase S, CNOT), and Pauli measurements can be classically simulated in polynomial time O(n^2) to O(n^3).\n\n"
                "2. Mechanism: An n-qubit stabilizer state is uniquely described by n mutually commuting Pauli operators g_1, ..., g_n such that g_i |ψ⟩ = |ψ⟩. Clifford gates normalize the Pauli group (U P U^† ∈ Pauli). Simulating the circuit reduces to tracking how 2n binary vectors (representing the stabilizer generators) transform under symplectic linear algebra over GF(2).\n\n"
                "3. Universality & Magic States: The Clifford group is not dense in SU(2^n). To achieve universal quantum computation, an additional non-Clifford gate such as the T-gate (T = diag(1, e^{iπ/4})) is required. Non-Clifford operations map Pauli operators to superpositions, causing exponential simulation overhead. Universal fault-tolerant systems synthesize non-Clifford gates via Magic State Distillation."
            )
        },
        {
            "domain": "Quantum Stabilizers & Topological Codes",
            "instruction": "Describe the Surface Code syndrome extraction cycle. How are bit-flip (X) and phase-flip (Z) errors detected and corrected without destroying quantum superposition?",
            "response": (
                "The Surface Code arranges data qubits and ancillary measurement qubits on a 2D planar patch:\n\n"
                "1. Ancillary Qubits: Ancilla qubits are placed at vertices and face centers between data qubits.\n\n"
                "2. Syndrome Extraction Cycle:\n"
                "- Z-Syndrome (detects X bit-flips): Ancilla prepared in |0⟩, CNOT gates applied from adjacent data qubits to ancilla, ancilla measured in Z-basis.\n"
                "- X-Syndrome (detects Z phase-flips): Ancilla prepared in |+⟩, CNOT gates applied from ancilla to adjacent data qubits, ancilla measured in X-basis.\n\n"
                "3. Superposition Preservation: Because measurements project only the multi-qubit parity eigenvalues (eigenvalues ±1 of stabilizer generators A_s and B_p) rather than individual data qubits, the quantum superposition of logical information is completely preserved.\n\n"
                "4. Minimum-Weight Perfect Matching (MWPM): Measurement outcomes yielding -1 form defects (syndrome endpoints). Decoding algorithms (MWPM or Union-Find) match error chains across cycles to determine the most probable Pauli correction operator."
            )
        },

        # Domain 4: Continuous Symplectic Neural Dynamics & Adjoint Solvers
        {
            "domain": "Symplectic Dynamics & Neural ODEs",
            "instruction": "Derive the Adjoint State Method for Neural Ordinary Differential Equations (Neural ODEs). How does it achieve O(1) memory cost during backpropagation?",
            "response": (
                "In a Neural ODE, hidden states evolve continuously: dz(t)/dt = f(z(t), t, θ). Given loss L(z(t_1)):\n\n"
                "1. Adjoint State Definition: Define a(t) = ∂L/∂z(t) as the continuous Lagrange multiplier.\n\n"
                "2. Continuous Adjoint Dynamics: By the continuous chain rule over time:\n"
                "   da(t)/dt = - a(t)^T · [∂f(z(t), t, θ) / ∂z(t)]\n"
                "with terminal boundary condition a(t_1) = ∂L/∂z(t_1).\n\n"
                "3. Parameter Gradient: The total derivative with respect to θ is:\n"
                "   dL/dθ = - ∫_{t_1}^{t_0} a(t)^T · [∂f(z(t), t, θ) / ∂θ] dt\n\n"
                "4. O(1) Memory Cost: Standard deep networks store all forward activations z(t_i) for backprop (O(N) memory). With the Adjoint Method, the backward solver simultaneously integrates [z(t), a(t), dL/dθ] backward from t_1 to t_0. Since z(t) is reconstructed dynamically in reverse, memory footprint is strictly constant O(1) regardless of integration depth."
            )
        },
        {
            "domain": "Symplectic Dynamics & Neural ODEs",
            "instruction": "Explain why Symplectic Integrators (such as Störmer-Verlet) are required for simulating Hamiltonian physical neural networks, and what geometric property standard Runge-Kutta integrators violate.",
            "response": (
                "Hamiltonian systems follow dq/dt = ∂H/∂p and dp/dt = -∂H/∂q in phase space (q, p):\n\n"
                "1. Symplectic Geometry: The Hamiltonian flow φ_t preserves the differential 2-form ω = dq ∧ dp. The Jacobian J satisfies J^T J_0 J = J_0 where J_0 = [[0, I], [-I, 0]], preserving phase space volume (Liouville's theorem).\n\n"
                "2. Standard Integrators (RK4, Explicit Euler): Standard Runge-Kutta methods are non-symplectic. They introduce artificial dissipation or numerical energy growth, causing artificial spiral decay or explosive instability in planetary/molecular simulations.\n\n"
                "3. Symplectic Integrators (Störmer-Verlet): Symplectic integrators preserve the symplectic 2-form exactly at every time step. By backward error analysis, they conserve a nearby shadow Hamiltonian H̃ = H + O(Δt^2). Energy error remains globally bounded without secular drift for millions of integration steps."
            )
        },
        {
            "domain": "Symplectic Dynamics & Neural ODEs",
            "instruction": "Explain the concept of Dynamic Lipschitz Matrix Regularization in continuous-depth deep learning and how it guarantees contraction mapping stability.",
            "response": (
                "1. Lipschitz Boundedness: A continuous vector field f: R^d → R^d is Lipschitz continuous if ||f(x) - f(y)|| ≤ L ||x - y||. For an ODE dz/dt = f(z), stability requires controlling the matrix logarithmic norm μ(J_f) = sup_{v ≠ 0} (v^T (J_f + J_f^T) v) / (2 ||v||^2).\n\n"
                "2. Contraction Mapping: If the symmetric Jacobian (J_f + J_f^T) / 2 has maximum eigenvalue λ_max ≤ -c < 0, all trajectories converge exponentially to a unique trajectory: ||z_1(t) - z_2(t)|| ≤ e^{-ct} ||z_1(0) - z_2(0)||. This forms a contraction metric.\n\n"
                "3. Dynamic Matrix Regularization: During inference or training, constraining the spectral norm of weight matrices W (e.g., via spectral normalization or singular value thresholding) prevents exploding gradients and guarantees robust contractive stability against adversarial perturbations."
            )
        },

        # Domain 5: Higher Categorical Semantics & Monadic Proofs
        {
            "domain": "Category Theory & Formal Semantics",
            "instruction": "State and prove the Yoneda Lemma in Category Theory. Explain its deep conceptual meaning regarding the identity and observability of mathematical objects.",
            "response": (
                "1. Statement: For a locally small category C, an object A ∈ C, and a functor F: C → Set, there exists a natural isomorphism:\n"
                "   Nat(Hom_C(A, -), F) ≅ F(A)\n\n"
                "2. Proof: Define Φ: Nat(Hom_C(A, -), F) → F(A) by evaluating the transformation α at identity 1_A: Φ(α) = α_A(1_A). Conversely, for u ∈ F(A), define natural transformation Ψ(u) with components Ψ(u)_X(f) = F(f)(u) for f ∈ Hom_C(A, X).\n"
                "- Φ(Ψ(u)) = Ψ(u)_A(1_A) = F(1_A)(u) = u.\n"
                "- For α ∈ Nat and f: A → X, naturality implies α_X(f) = F(f)(α_A(1_A)) = Ψ(Φ(α))_X(f).\n"
                "Thus Φ and Ψ are mutual inverses.\n\n"
                "3. Conceptual Meaning: An object A is completely and uniquely characterized by the network of its morphisms with all other objects in the category. The Yoneda embedding y: C → [C^op, Set] is fully faithful, meaning mathematical identity is entirely relational: to know an object is to know how every other object interacts with it."
            )
        },
        {
            "domain": "Category Theory & Formal Semantics",
            "instruction": "Define a Monad in Category Theory using an endofunctor and natural transformations (T, η, μ). Prove the Monad Laws on Kleisli composition.",
            "response": (
                "1. Monad Definition: A monad on category C is a triple (T, η, μ) where T: C → C is an endofunctor, η: 1_C ⇒ T is the unit, and μ: T^2 ⇒ T is multiplication, satisfying:\n"
                "- Associativity: μ ∘ Tμ = μ ∘ μT\n"
                "- Unit Laws: μ ∘ Tη = id_T and μ ∘ ηT = id_T\n\n"
                "2. Kleisli Composition (Fish operator >=>): For arrows f: A → T(B) and g: B → T(C), composition (g <=< f): A → T(C) is:\n"
                "   (g <=< f)(a) = μ_C (T(g)(f(a)))\n\n"
                "3. Proof of Laws:\n"
                "- Left Unit: (η <=< f)(a) = μ(T(η)(f(a))) = (μ ∘ Tη)(f(a)) = f(a).\n"
                "- Right Unit: (f <=< η)(a) = μ(T(f)(η(a))) = μ(η_{T B}(f(a))) = (μ ∘ ηT)(f(a)) = f(a).\n"
                "- Associativity: ((h <=< g) <=< f) = (h <=< (g <=< f)) follows directly from associativity of functor application and μ ∘ Tμ = μ ∘ μT."
            )
        },
        {
            "domain": "Category Theory & Formal Semantics",
            "instruction": "Formulate the Currying Adjunction in Cartesian Closed Categories (CCC) and explain its relationship to the simply typed lambda calculus via the Curry-Howard-Lambek correspondence.",
            "response": (
                "1. Cartesian Closed Category (CCC): A category C with finite products and exponential objects (function spaces). For any objects B and C, the exponential object C^B represents morphisms B → C.\n\n"
                "2. Currying Adjunction: The product functor (- × B) is left adjoint to the exponential functor (-)^B:\n"
                "   Hom_C(A × B, C) ≅ Hom_C(A, C^B)\n"
                "For any morphism f: A × B → C, there exists a unique morphism curry(f): A → C^B such that eval ∘ (curry(f) × id_B) = f.\n\n"
                "3. Curry-Howard-Lambek Correspondence: Establishes a three-way isomorphism:\n"
                "- Logic: Intuitionistic Propositional Logic (Conjunction A ∧ B corresponds to Product A × B, Implication A ⇒ B corresponds to Exponential B^A).\n"
                "- Computation: Simply Typed Lambda Calculus (Pair types correspond to Products, Function types A → B correspond to Exponentials, Beta-reduction corresponds to adjunction counit evaluation).\n"
                "- Category Theory: Cartesian Closed Categories provide the canonical categorical model of computation."
            )
        }
    ]
    return lessons

# 2. GENERATE 50 EXAM QUESTIONS WITH STRICT RUBRICS FOR ROUND 1
def generate_exam_round_1():
    # 50 challenging open-ended practical/generative questions (10 per domain)
    # Each question has 4 strict scoring rubrics (0.5 pt each = 2.0 pt total)
    questions = []
    
    # D1: Distributed Consensus (10 Qs)
    d1_prompts = [
        ("EX1_01", "Distributed Consensus", 
         "In the Raft consensus protocol, explain step-by-step why an election split occurs when multiple candidates launch concurrent RequestVote RPCs, and how randomized election timeouts mathematically minimize split election deadlocks.",
         ["election split", "randomized timeout", "split-vote deadlock", "majority quorum"]),
        ("EX1_02", "Distributed Consensus",
         "Explain how Raft log compaction via Snapshotting interacts with followers that lag far behind. Detail the role of the InstallSnapshot RPC.",
         ["snapshotting", "installsnapshot", "state machine", "log compaction"]),
        ("EX1_03", "Distributed Consensus",
         "Detail how Multi-Paxos optimizes the standard Synod protocol into a single round-trip for steady-state client requests once a stable leader is established.",
         ["multi-paxos", "phase 1 skip", "stable leader", "accept phase"]),
        ("EX1_04", "Distributed Consensus",
         "In distributed storage systems, what is the Split-Brain scenario? Prove mathematically why a system of N replicas can only tolerate f = (N - 1) / 2 crash faults under majority consensus.",
         ["split-brain", "strict majority", "quorum intersection", "f crash faults"]),
        ("EX1_05", "Distributed Consensus",
         "Describe the Observed-Remove Set (OR-Set) CRDT. How does it resolve concurrent additions and deletions of the exact same element across distributed replicas?",
         ["or-set", "unique tag", "add-wins", "concurrent delete"]),
        ("EX1_06", "Distributed Consensus",
         "Explain the primary difference between crash-fault-tolerant (CFT) protocols like Raft and Byzantine fault-tolerant (BFT) protocols like PBFT regarding quorum sizes and cryptographic signatures.",
         ["byzantine fault", "3f + 1 nodes", "cryptographic digest", "equivocation"]),
        ("EX1_07", "Distributed Consensus",
         "Explain the term 'Read Index' optimization in Raft. How does a leader serve linearizable read requests without appending a new entry to the replicated log?",
         ["read index", "linearizable read", "heartbeat quorum", "commit index"]),
        ("EX1_08", "Distributed Consensus",
         "Analyze the CAP theorem in modern distributed systems. Why is network partition tolerance (P) unavoidable in physical networks, leaving only the tradeoff between Consistency (C) and Availability (A)?",
         ["network partition", "linearizability", "availability tradeoff", "flp impossibility"]),
        ("EX1_09", "Distributed Consensus",
         "Explain how Vector Clocks handle concurrent writes in distributed key-value stores (e.g., Dynamo DB) and what role client-side sibling resolution plays.",
         ["vector clock", "concurrent sibling", "causal conflict", "client resolution"]),
        ("EX1_10", "Distributed Consensus",
         "In distributed transactions, contrast Two-Phase Commit (2PC) with Three-Phase Commit (3PC). Why does 2PC suffer from blocking upon coordinator failure, and how does 3PC's PreCommit phase alleviate this?",
         ["two-phase commit", "three-phase commit", "blocking coordinator", "precommit phase"]),
    ]

    # D2: Zero-Knowledge & Commitments (10 Qs)
    d2_prompts = [
        ("EX1_11", "Zero-Knowledge Cryptography",
         "Explain how an arithmetic circuit with addition and multiplication gates is converted into a system of Quadratic Arithmetic Programs (QAP). What role do Lagrange basis polynomials play?",
         ["quadratic arithmetic program", "lagrange interpolation", "polynomial division", "target polynomial"]),
        ("EX1_12", "Zero-Knowledge Cryptography",
         "In the KZG commitment scheme, explain why the trusted setup powers of tau must have toxic waste secret tau discarded. What catastrophic forgery occurs if tau is leaked?",
         ["toxic waste", "discrete logarithm", "polynomial forgery", "powers of tau"]),
        ("EX1_13", "Zero-Knowledge Cryptography",
         "Explain the Fiat-Shamir Heuristic. How does it convert an interactive zero-knowledge public-coin proof into a non-interactive proof (NIZK) using a cryptographic hash function?",
         ["fiat-shamir", "random oracle", "non-interactive", "challenge hash"]),
        ("EX1_14", "Zero-Knowledge Cryptography",
         "Explain the concept of Bilinear Pairings e: G1 x G2 -> GT. What properties (bilinearity and non-degeneracy) make them essential for cryptographic verification?",
         ["bilinearity", "non-degeneracy", "elliptic curve", "pairing verification"]),
        ("EX1_15", "Zero-Knowledge Cryptography",
         "In PLONK, explain how polynomial commitment schemes verify that a quotient polynomial t(X) divides the constraint polynomial without revealing witness values.",
         ["quotient polynomial", "vanishing polynomial", "random evaluation", "zero-knowledge witness"]),
        ("EX1_16", "Zero-Knowledge Cryptography",
         "What is the difference between Transparent SNARKs (like STARKs) and SNARKs requiring a Trusted Setup (like Groth16)? How do STARKs eliminate trusted setups using FRI?",
         ["trusted setup", "stark", "fast reed-solomon", "hash-based"]),
        ("EX1_17", "Zero-Knowledge Cryptography",
         "Explain the role of Range Proofs in confidential transactions (e.g., Bulletproofs) and how inner product arguments achieve logarithmic proof size.",
         ["range proof", "bulletproofs", "inner product argument", "logarithmic size"]),
        ("EX1_18", "Zero-Knowledge Cryptography",
         "Analyze the Rank-1 Constraint System (R1CS) satisfiability problem. Why is finding a witness vector for an arbitrary R1CS NP-complete?",
         ["r1cs", "witness vector", "np-complete", "boolean circuit reduction"]),
        ("EX1_19", "Zero-Knowledge Cryptography",
         "Explain how recursive SNARKs (proof of a proof) operate. How do cycles of pairing-friendly elliptic curves (like Pasta curves: Pallas and Vesta) prevent field emulation overhead?",
         ["recursive snark", "elliptic curve cycle", "pasta curves", "proof composition"]),
        ("EX1_20", "Zero-Knowledge Cryptography",
         "Explain how polynomial commitments in Halo 2 achieve recursive verification without a trusted setup using the accumulation scheme.",
         ["halo 2", "accumulation scheme", "ipa commitment", "no trusted setup"]),
    ]

    # D3: Quantum Information & Stabilizers (10 Qs)
    d3_prompts = [
        ("EX1_21", "Quantum Stabilizers & Topological Codes",
         "In quantum error correction, explain the Stabilizer Formalism. What mathematical conditions must a subgroup of the Pauli group satisfy to define a valid quantum code?",
         ["pauli group", "abelian subgroup", "-i not in group", "code subspace"]),
        ("EX1_22", "Quantum Stabilizers & Topological Codes",
         "Explain how the 9-qubit Shor Code protects against both arbitrary bit-flip (X) and phase-flip (Z) errors by concatenating repetition codes.",
         ["shor code", "concatenation", "bit-flip error", "phase-flip error"]),
        ("EX1_23", "Quantum Stabilizers & Topological Codes",
         "Describe the physical difference between star operators A_s and plaquette operators B_p in the Toric Code, and how error chains create anyonic excitations (e and m particles).",
         ["star operator", "plaquette operator", "anyonic excitation", "error chain endpoints"]),
        ("EX1_24", "Quantum Stabilizers & Topological Codes",
         "Explain why transversal gates in quantum error correction cannot achieve universal fault-tolerant quantum computing on their own (Eastin-Knill Theorem).",
         ["eastin-knill theorem", "transversal gates", "no-go theorem", "universal computing"]),
        ("EX1_25", "Quantum Stabilizers & Topological Codes",
         "Detail the process of Magic State Distillation. Why is it used to inject non-Clifford T gates into surface codes?",
         ["magic state distillation", "t-gate", "noisy ancilla", "clifford filtering"]),
        ("EX1_26", "Quantum Stabilizers & Topological Codes",
         "In surface code decoding, explain how Minimum-Weight Perfect Matching (MWPM) pairs defect syndromes to correct errors with threshold ~1%.",
         ["mwpm", "syndrome defect", "blossom algorithm", "error threshold"]),
        ("EX1_27", "Quantum Stabilizers & Topological Codes",
         "Explain the concept of Topological Braiding of anyons. How does non-Abelian anyon exchange realize fault-tolerant quantum gates geometrically?",
         ["topological braiding", "non-abelian anyons", "geometric phase", "decoherence immunity"]),
        ("EX1_28", "Quantum Stabilizers & Topological Codes",
         "What is the Quantum Threshold Theorem? State the condition under which arbitrarily long quantum computations can be executed reliably.",
         ["threshold theorem", "physical error rate", "fault tolerance", "concatenated scaling"]),
        ("EX1_29", "Quantum Stabilizers & Topological Codes",
         "Explain how the Steane [[7, 1, 3]] code works and why it is classified as a CSS (Calderbank-Shor-Steane) code.",
         ["steane code", "css code", "classical hamming code", "transversal clifford"]),
        ("EX1_30", "Quantum Stabilizers & Topological Codes",
         "Explain the difference between active quantum error correction and passive fault-tolerance in topological systems.",
         ["active error correction", "passive fault-tolerance", "energy gap", "syndrome measurement"]),
    ]

    # D4: Symplectic Dynamics & Neural ODEs (10 Qs)
    d4_prompts = [
        ("EX1_31", "Symplectic Dynamics & Neural ODEs",
         "Explain the continuous adjoint state equation in Neural ODEs. Why does the adjoint vector propagate backward in time with the negative transpose of the Jacobian?",
         ["adjoint state", "lagrange multiplier", "jacobian transpose", "backward propagation"]),
        ("EX1_32", "Symplectic Dynamics & Neural ODEs",
         "In Hamiltonian Neural Networks (HNNs), how does parameterizing the network with a scalar energy function H(q, p) enforce exact energy conservation?",
         ["hamiltonian neural network", "phase space", "symplectic gradient", "energy conservation"]),
        ("EX1_33", "Symplectic Dynamics & Neural ODEs",
         "Compare standard explicit Euler integration with the symplectic Euler method. Why does explicit Euler fail on harmonic oscillators while symplectic Euler preserves closed orbits?",
         ["symplectic euler", "harmonic oscillator", "phase space area", "shadow hamiltonian"]),
        ("EX1_34", "Symplectic Dynamics & Neural ODEs",
         "Explain the role of Dynamic Lipschitz Constant control L(t) in preventing exploding trajectories and ensuring contractive stability in deep continuous-time networks.",
         ["lipschitz constant", "contractive stability", "exploding trajectories", "spectral norm"]),
        ("EX1_35", "Symplectic Dynamics & Neural ODEs",
         "What is the Adjoint Sensitivity Method's main numerical pitfall when solving stiff differential equations, and how do checkpointing techniques alleviate it?",
         ["stiff ode", "numerical error accumulation", "reverse reconstruction", "checkpointing"]),
        ("EX1_36", "Symplectic Dynamics & Neural ODEs",
         "Explain how Continuous Normalizing Flows (CNFs) utilize the instantaneous change of variables formula: d log p(z(t)) / dt = - Tr(df / dz).",
         ["continuous normalizing flow", "instantaneous change of variables", "trace of jacobian", "hutchinson estimator"]),
        ("EX1_37", "Symplectic Dynamics & Neural ODEs",
         "Explain how Hutchinson's Trace Estimator enables unbiased O(1) computation of Tr(df/dz) in high-dimensional continuous neural flows using random Gaussian vectors.",
         ["hutchinson trace estimator", "random vector", "unbiased estimate", "quadratic form"]),
        ("EX1_38", "Symplectic Dynamics & Neural ODEs",
         "Describe the concept of Neural Controlled Differential Equations (Neural CDEs) and why they are continuous-time analogues to Recurrent Neural Networks (RNNs).",
         ["neural cde", "riemann-stieltjes integral", "rough paths", "continuous rnn"]),
        ("EX1_39", "Symplectic Dynamics & Neural ODEs",
         "Explain how Symplectic Störmer-Verlet integrators maintain invariant phase volume according to Liouville's theorem in N-body dynamical systems.",
         ["st\\u00f6rmer-verlet", "liouville theorem", "phase volume", "symplectic 2-form"]),
        ("EX1_40", "Symplectic Dynamics & Neural ODEs",
         "Explain why bounding the logarithmic norm of a neural vector field guarantees that perturbed trajectories do not diverge exponentially.",
         ["logarithmic norm", "trajectory divergence", "contraction metric", "matrix measure"]),
    ]

    # D5: Category Theory & Semantics (10 Qs)
    d5_prompts = [
        ("EX1_41", "Category Theory & Formal Semantics",
         "State the categorical definition of a Natural Transformation between two functors F, G: C -> D. Write the commutative diagram that must hold for every morphism f: A -> B.",
         ["natural transformation", "commutative diagram", "functor component", "naturality square"]),
        ("EX1_42", "Category Theory & Formal Semantics",
         "Explain why the Yoneda embedding y: C -> [C^op, Set] is fully faithful, and what that implies about morphisms in C versus natural transformations between presheaves.",
         ["yoneda embedding", "fully faithful", "presheaf category", "isomorphism preservation"]),
        ("EX1_43", "Category Theory & Formal Semantics",
         "Define an Adjunction between two categories (F -| G). What are the unit eta and counit epsilon of an adjunction, and what triangular identities must they satisfy?",
         ["adjunction", "unit eta", "counit epsilon", "triangle identities"]),
        ("EX1_44", "Category Theory & Formal Semantics",
         "Explain the categorical semantics of Monads in functional programming. How does the Kleisli category model effectful computations (such as state, I/O, or non-determinism)?",
         ["kleisli category", "monadic effects", "endofunctor", "bind operation"]),
        ("EX1_45", "Category Theory & Formal Semantics",
         "Explain the concept of an Exponential Object B^A in Cartesian Closed Categories (CCC) and write the universal property that characterizes it via evaluation morphism.",
         ["exponential object", "evaluation morphism", "cartesian closed category", "currying"]),
        ("EX1_46", "Category Theory & Formal Semantics",
         "What is a Topos in category theory? Explain the role of the Subobject Classifier Omega and why the internal logic of an elementary topos is intuitionistic rather than classical.",
         ["topos", "subobject classifier", "characteristic morphism", "intuitionistic logic"]),
        ("EX1_47", "Category Theory & Formal Semantics",
         "Explain the Curry-Howard-Lambek Correspondence. How do types, propositions, and categorical objects map to each other?",
         ["curry-howard-lambek", "propositions-as-types", "proofs-as-programs", "cartesian closed"]),
        ("EX1_48", "Category Theory & Formal Semantics",
         "Define a Monoidal Category (C, tensor, I, alpha, lambda, rho). What coherence conditions do the Mac Lane Pentagon and Triangle diagrams guarantee?",
         ["monoidal category", "tensor product", "pentagon diagram", "mac lane coherence"]),
        ("EX1_49", "Category Theory & Formal Semantics",
         "Explain the concept of a Comonad (W, epsilon, delta) in Category Theory and how its co-Kleisli category models contextual or stream computations.",
         ["comonad", "extract epsilon", "duplicate delta", "co-kleisli category"]),
        ("EX1_49b", "Category Theory & Formal Semantics",
         "Explain how Limits and Colimits generalize products and coproducts, and state what it means for a category to be complete and cocomplete.",
         ["limit and colimit", "universal cone", "complete category", "cocomplete"]),
        ("EX1_50", "Category Theory & Formal Semantics",
         "Explain F-Algebras and Initial Algebras in category theory. How do they formalize inductive data types (such as lists and trees) and catamorphisms (folds)?",
         ["f-algebra", "initial algebra", "catamorphism", "inductive data type"]),
    ]

    for p in d1_prompts + d2_prompts + d3_prompts + d4_prompts + d5_prompts:
        if len(questions) < 50:
            questions.append({
                "id": p[0],
                "domain": p[1],
                "question": p[2],
                "rubrics": p[3]
            })
    return questions

# Generate Round 2 (Remedial Exam Pool 1 - Completely Distinct Questions)
def generate_exam_round_2():
    questions = []
    # Distinct questions for Round 2
    items = [
        # D1: Distributed Consensus
        ("EX2_01", "Distributed Consensus", "Analyze how Raft handles non-voting members (Learners) when scaling clusters, and how it prevents network partition stalls during joint consensus membership changes.", ["learner node", "joint consensus", "cluster membership", "two-phase configuration"]),
        ("EX2_02", "Distributed Consensus", "Explain the Paxos Leases mechanism. How do master leases allow consistent linearizable reads without running Phase 2 accepts for read-only queries?", ["paxos lease", "linearizable read", "master lease", "clock synchronization"]),
        ("EX2_03", "Distributed Consensus", "In the Raft protocol, explain why a candidate node that increments its term must explicitly persist currentTerm and votedFor to non-volatile disk before sending RequestVote.", ["non-volatile disk", "votedfor", "currentterm", "split vote prevention"]),
        ("EX2_04", "Distributed Consensus", "Formulate the LWW-Element-Set (Last-Write-Wins Element Set) CRDT. What are its failure modes when distributed physical clocks experience NTP drift?", ["lww-element-set", "wall clock timestamp", "ntp drift", "causality loss"]),
        ("EX2_05", "Distributed Consensus", "Explain the concept of Chain Replication in distributed storage. How does it separate read path at the tail and write path at the head to achieve high throughput?", ["chain replication", "head node", "tail node", "strong consistency"]),
        ("EX2_06", "Distributed Consensus", "Describe the Zab (ZooKeeper Atomic Broadcast) protocol's recovery phase versus broadcast phase. How does it differ from Paxos in maintaining FIFO order?", ["zab protocol", "recovery phase", "zxid identifier", "fifo atomic broadcast"]),
        ("EX2_07", "Distributed Consensus", "Explain the concept of Quorum Leases in distributed caches. How do they balance read latency with cache invalidation guarantees across partitions?", ["quorum lease", "cache invalidation", "lease expiration", "stale read bound"]),
        ("EX2_08", "Distributed Consensus", "Analyze how Byzantine Fault Tolerant protocols like Tendermint consensus utilize 2/3+1 voting rounds (Pre-vote and Pre-commit) to achieve instant finality.", ["tendermint", "2/3 majority", "prevote precommit", "instant finality"]),
        ("EX2_09", "Distributed Consensus", "In distributed systems, what is Chandy-Lamport distributed snapshotting algorithm? How do marker messages record global state without stopping execution?", ["chandy-lamport", "marker message", "channel state", "global snapshot"]),
        ("EX2_10", "Distributed Consensus", "Analyze the Raft log truncation scenario: explain what occurs when an elected leader detects a follower log mismatch at index N and decrements nextIndex.", ["nextindex", "matchindex", "log truncation", "uncommitted entries"]),

        # D2: Zero-Knowledge
        ("EX2_11", "Zero-Knowledge Cryptography", "Explain how the Groth16 zk-SNARK achieves 3-group-element proof size. Why is it non-malleable only under specific simulation-sound variants?", ["groth16", "3 group elements", "trusted setup", "simulation soundness"]),
        ("EX2_12", "Zero-Knowledge Cryptography", "Detail the role of roots of unity in vanishing polynomial Z_H(X) = X^n - 1 over a multiplicative subgroup in polynomial SNARKs.", ["roots of unity", "vanishing polynomial", "multiplicative subgroup", "fft efficiency"]),
        ("EX2_13", "Zero-Knowledge Cryptography", "Explain how multi-scalar multiplication (MSM) dominates prover computation time in elliptic-curve SNARKs, and how Pippenger's algorithm optimizes it.", ["multi-scalar multiplication", "pippenger algorithm", "bucket method", "prover bottleneck"]),
        ("EX2_14", "Zero-Knowledge Cryptography", "Explain the Reed-Solomon proximity testing problem in STARKs and how the FRI (Fast Reed-Solomon Interactive Oracle Proof) protocol solves it.", ["reed-solomon", "fri protocol", "proximity testing", "folding degree"]),
        ("EX2_15", "Zero-Knowledge Cryptography", "Describe the difference between arithmetic circuits over prime fields versus boolean circuits. Why do hashes like Poseidon or Rescue outperform SHA-256 in ZKPs?", ["poseidon hash", "algebraic hash", "sha-256 constraint cost", "prime field"]),
        ("EX2_16", "Zero-Knowledge Cryptography", "Explain how look-up arguments (such as Plookup and Halo2 Lookups) bypass expensive R1CS multiplication gates for bitwise operations and ranges.", ["plookup", "lookup table", "grand product", "bitwise efficiency"]),
        ("EX2_17", "Zero-Knowledge Cryptography", "Detail how the Sumcheck protocol verifies that the sum of a multivariate polynomial over a Boolean hypercube equals a claimed scalar in O(d log N) verifier time.", ["sumcheck protocol", "boolean hypercube", "univariate reduction", "round challenges"]),
        ("EX2_18", "Zero-Knowledge Cryptography", "Explain how Zero-Knowledge Rollups (ZK-Rollups) verify thousands of layer-2 transactions off-chain and commit succinct state roots to layer-1.", ["zk-rollup", "state root", "validity proof", "data availability"]),
        ("EX2_19", "Zero-Knowledge Cryptography", "Explain the concept of DLOG hardness on elliptic curves and how it underpins the binding property of Pedersen commitments: C = g^m h^r.", ["pedersen commitment", "discrete log", "blinding factor", "perfect hiding"]),
        ("EX2_20", "Zero-Knowledge Cryptography", "Analyze the difference between interactive arguments and proofs. Why does statistical soundness differ from computational soundness in SNARKs?", ["computational soundness", "statistical zero knowledge", "negligible error", "prover bound"]),

        # D3: Quantum Information & Stabilizers
        ("EX2_21", "Quantum Stabilizers & Topological Codes", "Explain the 5-qubit quantum code [[5, 1, 3]]. Why is it the smallest possible code capable of protecting a logical qubit against arbitrary single-qubit errors?", ["5-qubit code", "quantum hamming bound", "single-qubit error", "stabilizer generators"]),
        ("EX2_22", "Quantum Stabilizers & Topological Codes", "Describe how X-cut and Z-cut topological boundary conditions in planar surface codes define logical operators X_L and Z_L on boundaries.", ["planar surface code", "rough boundary", "smooth boundary", "logical operator"]),
        ("EX2_23", "Quantum Stabilizers & Topological Codes", "Explain how Lattice Surgery implements fault-tolerant logical CNOT gates between two planar surface code patches using merge and split operations.", ["lattice surgery", "patch merge", "patch split", "logical cnot"]),
        ("EX2_24", "Quantum Stabilizers & Topological Codes", "Detail the role of the Clifford hierarchy P_k in quantum error correction. Why does the T gate sit at level 3, outside the Clifford group at level 2?", ["clifford hierarchy", "t gate level 3", "pauli group level 1", "non-transversal"]),
        ("EX2_25", "Quantum Stabilizers & Topological Codes", "Explain how the Union-Find decoder works as an alternative to MWPM in surface codes, and why its O(N) complexity enables real-time syndrome decoding.", ["union-find decoder", "linear time complexity", "cluster growth", "real-time decoding"]),
        ("EX2_26", "Quantum Stabilizers & Topological Codes", "Describe how leakage errors (where qubits leave the |0>, |1> computational subspace) threaten fault-tolerant codes, and how Leakage Reduction Units (LRUs) reset them.", ["leakage error", "computational subspace", "leakage reduction unit", "ancilla reset"]),
        ("EX2_27", "Quantum Stabilizers & Topological Codes", "Explain the Bacon-Shor subsystem code. How do gauge operators simplify syndrome extraction compared to standard stabilizer codes?", ["bacon-shor code", "subsystem code", "gauge operators", "two-body measurements"]),
        ("EX2_28", "Quantum Stabilizers & Topological Codes", "What is the physical meaning of fault-tolerance threshold ~1% in surface codes under depolarizing noise models?", ["depolarizing noise", "physical threshold", "fault-tolerant scaling", "syndrome measurement noise"]),
        ("EX2_29", "Quantum Stabilizers & Topological Codes", "Explain how quantum teleportation of gates (Gottesman-Chuang trick) enables execution of non-Clifford gates using offline prepared magic states.", ["gate teleportation", "gottesman-chuang", "magic state", "byproduct operator"]),
        ("EX2_30", "Quantum Stabilizers & Topological Codes", "Analyze the difference between geometric locality in 2D surface codes versus high-rate LDPC (Low-Density Parity-Check) quantum codes.", ["quantum ldpc", "geometric locality", "hyperbolic code", "syndrome sparsity"]),

        # D4: Symplectic Dynamics & Neural ODEs
        ("EX2_31", "Symplectic Dynamics & Neural ODEs", "Explain how Hamiltonian Neural Networks parameterize kinetic and potential energy T(p) and V(q) to learn physics from raw trajectory data.", ["hamiltonian mechanics", "kinetic energy", "potential energy", "phase trajectory"]),
        ("EX2_32", "Symplectic Dynamics & Neural ODEs", "Explain how Lagrangian Neural Networks (LNNs) parameterize dynamics using L(q, q_dot) and the Euler-Lagrange equations without requiring canonical momenta.", ["lagrangian neural network", "euler-lagrange equation", "generalized coordinates", "mass matrix inversion"]),
        ("EX2_33", "Symplectic Dynamics & Neural ODEs", "Describe how adaptive step-size ODE solvers (such as Dormand-Prince / RK45) adjust step size Delta t based on local truncation error estimates during forward Neural ODE passes.", ["adaptive step size", "dormand-prince", "local truncation error", "step rejection"]),
        ("EX2_34", "Symplectic Dynamics & Neural ODEs", "Explain how Augmented Neural ODEs resolve the limitation of standard Neural ODEs in learning trajectories with self-intersecting flows.", ["augmented neural ode", "extra dimensions", "flow intersection", "homeomorphism"]),
        ("EX2_35", "Symplectic Dynamics & Neural ODEs", "Explain the concept of Symplectic Matrix group Sp(2n, R) and why any Hamiltonian transformation must satisfy M^T J M = J.", ["symplectic group", "canonical transformation", "phase space volume", "invariance of j"]),
        ("EX2_36", "Symplectic Dynamics & Neural ODEs", "Describe the role of Picard-Lindelof Theorem in ensuring unique solutions to Neural ODE initial value problems: dz/dt = f(z, t).", ["picard-lindel\\u00f6f theorem", "existence and uniqueness", "lipschitz continuity", "initial value problem"]),
        ("EX2_37", "Symplectic Dynamics & Neural ODEs", "Explain how neural vector fields with non-zero curl fail to represent conservative forces in physical systems.", ["conservative force", "curl free vector field", "gradient system", "potential energy"]),
        ("EX2_38", "Symplectic Dynamics & Neural ODEs", "Explain how discrete residual connections z_{k+1} = z_k + f(z_k) approximate continuous Euler discretization of an underlying continuous neural flow.", ["residual connection", "euler discretization", "continuous flow", "step size dt"]),
        ("EX2_39", "Symplectic Dynamics & Neural ODEs", "Analyze why stiff differential equations cause extreme step size contraction in explicit Runge-Kutta solvers and require implicit A-stable methods.", ["stiff equation", "implicit solver", "a-stability", "step size collapse"]),
        ("EX2_40", "Symplectic Dynamics & Neural ODEs", "Explain how spectral normalization restricts the Lipschitz constant of each linear layer W to ||W||_2 <= 1 to enforce globally contractive neural networks.", ["spectral normalization", "singular value", "power iteration", "contractive network"]),

        # D5: Category Theory & Semantics
        ("EX2_41", "Category Theory & Formal Semantics", "Explain how Functoriality preserves identities and composition: F(id_A) = id_{F(A)} and F(g o f) = F(g) o F(f). What happens if a mapping fails compositionality?", ["functoriality", "identity preservation", "composition preservation", "arrow mapping"]),
        ("EX2_42", "Category Theory & Formal Semantics", "Explain the universal property of the Categorical Product A x B. How do projection morphisms pi_1 and pi_2 uniquely factor any pair of morphisms f: C -> A, g: C -> B?", ["categorical product", "projection morphisms", "universal property", "unique mediating morphism"]),
        ("EX2_43", "Category Theory & Formal Semantics", "Explain the universal property of the Categorical Coproduct A + B and its canonical inclusion morphisms i_1 and i_2.", ["categorical coproduct", "inclusion morphisms", "dual product", "sum type"]),
        ("EX2_44", "Category Theory & Formal Semantics", "Explain how the Reader Monad (-> R) and State Monad (s -> (a, s)) are formulated categorically as endofunctors with Kleisli triples.", ["state monad", "reader monad", "kleisli triple", "environment threading"]),
        ("EX2_45", "Category Theory & Formal Semantics", "Describe the difference between an Equivalence of Categories versus an Isomorphism of Categories. What role do natural isomorphisms of functors play?", ["equivalence of categories", "isomorphism of categories", "full faithful essentially surjective", "natural isomorphism"]),
        ("EX2_46", "Category Theory & Formal Semantics", "Explain the concept of a Sheaf on a topological space and how the gluing axiom ensures local consistency leads to unique global sections.", ["sheaf", "gluing axiom", "presheaf", "local sections"]),
        ("EX2_47", "Category Theory & Formal Semantics", "Explain how an Epimorphism is defined categorically without referring to element surjectivity: g o f = h o f implies g = h.", ["epimorphism", "right cancellation", "dual of monomorphism", "surjection"]),
        ("EX2_48", "Category Theory & Formal Semantics", "Explain how a Monomorphism is defined categorically: f o g = f o h implies g = h, and why it models injective structure preservation.", ["monomorphism", "left cancellation", "injection", "subobject"]),
        ("EX2_49", "Category Theory & Formal Semantics", "Explain the adjunction between Free and Forgetful functors (Free -| Forgetful) using the example of Sets and Groups.", ["free functor", "forgetful functor", "free group", "universal property"]),
        ("EX2_50", "Category Theory & Formal Semantics", "Explain how Dependent Type Theory formalizes Pi-types (dependent function types) and Sigma-types (dependent pair types) categorically as local cartesian closed categories.", ["pi-type", "sigma-type", "dependent type theory", "locally cartesian closed"])
    ]
    for item in items:
        questions.append({
            "id": item[0],
            "domain": item[1],
            "question": item[2],
            "rubrics": item[3]
        })
    return questions

# Save all pools
curr = get_master_curriculum()
with open(STUDY_DIR / "frontier_curriculum_master.json", "w", encoding="utf-8") as f:
    json.dump(curr, f, indent=2)
print(f"[+] Master Curriculum saved: {len(curr)} rich lessons.")

ex1 = generate_exam_round_1()
with open(STUDY_DIR / "exam_pool_round1.json", "w", encoding="utf-8") as f:
    json.dump(ex1, f, indent=2)
print(f"[+] Exam Pool Round 1 saved: {len(ex1)} practical questions.")

ex2 = generate_exam_round_2()
with open(STUDY_DIR / "exam_pool_round2.json", "w", encoding="utf-8") as f:
    json.dump(ex2, f, indent=2)
print(f"[+] Exam Pool Round 2 saved: {len(ex2)} practical questions.")
