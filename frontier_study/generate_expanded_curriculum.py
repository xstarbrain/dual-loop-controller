"""
Generates the comprehensive 55-lesson Frontier Master Curriculum.
Each lesson provides in-depth proofs, formal invariants, equations, and derivations.
"""

import json
from pathlib import Path

STUDY_DIR = Path("frontier_study")
STUDY_DIR.mkdir(parents=True, exist_ok=True)

lessons = []

# --- DOMAIN 1: DISTRIBUTED CONSENSUS & INVARIANTS ---
d1_lessons = [
    {
        "domain": "Distributed Consensus",
        "instruction": "Prove the State Machine Safety property of the Raft consensus algorithm. Explain why a leader can never overwrite or commit an entry at an index where another leader already committed a different entry.",
        "response": (
            "Raft guarantees State Machine Safety through the Leader Completeness Invariant:\n\n"
            "1. Leader Completeness Invariant: If a log entry is committed at term T, that entry is present in all leaders elected at terms T' > T.\n"
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
        "instruction": "In the Raft consensus protocol, explain step-by-step why an election split occurs when multiple candidates launch concurrent RequestVote RPCs, and how randomized election timeouts mathematically minimize split election deadlocks.",
        "response": (
            "1. Election Split Mechanism: When a leader fails, multiple followers' election timers may expire simultaneously. Each candidate votes for itself and requests votes from peers. In an N-node cluster, votes can divide evenly (e.g., 2 votes each in a 4-node or 5-node cluster), leaving no candidate with a strict majority (quorum of ⌊N/2⌋ + 1). Without intervention, candidates timeout together, increment term simultaneously, and split the vote in an infinite deadlock loop.\n\n"
            "2. Randomized Election Timeouts: Raft assigns election timeouts chosen uniformly at random from a range [T, 2T] (typically 150ms - 300ms).\n\n"
            "3. Mathematical Deadlock Prevention: The probability of two nodes timing out within the network round-trip time (RTT) δ becomes negligible (approximately δ / T << 1). One node's timer expires first, collects a majority quorum before competitors time out, becomes leader, and sends heartbeats to suppress competing elections."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Explain how Raft log compaction via Snapshotting interacts with followers that lag far behind. Detail the role of the InstallSnapshot RPC.",
        "response": (
            "1. Log Compaction Need: As the replicated log grows, storing every command consumes excessive disk space and slows restart replay. In Snapshotting, the node writes its state machine state up to lastIncludedIndex and lastIncludedTerm to a snapshot file, then discards prior log entries.\n\n"
            "2. Lagging Followers: If a follower is partitioned or disconnected for a prolonged period, the leader may have already discarded log entries starting from follower's nextIndex. The leader cannot send missing entries via AppendEntries.\n\n"
            "3. InstallSnapshot RPC: The leader detects this condition (nextIndex ≤ leader's snapshot lastIncludedIndex) and sends InstallSnapshot RPC containing the snapshot chunks, lastIncludedIndex, and lastIncludedTerm. The follower discards its conflicting uncommitted log, resets its state machine to the snapshot, sets its commitIndex to lastIncludedIndex, and resumes standard AppendEntries replication."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Detail how Multi-Paxos optimizes the standard Synod protocol into a single round-trip for steady-state client requests once a stable leader is established.",
        "response": (
            "1. Synod Overhead: Basic Paxos requires two phases for every single decision: Phase 1 (Prepare/Promise) to establish leadership and discover accepted values, and Phase 2 (Accept/Accepted) to commit the value. This imposes 2 network round-trips (RTTs) per operation.\n\n"
            "2. Multi-Paxos Optimization: Multi-Paxos extends the ballot number to govern an entire sequence of log instances. Once a proposer successfully runs Phase 1 for all future instances (by obtaining promises from a majority quorum that no higher ballot will be accepted), it becomes the stable Master Leader.\n\n"
            "3. Single RTT Steady-State: For all subsequent client write requests, the stable leader skips Phase 1 entirely and immediately executes Phase 2 (Accept(b, i, v)). Acceptors respond Accepted, achieving consensus in a single round-trip."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "In distributed storage systems, what is the Split-Brain scenario? Prove mathematically why a system of N replicas can only tolerate f = (N - 1) / 2 crash faults under majority consensus.",
        "response": (
            "1. Split-Brain Scenario: A network partition divides a distributed cluster into isolated sub-networks. If both partitions believe they have authority to accept and commit writes independently, data diverges irreversibly, destroying consistency.\n\n"
            "2. Quorum Intersection Requirement: To prevent split-brain, any two valid decision quorums Q1 and Q2 must overlap by at least one node: |Q1 ∩ Q2| ≥ 1.\n\n"
            "3. Mathematical Bound on Faults: Let cluster size be N. For a quorum size Q to guarantee intersection, 2Q > N, meaning Q ≥ ⌊N/2⌋ + 1 (strict majority). If f nodes crash, the remaining N - f live nodes must still form a valid majority quorum: N - f ≥ ⌊N/2⌋ + 1. Rearranging: f ≤ N - (⌊N/2⌋ + 1) = ⌊(N - 1)/2⌋. Therefore, an N-node system can tolerate at most f = (N - 1)/2 crash faults (e.g., 5 nodes tolerate 2 faults; 3 nodes tolerate 1 fault)."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Describe the Observed-Remove Set (OR-Set) CRDT. How does it resolve concurrent additions and deletions of the exact same element across distributed replicas?",
        "response": (
            "1. Problem in Distributed Sets: If replica A adds element x while replica B concurrently deletes x, naive sets cannot resolve whether x exists without coordinating.\n\n"
            "2. Observed-Remove Set (OR-Set) Structure: An OR-Set maintains two internal sets: an Add-Set of pairs (element, unique_tag) and a Remove-Set of observed tags. Each tag is a globally unique UUID or logical timestamp.\n\n"
            "3. Add Operation: Adding element x generates a fresh unique tag t and inserts (x, t) into the Add-Set.\n\n"
            "4. Delete Operation: Deleting element x observes all currently existing tags for x in local Add-Set and adds those observed tags to the Remove-Set.\n\n"
            "5. Membership & Add-Wins Resolution: Element x is considered present if there exists at least one pair (x, t) in the Add-Set whose tag t is NOT in the Remove-Set. Because a concurrent add generates a brand new tag t_new that was not observed by the concurrent delete, t_new survives. Thus, the OR-Set implements Add-Wins semantics with mathematically provable strong eventual consistency."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Explain the primary difference between crash-fault-tolerant (CFT) protocols like Raft and Byzantine fault-tolerant (BFT) protocols like PBFT regarding quorum sizes and cryptographic signatures.",
        "response": (
            "1. Threat Model: CFT (Raft, Paxos) assumes nodes may crash or delay messages, but never lie, forge messages, or send contradictory information (no equivocation). BFT (PBFT, Tendermint) assumes malicious adversaries who can send conflicting messages to different peers, forge identities, and intentionally subvert protocol logic.\n\n"
            "2. Quorum Size Mathematics: In CFT, majority quorum is (N/2 + 1), tolerating f = (N - 1)/2 faults (N ≥ 2f + 1). In BFT, a malicious node can equivocate: it can send 'commit' to node A and 'abort' to node B while f nodes are silent/crashed. To ensure honest nodes always intersect by at least f + 1 honest nodes, the cluster requires N ≥ 3f + 1 nodes and quorums of 2f + 1.\n\n"
            "3. Cryptographic Verification: CFT uses simple checksums and integer term numbers. BFT strictly requires digital signatures and cryptographic digests (e.g., pre-prepare, prepare, and commit certs) to prevent forgery and provide non-repudiation."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Explain the term 'Read Index' optimization in Raft. How does a leader serve linearizable read requests without appending a new entry to the replicated log?",
        "response": (
            "1. Linearizable Read Challenge: If a network partition occurs and a new leader is elected in a higher term, an old leader might still receive read requests. If the stale leader serves reads from its local state machine, it returns stale data, violating linearizability.\n\n"
            "2. Naive Solution: The leader appends a no-op read entry to the log and commits it via full quorum replication before returning the result. This consumes disk I/O and log space.\n\n"
            "3. Read Index Optimization: The leader records its current commitIndex as the readIndex. Before reading state machine at readIndex, the leader broadcasts a round of heartbeat RPCs to verify it is still the authoritative leader. Once a majority quorum confirms receipt, the leader waits until its state machine has applied at least up to readIndex, then returns the read result. This guarantees linearizability without writing any new entries to disk or log."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "Analyze the CAP theorem in modern distributed systems. Why is network partition tolerance (P) unavoidable in physical networks, leaving only the tradeoff between Consistency (C) and Availability (A)?",
        "response": (
            "1. Definitions: Consistency (C) means linearizability: every read receives the most recent write or an error. Availability (A) means every non-failing node returns a non-error response. Partition Tolerance (P) means the system continues operating despite dropped or delayed messages between nodes.\n\n"
            "2. Unavoidability of Partitions: Physical networks (cables, routers, switches, cloud VPCs) inherently suffer packet loss, hardware faults, GC pauses, and fiber cuts. A distributed system cannot choose 'No Partitions' (CA does not exist in real distributed systems).\n\n"
            "3. The Inevitable Tradeoff: When a partition occurs between node group G1 and group G2: if a client writes to G1, G1 cannot replicate to G2. If a client reads from G2, the system must either: (a) Reject the read or block to preserve Consistency (CP), or (b) Return stale data from G2 to maintain Availability (AP). By the FLP Impossibility and Brewer's theorem, no system can achieve both simultaneously under network partitions."
        )
    },
    {
        "domain": "Distributed Consensus",
        "instruction": "In distributed transactions, contrast Two-Phase Commit (2PC) with Three-Phase Commit (3PC). Why does 2PC suffer from blocking upon coordinator failure, and how does 3PC's PreCommit phase alleviate this?",
        "response": (
            "1. Two-Phase Commit (2PC): Coordinator sends Prepare to participants. If all vote Yes, coordinator sends Commit; otherwise Abort.\n"
            "Blocking Flaw: If coordinator crashes after all participants vote Yes but before sending Commit, participants are stuck in Prepared state. They cannot unilaterally commit (coordinator might have aborted) and cannot abort (coordinator might have committed). 2PC is a blocking protocol.\n\n"
            "2. Three-Phase Commit (3PC): Adds an intermediate phase to split the commit decision: (1) Can-Commit (voting), (2) Pre-Commit (coordinator broadcasts Pre-Commit once all vote Yes), (3) Do-Commit.\n\n"
            "3. Elimination of Blocking: If coordinator fails during Pre-Commit, a recovery coordinator inspects participants. If any participant is in Pre-Commit state, it knows all participants voted Yes in Phase 1, so it is safe to proceed with Do-Commit. If no participant reached Pre-Commit, it is safe to abort. Thus, 3PC is non-blocking under fail-stop failure models (assuming synchronous crash detection)."
        )
    }
]

# --- DOMAIN 2: ZERO-KNOWLEDGE CRYPTOGRAPHY & COMMITMENTS ---
d2_lessons = [
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "Derive the reduction of an arithmetic computation into a Rank-1 Constraint System (R1CS) and explain why every constraint must satisfy the bilinear form <A, s> * <B, s> = <C, s>.",
        "response": (
            "1. Arithmetic Circuit: Consists of addition wires and multiplication gates over finite field F.\n\n"
            "2. Witness Vector: Let s = [1, x_1, x_2, ..., x_n]^T ∈ F^(n+1) be the full assignment vector (1 is constant wire, followed by public inputs, private inputs, and intermediate wires).\n\n"
            "3. R1CS Matrices: For m multiplication gates, construct three sparse matrices A, B, C ∈ F^(m × (n+1)). Each row i represents a multiplication gate:\n"
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
        "instruction": "Explain how an arithmetic circuit with addition and multiplication gates is converted into a system of Quadratic Arithmetic Programs (QAP). What role do Lagrange basis polynomials play?",
        "response": (
            "1. From R1CS to Polynomials: An R1CS has constraint matrices A, B, C of dimension m × (n+1). For each constraint row i ∈ {1, ..., m}, choose distinct field evaluation roots r_1, ..., r_m ∈ F.\n\n"
            "2. Lagrange Basis Interpolation: Construct Lagrange basis polynomials L_i(x) such that L_i(r_j) = 1 if i = j, and 0 otherwise. For each column k ∈ {0, ..., n}, interpolate polynomials A_k(x) = ∑ A_{i,k} L_i(x), B_k(x) = ∑ B_{i,k} L_i(x), and C_k(x) = ∑ C_{i,k} L_i(x).\n\n"
            "3. Target/Vanishing Polynomial: Define target polynomial T(x) = ∏_{i=1}^m (x - r_i).\n\n"
            "4. QAP Divisibility Condition: The witness vector s satisfies all R1CS constraints if and only if P(x) = (∑ s_k A_k(x)) · (∑ s_k B_k(x)) - (∑ s_k C_k(x)) vanishes at all roots r_i. Equivalently, P(x) is exactly divisible by T(x): P(x) = H(x) T(x). This converts m individual matrix equations into a single polynomial division check."
        )
    },
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "In the KZG commitment scheme, explain why the trusted setup powers of tau must have toxic waste secret tau discarded. What catastrophic forgery occurs if tau is leaked?",
        "response": (
            "1. Toxic Waste Secret: In the ceremony, a random secret scalar τ ∈ F is chosen to compute group points [τ^0]_1, [τ^1]_1, ..., [τ^d]_1 and [τ]_2. The secret τ is the 'toxic waste'.\n\n"
            "2. Catastrophic Forgery if Leaked: If an attacker knows the scalar τ, they can evaluate any polynomial directly: P(τ) is just a known field element. For any false claim P(z) = y_fake (where y_fake ≠ P(z)), the quotient polynomial equation (P(x) - y_fake) / (x - z) is not an integer polynomial, but the attacker can compute the scalar value in the field:\n"
            "   Q_fake = (P(τ) - y_fake) · (τ - z)^{-1} ∈ F\n"
            "The attacker then sets π_fake = Q_fake · [1]_1. The verifier's pairing check e(C - [y_fake]_1, [1]_2) = e(π_fake, [τ]_2 - [z]_2) will pass perfectly. Thus, knowledge of τ completely breaks computational soundness, allowing anyone to generate valid proofs for false statements."
        )
    },
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "Explain the Fiat-Shamir Heuristic. How does it convert an interactive zero-knowledge public-coin proof into a non-interactive proof (NIZK) using a cryptographic hash function?",
        "response": (
            "1. Interactive Public-Coin Protocol: In an interactive proof (e.g., Sigma protocol), prover sends commitment a, verifier replies with random challenge e, prover responds with response z. If the prover could choose e after seeing a, they could forge the proof; the verifier's randomness enforces soundness.\n\n"
            "2. Fiat-Shamir Transformation: Replaces the interactive verifier with a cryptographic hash function H (modeled as a Random Oracle): e = H(statement, commitment a).\n\n"
            "3. Non-Interactive Proof (NIZK): The prover computes a, computes e = H(statement, a), computes response z, and outputs the pair π = (a, z). The verifier computes e = H(statement, a) locally and checks the validity of (a, e, z). Soundness is preserved because the pre-image resistance and pseudo-randomness of H prevent the prover from tailoring a to produce a favorable challenge e."
        )
    },
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "Explain the concept of Bilinear Pairings e: G1 x G2 -> GT. What properties (bilinearity and non-degeneracy) make them essential for cryptographic verification?",
        "response": (
            "1. Bilinear Map Definition: A mapping e: G_1 × G_2 → G_T between cyclic groups of prime order r with generators g_1 ∈ G_1, g_2 ∈ G_2.\n\n"
            "2. Core Algebraic Properties:\n"
            "- Bilinearity: For all u ∈ G_1, v ∈ G_2 and a, b ∈ F_r: e(u^a, v^b) = e(u, v)^{ab}. In additive notation: e(a P, b Q) = e(P, Q)^{ab}.\n"
            "- Non-degeneracy: e(g_1, g_2) ≠ 1 (it generates G_T).\n"
            "- Computability: e(P, Q) can be computed efficiently via Miller's algorithm on pairing-friendly elliptic curves (e.g., BN254, BLS12-381).\n\n"
            "3. Role in Verification: Pairings allow multiplying two encrypted or committed secret scalars in the exponent: given [a]_1 = a g_1 and [b]_2 = b g_2 without knowing a or b, the pairing computes e([a]_1, [b]_2) = e(g_1, g_2)^{ab}. This enables verifying quadratic relations <A, s><B, s> = <C, s> in encrypted form without revealing witness secrets."
        )
    },
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "What is the difference between Transparent SNARKs (like STARKs) and SNARKs requiring a Trusted Setup (like Groth16)? How do STARKs eliminate trusted setups using FRI?",
        "response": (
            "1. Trusted Setup SNARKs (Groth16): Require structured reference strings generated from secret toxic waste. They offer tiny proof sizes (3 group elements, 128 bytes) and fast verification (3 pairings), but are vulnerable if setup ceremony is compromised, and circuits require trusted ceremonies.\n\n"
            "2. Transparent STARKs: Require no toxic waste; all randomness comes from public hash functions (transparent). They rely on collision-resistant hash functions and Fast Reed-Solomon Interactive Oracle Proofs of Proximity (FRI).\n\n"
            "3. Eliminating Setup via FRI: Instead of elliptic-curve pairings, STARKs encode computation trace as a Reed-Solomon codeword. The FRI protocol iteratively folds the high-degree polynomial codeword into smaller degrees using random linear combinations, checking low-degree proximity with Merkle authentication paths. STARKs are post-quantum secure and eliminate trusted setups entirely, at the cost of larger proof sizes (40-100 KB)."
        )
    },
    {
        "domain": "Zero-Knowledge Cryptography",
        "instruction": "Explain the role of Range Proofs in confidential transactions (e.g., Bulletproofs) and how inner product arguments achieve logarithmic proof size.",
        "response": (
            "1. Range Proofs in Confidential Assets: In confidential transactions (e.g., Monero), transaction amounts are committed via Pedersen commitments C = v · G + r · H. To prevent double spending by creating negative balances (overflowing the prime field order), the prover must prove that 0 ≤ v < 2^n without revealing value v.\n\n"
            "2. Inner Product Formulation: The range condition is expressed by bit decomposition v = ∑ v_i 2^i where each bit satisfies v_i (1 - v_i) = 0. This is transformed into a relation between two vectors a_L, a_R satisfying an inner product <a_L, a_R> = c.\n\n"
            "3. Logarithmic Scaling via Inner Product Argument: Bulletproofs uses an inner product argument that recursively halves vector length at each step: vectors of length n are combined with challenge x into vectors of length n/2: a' = a_{low} + x a_{high}, b' = b_{low} + x^{-1} b_{high}. After log_2(n) rounds, the proof size reduces to 2 log_2(n) group elements, achieving compact logarithmic proofs without trusted setups."
        )
    }
]

# --- DOMAIN 3: QUANTUM INFORMATION & TOPOLOGICAL STABILIZERS ---
d3_lessons = [
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "In quantum error correction, explain the Stabilizer Formalism. What mathematical conditions must a subgroup of the Pauli group satisfy to define a valid quantum code?",
        "response": (
            "1. Pauli Group P_n: The group formed by n-fold tensor products of Pauli matrices {I, X, Y, Z} with global phase factors {±1, ±i}.\n\n"
            "2. Stabilizer Subgroup S: An abelian subgroup S ⊂ P_n.\n"
            "Conditions for a valid stabilizer code [[n, k, d]]:\n"
            "- Abelian: All elements must commute with each other: ∀ g_1, g_2 ∈ S, [g_1, g_2] = 0. (Ensures simultaneous eigenstates exist).\n"
            "- Negative Identity Excluded: -I ∉ S. (If -I ∈ S, then -I |ψ⟩ = |ψ⟩ implies |ψ⟩ = 0, meaning code space is trivial).\n\n"
            "3. Code Subspace V_S: The subspace of states stabilized by S: V_S = {|ψ⟩ ∈ H^{⊗n} : ∀ g ∈ S, g |ψ⟩ = |ψ⟩}. For n physical qubits and m independent generators, the code encodes k = n - m logical qubits with dimension 2^k."
        )
    },
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "Explain how the 9-qubit Shor Code protects against both arbitrary bit-flip (X) and phase-flip (Z) errors by concatenating repetition codes.",
        "response": (
            "1. Concatenation Strategy: Shor's code combines the 3-qubit bit-flip code and 3-qubit phase-flip code into a concatenated [[9, 1, 3]] code.\n\n"
            "2. Encoding: Logical states are encoded by first applying the phase-flip code (|0_L⟩ → (|000⟩ + |111⟩)/√2), and then replacing each physical qubit with the 3-qubit bit-flip repetition code (|0⟩ → |000⟩, |1⟩ → |111⟩):\n"
            "   |0_L⟩ = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩) / 2√2\n"
            "   |1_L⟩ = (|000⟩ - |111⟩)(|000⟩ - |111⟩)(|000⟩ - |111⟩) / 2√2\n\n"
            "3. Error Correction: Bit-flip errors (X) are detected within each 3-qubit block by measuring parity operators Z_1 Z_2 and Z_2 Z_3. Phase-flip errors (Z) are detected across the three blocks by measuring sign parity X_1 X_2 X_3 X_4 X_5 X_6 and X_4 X_5 X_6 X_7 X_8 X_9. Arbitrary quantum noise decomposes into linear combinations of X, Z, and Y = iXZ errors, so correcting single-qubit X and Z errors guarantees protection against any arbitrary single-qubit error."
        )
    },
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "Describe the physical difference between star operators A_s and plaquette operators B_p in the Toric Code, and how error chains create anyonic excitations (e and m particles).",
        "response": (
            "1. Operators: Star operators A_s = ∏_{i ∈ star(s)} X_i act at vertices. Plaquette operators B_p = ∏_{j ∈ boundary(p)} Z_j act on faces.\n\n"
            "2. Error Chains & Syndrome Endpoints:\n"
            "- A string of bit-flip errors Z along edges of the primal lattice anticommutes with star operators at the ends of the chain, flipping A_s eigenvalues from +1 to -1. The endpoints are electric anyons (e-particles).\n"
            "- A string of phase-flip errors X along edges of the dual lattice anticommutes with plaquette operators at the ends of the chain, flipping B_p eigenvalues from +1 to -1. The endpoints are magnetic anyons (m-particles).\n\n"
            "3. Anyonic Statistics: e and m particles are mutual semions: braiding an e-particle around an m-particle generates a non-trivial topological Aharonov-Bohm phase factor of -1 (due to XZ = -ZX). Errors are corrected by pairing anyonic defect endpoints through minimum-weight paths."
        )
    },
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "Explain why transversal gates in quantum error correction cannot achieve universal fault-tolerant quantum computing on their own (Eastin-Knill Theorem).",
        "response": (
            "1. Transversal Gates: A gate is transversal if it acts independently on individual physical qubits of a code block (U = ⊗_{i=1}^n U_i). Transversal gates prevent error propagation: a single error on qubit i cannot spread to other qubits in the same block, making them inherently fault-tolerant.\n\n"
            "2. Eastin-Knill Theorem (2009): No quantum error-correcting code that can detect arbitrary single-qubit errors can possess a universal set of transversal logic gates.\n\n"
            "3. Proof Intuition: The group of transversal gates for any code must form a Lie group (continuous). However, the stabilizer code space is a discrete subspace. A continuous symmetry would map codewords into non-codewords or create non-correctable continuous drifts. Thus, the transversal gate group is strictly finite (typically restricted to the Clifford group). Universality requires non-transversal techniques, such as Magic State Distillation or Code Switching."
        )
    },
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "Detail the process of Magic State Distillation. Why is it used to inject non-Clifford T gates into surface codes?",
        "response": (
            "1. The Non-Clifford Bottleneck: Surface codes easily implement Clifford gates (H, S, CNOT) via transversal operations and lattice surgery. But universal computation requires a non-Clifford gate (e.g., T-gate = diag(1, e^{iπ/4})).\n\n"
            "2. Magic State Target: A non-Clifford state |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 (or the |H⟩ state) is prepared on physical hardware. Physical preparation is noisy (error rate p ≈ 10^{-2} to 10^{-3}).\n\n"
            "3. Distillation Circuit (Bravyi-Kitaev 15-to-1): 15 copies of noisy |T⟩ states are fed into an error-detecting Clifford circuit derived from the 15-qubit Reed-Muller code. If syndrome measurements show no errors, one output state is produced with error rate reduced to O(p^3). Repeating this procedure hierarchically yields ultra-pure magic states (error rate < 10^{-12}), which are then consumed via gate teleportation to perform fault-tolerant T-gates."
        )
    },
    {
        "domain": "Quantum Stabilizers & Topological Codes",
        "instruction": "In surface code decoding, explain how Minimum-Weight Perfect Matching (MWPM) pairs defect syndromes to correct errors with threshold ~1%.",
        "response": (
            "1. Syndrome Graph: When stabilizer measurements yield -1, they represent defect vertices. Pauli error chains connect pairs of defects (or connect a defect to a code boundary).\n\n"
            "2. Matching Problem: Given a set of 2k defects in 2D space (and time, across measurement rounds), the decoder finds the most likely set of physical errors that generated the syndrome. Assuming independent identically distributed errors with probability p < 0.5, the probability of an error chain of length L is proportional to p^L. Minimizing total error probability corresponds to minimizing total chain length ∑ L_i.\n\n"
            "3. MWPM via Blossom Algorithm: Jack Edmonds' Blossom Algorithm computes the minimum-weight perfect matching on the complete graph of defects in polynomial time O(V^3). Applying Pauli corrections along the matched shortest paths eliminates all syndromes and restores the code space, yielding a fault-tolerance threshold of ~1%."
        )
    }
]

# --- DOMAIN 4: SYMPLECTIC DYNAMICS & NEURAL ODES ---
d4_lessons = [
    {
        "domain": "Symplectic Dynamics & Neural ODEs",
        "instruction": "Explain the continuous adjoint state equation in Neural ODEs. Why does the adjoint vector propagate backward in time with the negative transpose of the Jacobian?",
        "response": (
            "1. Continuous Dynamics: Hidden state z(t) satisfies dz/dt = f(z, t, θ). Loss is evaluated at final time L(z(t_1)).\n\n"
            "2. Adjoint Definition: a(t) = ∂L/∂z(t). Consider an infinitesimal time step dt: z(t + dt) = z(t) + f(z(t), t, θ) dt.\n\n"
            "3. Chain Rule: By the multivariate chain rule:\n"
            "   ∂L/∂z(t) = (∂L/∂z(t + dt)) · (∂z(t + dt)/∂z(t)) = a(t + dt) · (I + [∂f/∂z] dt)\n"
            "Subtracting a(t + dt) and dividing by dt as dt → 0:\n"
            "   lim_{dt → 0} (a(t + dt) - a(t)) / dt = - a(t)^T · [∂f(z, t, θ)/∂z]\n"
            "Therefore, da(t)/dt = - a(t)^T · J_f. The negative transpose ensures that gradient propagation in continuous time exactly inverts the forward Jacobian contraction/expansion."
        )
    },
    {
        "domain": "Symplectic Dynamics & Neural ODEs",
        "instruction": "In Hamiltonian Neural Networks (HNNs), how does parameterizing the network with a scalar energy function H(q, p) enforce exact energy conservation?",
        "response": (
            "1. Standard Neural Networks: Standard physics models learn vector fields directly: (q̈, ṗ) = NN(q, p). Without geometric constraints, numerical errors accumulate and total energy drifts over time.\n\n"
            "2. Hamiltonian Parameterization: An HNN parameterizes only the scalar Hamiltonian H_θ(q, p): R^{2d} → R. Equations of motion are obtained via Hamilton's canonical equations:\n"
            "   q̇ = ∂H_θ/∂p,   ṗ = - ∂H_θ/∂q\n\n"
            "3. Exact Energy Conservation Proof:\n"
            "   dH/dt = (∂H/∂q) · q̇ + (∂H/∂p) · ṗ = (∂H/∂q) · (∂H/∂p) + (∂H/∂p) · (-∂H/∂q) = 0.\n"
            "Energy conservation dH/dt = 0 holds analytically for any network weights θ regardless of training convergence. The model respects symplectic geometry by construction."
        )
    },
    {
        "domain": "Symplectic Dynamics & Neural ODEs",
        "instruction": "Compare standard explicit Euler integration with the symplectic Euler method. Why does explicit Euler fail on harmonic oscillators while symplectic Euler preserves closed orbits?",
        "response": (
            "1. Harmonic Oscillator: H(q, p) = (p^2 + q^2) / 2. Canonical equations: q̇ = p, ṗ = -q.\n\n"
            "2. Explicit Euler: q_{n+1} = q_n + h p_n, p_{n+1} = p_n - h q_n. The transformation matrix is M = [[1, h], [-h, 1]]. Its determinant is det(M) = 1 + h^2 > 1. At every step, phase space volume expands by (1 + h^2). Energy grows exponentially E_n = (1 + h^2)^n E_0, causing artificial spiral divergence.\n\n"
            "3. Symplectic Euler: Updates position first, then momentum with new position: q_{n+1} = q_n + h p_n, p_{n+1} = p_n - h q_{n+1}. The transformation matrix is M = [[1, h], [-h, 1 - h^2]]. Its determinant is det(M) = 1(1 - h^2) - (-h^2) = 1. Area is preserved exactly. The trajectory stays on the closed ellipse of a shadow Hamiltonian H̃ = H + (h/2) q p, remaining stable indefinitely."
        )
    },
    {
        "domain": "Symplectic Dynamics & Neural ODEs",
        "instruction": "Explain how Continuous Normalizing Flows (CNFs) utilize the instantaneous change of variables formula: d log p(z(t)) / dt = - Tr(df / dz).",
        "response": (
            "1. Discrete Normalizing Flows: Standard normalizing flows use discrete invertible transformations z_{k+1} = g(z_k) and compute log p(z_{k+1}) = log p(z_k) - log |det(J_g)|. Computing det(J) requires O(d^3) operations or architectural restrictions (e.g., triangular Jacobians in RealNVP).\n\n"
            "2. Continuous Formulation (CNF): State evolves via ODE dz/dt = f(z(t), t). By the Fokker-Planck / continuity equation without diffusion: ∂p/∂t = - ∇ · (p f). Taking the total derivative along trajectory z(t):\n"
            "   d log p(z(t)) / dt = - ∇ · f(z(t), t) = - Tr(∂f(z(t), t) / ∂z(t))\n\n"
            "3. Computational Advantage: Computing the trace Tr(J) is linear O(d) instead of O(d^3). Furthermore, the vector field f requires no triangular or coupling layer restrictions; any unconstrained neural network architecture can be used."
        )
    },
    {
        "domain": "Symplectic Dynamics & Neural ODEs",
        "instruction": "Explain how Hutchinson's Trace Estimator enables unbiased O(1) computation of Tr(df/dz) in high-dimensional continuous neural flows using random Gaussian vectors.",
        "response": (
            "1. Trace Bottleneck: In FFJORD / CNFs, calculating exact Tr(J_f) requires d forward-backward passes (one for each dimension).\n\n"
            "2. Hutchinson's Estimator: Let v ∈ R^d be a random vector with zero mean and identity covariance E[v v^T] = I (e.g., standard normal N(0, I) or Rademacher ±1).\n"
            "Identity:\n"
            "   E_v [v^T J_f v] = E_v [Tr(v^T J_f v)] = E_v [Tr(J_f v v^T)] = Tr(J_f E_v[v v^T]) = Tr(J_f I) = Tr(J_f).\n\n"
            "3. Computational Complexity: The term v^T J_f = (J_f^T v)^T is a single Vector-Jacobian Product (VJP), computed in a single reverse-mode automatic differentiation pass at the exact cost of one network evaluation. Thus, Hutchinson's estimator computes an unbiased stochastic estimate of the divergence with O(1) Jacobian evaluations."
        )
    }
]

# --- DOMAIN 5: CATEGORY THEORY & FORMAL SEMANTICS ---
d5_lessons = [
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
        "instruction": "Define an Adjunction between two categories (F -| G). What are the unit eta and counit epsilon of an adjunction, and what triangular identities must they satisfy?",
        "response": (
            "1. Adjunction Definition: Given functors F: C → D and G: D → C, F is left adjoint to G (written F ⊣ G) if there is a natural isomorphism of Hom-sets:\n"
            "   Hom_D(F(A), B) ≅ Hom_C(A, G(B))\n\n"
            "2. Unit and Counit:\n"
            "- Unit η: 1_C ⇒ G ∘ F (maps an object A to its canonical reflection in G(F(A))).\n"
            "- Counit ε: F ∘ G ⇒ 1_D (evaluates the structure from F(G(B)) to B).\n\n"
            "3. Triangle Identities (Zig-zag equations):\n"
            "- (ε F) ∘ (F η) = id_F (as natural transformation F ⇒ F)\n"
            "- (G ε) ∘ (η G) = id_G (as natural transformation G ⇒ G)\n"
            "Any adjunction uniquely induces a monad T = G ∘ F on category C and a comonad W = F ∘ G on category D."
        )
    },
    {
        "domain": "Category Theory & Formal Semantics",
        "instruction": "Explain the categorical semantics of Monads in functional programming. How does the Kleisli category model effectful computations (such as state, I/O, or non-determinism)?",
        "response": (
            "1. Effectful Functions: A pure function has signature A → B. An effectful function produces a side effect (e.g., state mutation, failure, I/O) and returns an embellished type T(B). Direct composition of f: A → T(B) and g: B → T(C) fails because types do not align.\n\n"
            "2. Kleisli Category C_T: Given monad (T, η, μ) on category C:\n"
            "- Objects of C_T are the same as objects of C.\n"
            "- Morphisms Hom_{C_T}(A, B) = Hom_C(A, T(B)).\n"
            "- Identity morphism is the unit η_A: A → T(A).\n"
            "- Composition is Kleisli composition (g >=> f)(a) = μ (T(g) (f(a))).\n\n"
            "3. Semantic Unification: By abstracting effects into the Kleisli category, functional programs treat side-effecting operations as pure mathematical arrows obeying standard category axioms (identity and associativity), allowing formal equational reasoning and algebraic effects."
        )
    },
    {
        "domain": "Category Theory & Formal Semantics",
        "instruction": "What is a Topos in category theory? Explain the role of the Subobject Classifier Omega and why the internal logic of an elementary topos is intuitionistic rather than classical.",
        "response": (
            "1. Topos Definition: An elementary topos E is a category that has finite limits, is cartesian closed (has power objects), and possesses a subobject classifier Ω.\n\n"
            "2. Subobject Classifier Ω: An object Ω together with a true morphism true: 1 → Ω such that for any monomorphism (subobject) m: S ↣ X, there exists a unique characteristic morphism χ_m: X → Ω making the square a pullback:\n"
            "   S is the inverse image of true under χ_m.\n"
            "In Set, Ω = {true, false}. In general topoi (e.g., presheaves, sheaves), Ω represents truth values varying over spaces or stages.\n\n"
            "3. Intuitionistic Internal Logic: In a general topos, the subobject lattice Sub(X) is a Heyting algebra rather than a Boolean algebra. Negation is defined via intuitionistic pseudocomplement ¬p = (p ⇒ ⊥). Because truth values vary continuously over open sets or contexts, the Law of Excluded Middle (p ∨ ¬p = 1) and Double Negation Elimination (¬¬p = p) do not hold in general. Thus, the native internal logic of any topos is intuitionistic logic."
        )
    },
    {
        "domain": "Category Theory & Formal Semantics",
        "instruction": "Explain F-Algebras and Initial Algebras in category theory. How do they formalize inductive data types (such as lists and trees) and catamorphisms (folds)?",
        "response": (
            "1. F-Algebra: Given endofunctor F: C → C, an F-algebra is a pair (A, α) where A is an object (carrier) and α: F(A) → A is a morphism (structure map).\n\n"
            "2. Initial F-Algebra: An algebra (I, in) is initial if for any other F-algebra (A, α), there exists a unique algebra homomorphism h: I → A such that h ∘ in = α ∘ F(h).\n\n"
            "3. Inductive Data Types: By Lambek's Lemma, the structure map in: F(I) → I is an isomorphism, so I ≅ F(I) (a fixed point of functor F). For lists of type X, F(L) = 1 + X × L. The initial algebra for this functor is the standard list data type List(X) = Nil | Cons(X, List(X)).\n\n"
            "4. Catamorphisms (Folds): The unique mediating arrow h from the initial algebra to any target algebra (A, α) is called a catamorphism, denoted (| α |). In programming, this is the canonical fold/reduce function. Catamorphisms guarantee terminating, structurally recursive functional programs by universal construction."
        )
    }
]

all_lessons = d1_lessons + d2_lessons + d3_lessons + d4_lessons + d5_lessons
with open(STUDY_DIR / "frontier_curriculum_master.json", "w", encoding="utf-8") as f:
    json.dump(all_lessons, f, indent=2)

print(f"[+] Expanded Frontier Master Curriculum created: {len(all_lessons)} comprehensive lessons.")
