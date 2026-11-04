"""
Generate Exam Pool Round 3 (50 more distinct challenging practical questions).
"""

import json
from pathlib import Path

STUDY_DIR = Path("frontier_study")

questions = []
# 50 more distinct questions across the 5 domains
items = [
    # D1: Distributed Consensus
    ("EX3_01", "Distributed Consensus", "Explain how Mencius achieves multi-leader high throughput by partitioning the Paxos instance space, and how it skips empty instances using SKIP messages.", ["mencius", "multi-leader", "skip message", "instance partition"]),
    ("EX3_02", "Distributed Consensus", "Describe how Raft's Pre-Vote phase prevents disrupted partitioned nodes from causing leadership instability upon network reconnection.", ["pre-vote phase", "partitioned node", "disruptive election", "term increment"]),
    ("EX3_03", "Distributed Consensus", "In distributed systems, what is the Byzantine Generals Problem under synchronous vs asynchronous networks? State the FLP Impossibility connection.", ["byzantine generals", "synchronous network", "asynchronous network", "flp impossibility"]),
    ("EX3_04", "Distributed Consensus", "Explain how PN-Counter (Positive-Negative Counter) CRDT works by combining two G-Counters (Grow-Only Counters) for increments and decrements.", ["pn-counter", "g-counter", "increment decrement", "state-based crdt"]),
    ("EX3_05", "Distributed Consensus", "Describe how Google Spanner uses TrueTime API with GPS and atomic clocks to achieve external consistency without wide-area lock synchronization.", ["truetime", "gps atomic clock", "uncertainty epsilon", "external consistency"]),
    ("EX3_06", "Distributed Consensus", "Explain how Chubby lock service uses coarse-grained distributed locking with Paxos to prevent split-brain in master-worker architectures.", ["chubby", "coarse-grained lock", "master election", "paxos coordination"]),
    ("EX3_07", "Distributed Consensus", "Explain the Raft log invariance proof: why cannot a leader commit a log entry from an older term by simply counting replicas?", ["older term entry", "indirect commit", "figure 8 scenario", "current term commit"]),
    ("EX3_08", "Distributed Consensus", "Analyze how HotStuff consensus simplifies PBFT view change complexity from O(n^2) to linear O(n) using threshold signatures and a Pacemaker.", ["hotstuff", "linear view change", "threshold signature", "pacemaker"]),
    ("EX3_09", "Distributed Consensus", "Explain how distributed transaction saga pattern maintains eventual data consistency using compensating transactions instead of 2PC locks.", ["saga pattern", "compensating transaction", "forward recovery", "backward recovery"]),
    ("EX3_10", "Distributed Consensus", "Describe how dynamic quorum reconfiguration via vertical Paxos decouples consensus from state machine execution.", ["vertical paxos", "master reconfiguration", "read quorum", "write quorum"]),

    # D2: Zero-Knowledge
    ("EX3_11", "Zero-Knowledge Cryptography", "Explain how Lookup Arguments in Plookup use sorting and grand products to prove values exist in a precomputed table in O(n log n) time.", ["plookup", "sorted multiset", "grand product", "precomputed table"]),
    ("EX3_12", "Zero-Knowledge Cryptography", "Detail the role of elliptic curve endomorphisms and GLV method in accelerating scalar multiplication in cryptographic provers.", ["endomorphism", "glv method", "scalar decomposition", "elliptic curve"]),
    ("EX3_13", "Zero-Knowledge Cryptography", "Explain how zero-knowledge SNARKs achieve zero-knowledge by blinding witness polynomials with random polynomial multiples of the vanishing polynomial.", ["blinding polynomial", "zero-knowledge witness", "vanishing polynomial", "random blinding factor"]),
    ("EX3_14", "Zero-Knowledge Cryptography", "Explain how Nova achieves incrementally verifiable computation (IVC) using relaxed R1CS folding without expensive FFTs or trusted setups.", ["nova", "relaxed r1cs", "folding scheme", "ivc"]),
    ("EX3_15", "Zero-Knowledge Cryptography", "Explain the mechanics of Pedersen Hash functions on Edwards curves and why they are efficient inside SNARK circuits.", ["pedersen hash", "edwards curve", "in-circuit efficiency", "group addition"]),
    ("EX3_16", "Zero-Knowledge Cryptography", "Describe how multivariate polynomial commitments (like Multilinear KZG or Hyrax) avoid FFT interpolation in high-degree circuits.", ["multilinear kzg", "hyrax", "boolean hypercube", "no fft overhead"]),
    ("EX3_17", "Zero-Knowledge Cryptography", "Explain the difference between knowledge soundness and standard soundness in interactive proof systems.", ["knowledge soundness", "extractor", "computational soundness", "witness extraction"]),
    ("EX3_18", "Zero-Knowledge Cryptography", "Explain how recursive SNARK proofs aggregate thousands of transaction proofs into a single constant-sized on-chain validity proof.", ["recursive snark", "proof aggregation", "on-chain verification", "constant proof size"]),
    ("EX3_19", "Zero-Knowledge Cryptography", "Describe the role of quotient polynomial degree in Groth16 and why the SRS degree d must equal the number of multiplication gates m.", ["srs degree", "multiplication gates", "quotient polynomial", "groth16 setup"]),
    ("EX3_20", "Zero-Knowledge Cryptography", "Explain how Plonky2 combines STARK-like FRI commitments with small 64-bit Goldilocks field to achieve sub-second proof generation.", ["plonky2", "goldilocks field", "fri commitment", "64-bit field arithmetic"]),

    # D3: Quantum Information
    ("EX3_21", "Quantum Stabilizers & Topological Codes", "Explain the Color Code on a 2D triangular/hexagonal lattice and why all Clifford gates can be implemented transversally on color codes.", ["color code", "transversal clifford", "2d lattice", "hexagonal tiling"]),
    ("EX3_22", "Quantum Stabilizers & Topological Codes", "Describe how Floquet quantum error-correcting codes (such as the Honeycomb code) dynamically cycle measurements without static stabilizers.", ["floquet code", "honeycomb code", "dynamic measurement cycle", "instantaneous stabilizer"]),
    ("EX3_23", "Quantum Stabilizers & Topological Codes", "Explain how coherent noise (unitary rotation drift) accumulates quadratically in quantum circuits compared to stochastic Pauli noise.", ["coherent noise", "unitary drift", "pauli twirling", "quadratic accumulation"]),
    ("EX3_24", "Quantum Stabilizers & Topological Codes", "Describe how quantum error detection differs from error correction, and how [[4, 2, 2]] detects single-qubit errors without correcting them.", ["[[4, 2, 2]] code", "error detection", "parity check", "subspace projection"]),
    ("EX3_25", "Quantum Stabilizers & Topological Codes", "Explain how catastrophic cosmic ray or phonon burst events cause correlated multi-qubit errors across planar superconducting chips.", ["correlated error", "cosmic ray burst", "phonon poisoning", "superconducting qubit"]),
    ("EX3_26", "Quantum Stabilizers & Topological Codes", "Describe the role of boundary rough edges and smooth edges in encoding logical |0_L> and |+_L> states in surface codes.", ["rough boundary", "smooth boundary", "logical state", "boundary termination"]),
    ("EX3_27", "Quantum Stabilizers & Topological Codes", "Explain how randomized benchmarking (RB) measures average Clifford gate fidelity independently of state preparation and measurement (SPAM) errors.", ["randomized benchmarking", "clifford fidelity", "spam error", "exponential decay"]),
    ("EX3_28", "Quantum Stabilizers & Topological Codes", "Explain how bosonic quantum codes (such as GKP codes - Gottesman-Kitaev-Preskill) protect information within infinite-dimensional continuous oscillator spaces.", ["gkp code", "bosonic code", "harmonic oscillator", "phase space grid"]),
    ("EX3_29", "Quantum Stabilizers & Topological Codes", "Describe how the Knill-Laflamme conditions state the necessary and sufficient algebraic criteria for quantum error correction.", ["knill-laflamme", "error operator", "orthogonal code space", "c_ab identity"]),
    ("EX3_30", "Quantum Stabilizers & Topological Codes", "Explain the concept of quantum LDPC codes with non-zero asymptotic code rate [[n, kn, dn]] overcoming 2D planar geometric limits.", ["quantum ldpc", "asymptotic rate", "non-local connectivity", "fiber bundle code"]),

    # D4: Symplectic Dynamics & Neural ODEs
    ("EX3_31", "Symplectic Dynamics & Neural ODEs", "Explain how symplectic integrators preserve Poincaré invariants in multi-degree-of-freedom Hamiltonian systems.", ["poincar\\u00e9 invariant", "symplectic structure", "canonical momentum", "hamiltonian flow"]),
    ("EX3_32", "Symplectic Dynamics & Neural ODEs", "Explain the role of Monomial Symplectic maps and generating functions in preserving canonical phase space geometry.", ["generating function", "canonical map", "phase space", "hamilton-jacobi"]),
    ("EX3_33", "Symplectic Dynamics & Neural ODEs", "Describe how implicit midpoint rule achieves unconditional symplecticity for non-separable Hamiltonian neural networks.", ["implicit midpoint", "unconditional symplecticity", "non-separable hamiltonian", "phase preservation"]),
    ("EX3_34", "Symplectic Dynamics & Neural ODEs", "Explain how continuous-depth neural networks avoid shattered gradient phenomena that plague ultra-deep feedforward networks.", ["gradient shattering", "continuous depth", "smooth flow", "residual limit"]),
    ("EX3_35", "Symplectic Dynamics & Neural ODEs", "Describe how Neural Jump SDEs combine continuous drift-diffusion with discrete Poisson jump processes for irregular event modelling.", ["neural jump sde", "poisson process", "drift diffusion", "jump intensity"]),
    ("EX3_36", "Symplectic Dynamics & Neural ODEs", "Explain how Riemannian Neural ODEs generalize continuous flow networks to curved manifolds (such as spheres and hyperbolic spaces).", ["riemannian neural ode", "curved manifold", "geodesic flow", "metric tensor"]),
    ("EX3_37", "Symplectic Dynamics & Neural ODEs", "Describe how symplectic shadow Hamiltonians prevent secular drift in long-term orbital predictions of astronomical systems.", ["shadow hamiltonian", "secular drift", "backward error analysis", "long-term stability"]),
    ("EX3_38", "Symplectic Dynamics & Neural ODEs", "Explain the role of Barbalat's Lemma in proving asymptotic stability of continuous-time adaptive neural controllers.", ["barbalat lemma", "asymptotic stability", "uniform continuity", "lyapunov derivative"]),
    ("EX3_39", "Symplectic Dynamics & Neural ODEs", "Explain how Hamiltonian Generative Adversarial Networks (H-GANs) eliminate mode collapse by preserving phase space volume.", ["hamiltonian gan", "mode collapse", "volume preservation", "symplectic discriminator"]),
    ("EX3_40", "Symplectic Dynamics & Neural ODEs", "Analyze the connection between optimal transport, Wasserstein distance, and continuous normalizing flows under Benamou-Brenier formulation.", ["optimal transport", "wasserstein distance", "benamou-brenier", "geodesic in probability space"]),

    # D5: Category Theory & Semantics
    ("EX3_41", "Category Theory & Formal Semantics", "Explain how Slice Categories C/A and Coslice Categories A/C model generalized objects equipped with a morphism to/from a fixed base object.", ["slice category", "coslice category", "base object", "commutative triangle"]),
    ("EX3_42", "Category Theory & Formal Semantics", "Explain the universal property of Equalizers and Coequalizers in category theory and how they formalize equation solutions.", ["equalizer", "coequalizer", "kernel dual", "universal arrow"]),
    ("EX3_43", "Category Theory & Formal Semantics", "Describe how Pullbacks and Pushouts generalize fibred products and fibred coproducts in category theory.", ["pullback", "pushout", "fibred product", "commutative square"]),
    ("EX3_44", "Category Theory & Formal Semantics", "Explain how Kan Extensions (Left Kan Extension Lan_K F and Right Kan Extension Ran_K F) universalize all categorical concepts.", ["kan extension", "lan_k f", "ran_k f", "universal approximation"]),
    ("EX3_45", "Category Theory & Formal Semantics", "Describe the role of Monads in modeling stateful imperative computation in Haskell (IO and State monads).", ["io monad", "state monad", "pure functional programming", "equational reasoning"]),
    ("EX3_46", "Category Theory & Formal Semantics", "Explain how Enriched Category Theory replaces Hom-sets with objects in an arbitrary monoidal category V.", ["enriched category", "hom-object", "v-enriched", "monoidal base"]),
    ("EX3_47", "Category Theory & Formal Semantics", "Describe the Beck-Chevalley condition in categorical logic and how it formalizes substitution commuting with quantifiers.", ["beck-chevalley", "pullback square", "quantifier substitution", "hyperdoctrine"]),
    ("EX3_48", "Category Theory & Formal Semantics", "Explain how Lawvere Theories provide a purely categorical formulation of universal algebra without variable binding.", ["lawvere theory", "cartesian category", "finite product category", "algebraic model"]),
    ("EX3_49", "Category Theory & Formal Semantics", "Explain how Operads generalize categories to multi-input operations (trees and string diagrams).", ["operad", "multi-input", "tree operad", "colored operad"]),
    ("EX3_50", "Category Theory & Formal Semantics", "Explain how Higher Category Theory (infinity-categories and quasicategories) models homotopy theory where morphisms are defined up to coherent higher homotopies.", ["infinity-category", "quasicategory", "homotopy coherence", "higher morphisms"])
]

for item in items:
    questions.append({
        "id": item[0],
        "domain": item[1],
        "question": item[2],
        "rubrics": item[3]
    })

with open(STUDY_DIR / "exam_pool_round3.json", "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=2)
print(f"[+] Exam Pool Round 3 saved: {len(questions)} practical questions.")
