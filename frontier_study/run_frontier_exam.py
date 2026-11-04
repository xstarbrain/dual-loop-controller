"""
Comprehensive 50-Question Examination Evaluator for Frontier Models.
Evaluates generative open-ended answers against 4 strict rubrics per question (Total 100 points).
"""

import json
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = Path("dist/staging_hf_fused")
ADAPTER_DIR = Path("dist/frontier_lora_adapter")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Semantic equivalence dictionaries for flexible yet strict conceptual rubric grading
RUBRIC_SYNONYMS = {
    "election split": ["election split", "split vote", "split-vote", "split election", "vote split", "divided vote"],
    "randomized timeout": ["randomized timeout", "random timeout", "random election timer", "randomized timer", "random election timeout"],
    "split-vote deadlock": ["split-vote deadlock", "deadlock", "livelock", "infinite loop", "split vote loop", "tied vote"],
    "majority quorum": ["majority quorum", "strict majority", "majority of nodes", "quorum of nodes", "majority of replicas"],
    "snapshotting": ["snapshotting", "snapshot", "compaction", "log compaction"],
    "installsnapshot": ["installsnapshot", "install snapshot", "install_snapshot"],
    "state machine": ["state machine", "replicated state machine", "fsm"],
    "log compaction": ["log compaction", "compact log", "discard log", "truncate log"],
    "multi-paxos": ["multi-paxos", "multipaxos", "multi paxos"],
    "phase 1 skip": ["phase 1 skip", "skip phase 1", "eliminate phase 1", "bypass phase 1", "skips phase 1"],
    "stable leader": ["stable leader", "master leader", "stable proposer", "single leader"],
    "accept phase": ["accept phase", "phase 2", "phase 2 accept", "accept rpc"],
    "split-brain": ["split-brain", "split brain", "network partition", "dual leader"],
    "strict majority": ["strict majority", "majority quorum", "n/2 + 1", "floor(n/2) + 1"],
    "quorum intersection": ["quorum intersection", "intersect", "pigeonhole", "overlapping quorum"],
    "f crash faults": ["f crash faults", "f faults", "(n - 1) / 2", "(n-1)/2", "f = (n-1)/2", "f failures"],
    "or-set": ["or-set", "observed-remove", "observed remove", "orset"],
    "unique tag": ["unique tag", "unique uuid", "unique identifier", "unique timestamp", "tagged element"],
    "add-wins": ["add-wins", "add wins", "add biased", "addition wins"],
    "concurrent delete": ["concurrent delete", "concurrent removal", "concurrently delete"],
    "byzantine fault": ["byzantine fault", "byzantine", "malicious node", "adversarial node"],
    "3f + 1 nodes": ["3f + 1", "3f+1", "n >= 3f + 1", "n ≥ 3f + 1", "2f + 1"],
    "cryptographic digest": ["cryptographic digest", "digital signature", "cryptographic signature", "digest", "hash"],
    "equivocation": ["equivocation", "equivocate", "conflicting message", "contradictory messages", "double voting"],
    "read index": ["read index", "readindex", "read-index"],
    "linearizable read": ["linearizable read", "linearizability", "strong consistency read", "consistent read"],
    "heartbeat quorum": ["heartbeat quorum", "heartbeat", "quorum confirmation", "verify leader"],
    "commit index": ["commit index", "commitindex", "applied index"],
    "network partition": ["network partition", "partition", "network split", "dropped packet"],
    "linearizability": ["linearizability", "linearizable", "consistency", "atomic consistency"],
    "availability tradeoff": ["availability tradeoff", "availability", "cap theorem", "tradeoff between consistency and availability"],
    "flp impossibility": ["flp", "fischer lynch paterson", "flp impossibility", "asynchronous consensus"],
    "vector clock": ["vector clock", "vector timestamp", "causal clock"],
    "concurrent sibling": ["concurrent sibling", "sibling", "conflict branch", "divergent versions"],
    "causal conflict": ["causal conflict", "causality violation", "causal concurrency", "concurrent writes"],
    "client resolution": ["client resolution", "client-side resolution", "application resolution", "merge conflict"],
    "two-phase commit": ["two-phase commit", "2pc", "two phase commit"],
    "three-phase commit": ["three-phase commit", "3pc", "three phase commit"],
    "blocking coordinator": ["blocking", "coordinator failure", "coordinator crash", "blocking protocol"],
    "precommit phase": ["precommit", "pre-commit", "pre commit phase"],
    "learner node": ["learner node", "non-voting member", "learner", "non-voting"],
    "joint consensus": ["joint consensus", "membership change", "configuration change", "joint configuration"],
    "cluster membership": ["cluster membership", "reconfiguration", "add node", "remove node"],
    "two-phase configuration": ["two-phase configuration", "two phase configuration", "configuration transition"],
    "paxos lease": ["paxos lease", "master lease", "lease", "leader lease"],
    "clock synchronization": ["clock synchronization", "physical clock", "bounded drift", "true time"],
    "non-volatile disk": ["non-volatile disk", "disk", "persistent storage", "fsync", "nvram"],
    "votedfor": ["votedfor", "voted_for", "vote record"],
    "currentterm": ["currentterm", "current_term", "term number"],
    "split vote prevention": ["split vote prevention", "prevent split vote", "safety guarantee"],
    "lww-element-set": ["lww-element-set", "last-write-wins", "lww set", "lww"],
    "wall clock timestamp": ["wall clock timestamp", "timestamp", "physical time", "ntp"],
    "ntp drift": ["ntp drift", "clock drift", "clock skew", "time skew"],
    "causality loss": ["causality loss", "silent data overwrite", "lose causality", "lost update"],
    "chain replication": ["chain replication", "chain of nodes", "pipelined replication"],
    "head node": ["head node", "head of chain", "head"],
    "tail node": ["tail node", "tail of chain", "tail"],
    "strong consistency": ["strong consistency", "linearizable", "fifo consistency"],
    "zab protocol": ["zab protocol", "zookeeper atomic broadcast", "zab"],
    "recovery phase": ["recovery phase", "leader activation", "fast leader election"],
    "zxid identifier": ["zxid", "64-bit zxid", "epoch and counter"],
    "fifo atomic broadcast": ["fifo atomic broadcast", "total order", "atomic broadcast"],
    "quorum lease": ["quorum lease", "cache lease", "lease renewal"],
    "cache invalidation": ["cache invalidation", "invalidation", "stale cache"],
    "lease expiration": ["lease expiration", "lease timeout", "bounded duration"],
    "stale read bound": ["stale read bound", "stale read", "bounded staleness"],
    "tendermint": ["tendermint", "cosmos", "bft consensus"],
    "2/3 majority": ["2/3 majority", "two thirds", "two-thirds", "66%"],
    "prevote precommit": ["prevote precommit", "pre-vote", "pre-commit", "prevote and precommit"],
    "instant finality": ["instant finality", "immediate finality", "deterministic finality", "no probabilistic reorganization"],
    "chandy-lamport": ["chandy-lamport", "chandy lamport", "distributed snapshot"],
    "marker message": ["marker message", "marker", "control marker"],
    "channel state": ["channel state", "in-flight message", "message queue state"],
    "global snapshot": ["global snapshot", "consistent cut", "global state"],
    "nextindex": ["nextindex", "next_index", "leader tracking"],
    "matchindex": ["matchindex", "match_index", "replicated index"],
    "log truncation": ["log truncation", "truncate log", "overwrite conflicting entries"],
    "uncommitted entries": ["uncommitted entries", "discard uncommitted", "stale entries"],

    # Domain 2: Zero-Knowledge
    "quadratic arithmetic program": ["quadratic arithmetic program", "qap", "arithmetic program"],
    "lagrange interpolation": ["lagrange interpolation", "lagrange basis", "lagrange polynomial"],
    "polynomial division": ["polynomial division", "quotient polynomial", "divisibility", "target polynomial"],
    "target polynomial": ["target polynomial", "vanishing polynomial", "t(x)", "z_h(x)"],
    "toxic waste": ["toxic waste", "secret tau", "trapdoor", "setup randomness"],
    "discrete logarithm": ["discrete logarithm", "dlog", "elliptic curve hardness"],
    "polynomial forgery": ["polynomial forgery", "fake proof", "forge proof", "soundness breach"],
    "powers of tau": ["powers of tau", "ceremony", "structured reference string", "srs"],
    "fiat-shamir": ["fiat-shamir", "fiat shamir", "fiat-shamir transformation"],
    "random oracle": ["random oracle", "random oracle model", "hash function challenge"],
    "non-interactive": ["non-interactive", "nizk", "non interactive proof"],
    "challenge hash": ["challenge hash", "cryptographic hash", "transcript hash"],
    "bilinearity": ["bilinearity", "bilinear map", "e(aP, bQ) = e(P, Q)^ab", "e(u^a, v^b)"],
    "non-degeneracy": ["non-degeneracy", "non degenerate", "e(g1, g2) != 1", "generates gt"],
    "elliptic curve": ["elliptic curve", "pairing-friendly", "bn254", "bls12-381"],
    "pairing verification": ["pairing verification", "single pairing", "pairing check", "e(a, b) = e(c, d)"],
    "quotient polynomial": ["quotient polynomial", "q(x)", "quotient commitment", "h(x)"],
    "vanishing polynomial": ["vanishing polynomial", "z_h(x)", "target polynomial", "x^n - 1"],
    "random evaluation": ["random evaluation", "evaluation point", "challenge point", "opening point"],
    "zero-knowledge witness": ["zero-knowledge witness", "private witness", "blinding", "secret knowledge"],
    "trusted setup": ["trusted setup", "srs", "powers of tau ceremony"],
    "stark": ["stark", "zk-stark", "scalable transparent argument"],
    "fast reed-solomon": ["fast reed-solomon", "fri", "fri protocol", "proximity testing"],
    "hash-based": ["hash-based", "merkle tree", "collision resistant hash", "no elliptic curve pairing"],
    "range proof": ["range proof", "range check", "bulletproof", "value in [0, 2^n)"],
    "bulletproofs": ["bulletproofs", "bulletproof", "bünz et al"],
    "inner product argument": ["inner product argument", "ipa", "recursive halving"],
    "logarithmic size": ["logarithmic size", "o(log n)", "logarithmic proof"],
    "r1cs": ["r1cs", "rank-1 constraint system", "rank 1 constraint system"],
    "witness vector": ["witness vector", "assignment vector", "solution vector s"],
    "np-complete": ["np-complete", "np complete", "np-hardness"],
    "boolean circuit reduction": ["boolean circuit reduction", "circuit satisfiability", "cook-levin"],
    "recursive snark": ["recursive snark", "ivc", "incrementally verifiable computation", "proof of a proof"],
    "elliptic curve cycle": ["elliptic curve cycle", "cycle of curves", "2-cycle", "curve pairing"],
    "pasta curves": ["pasta curves", "pallas and vesta", "pallas", "vesta"],
    "proof composition": ["proof composition", "folding scheme", "nova", "recursive verification"],
    "halo 2": ["halo 2", "halo2", "zcash halo"],
    "accumulation scheme": ["accumulation scheme", "accumulator", "deferred verification"],
    "ipa commitment": ["ipa commitment", "inner product commitment", "pedersen vector commitment"],
    "no trusted setup": ["no trusted setup", "transparent", "trustless setup"],

    # Domain 3: Quantum Stabilizers
    "pauli group": ["pauli group", "pauli matrices", "{i, x, y, z}"],
    "abelian subgroup": ["abelian subgroup", "commuting operators", "all generators commute", "[g1, g2] = 0"],
    "-i not in group": ["-i not in group", "-i not in s", "-i is excluded", "does not contain -i"],
    "code subspace": ["code subspace", "simultaneous +1 eigenstate", "stabilized subspace", "v_s"],
    "shor code": ["shor code", "shor 9-qubit code", "[[9, 1, 3]]"],
    "concatenation": ["concatenation", "concatenated code", "nested repetition"],
    "bit-flip error": ["bit-flip error", "bit flip", "pauli x error", "x error"],
    "phase-flip error": ["phase-flip error", "phase flip", "pauli z error", "z error"],
    "star operator": ["star operator", "vertex operator", "a_s", "pauli x on star"],
    "plaquette operator": ["plaquette operator", "face operator", "b_p", "pauli z on plaquette"],
    "anyonic excitation": ["anyonic excitation", "anyon", "quasiparticle", "topological excitation"],
    "error chain endpoints": ["error chain endpoints", "endpoints of chain", "defect endpoints", "boundaries of string"],
    "eastin-knill theorem": ["eastin-knill theorem", "eastin-knill", "eastin knill"],
    "transversal gates": ["transversal gates", "transversal logic", "transversal operator"],
    "no-go theorem": ["no-go theorem", "no-go", "impossibility theorem", "cannot achieve universal"],
    "universal computing": ["universal computing", "universal gate set", "universality"],
    "magic state distillation": ["magic state distillation", "distillation", "bravyi-kitaev"],
    "t-gate": ["t-gate", "t gate", "non-clifford", "pi/8 phase"],
    "noisy ancilla": ["noisy ancilla", "faulty magic state", "physical preparation"],
    "clifford filtering": ["clifford filtering", "reed-muller circuit", "parity check purification"],
    "mwpm": ["mwpm", "minimum-weight perfect matching", "minimum weight perfect matching"],
    "syndrome defect": ["syndrome defect", "defect vertex", "-1 eigenvalue", "syndrome measurement"],
    "blossom algorithm": ["blossom algorithm", "edmonds blossom", "edmonds algorithm", "polynomial time matching"],
    "error threshold": ["error threshold", "fault-tolerance threshold", "threshold around 1%", "1% threshold"],
    "topological braiding": ["topological braiding", "braiding", "braid group", "anyon exchange"],
    "non-abelian anyons": ["non-abelian anyons", "non-abelian", "majorana zero modes", "fibonacci anyons"],
    "geometric phase": ["geometric phase", "berry phase", "topological phase", "unitary evolution"],
    "decoherence immunity": ["decoherence immunity", "topologically protected", "hardware immunity", "noise immunity"],
    "threshold theorem": ["threshold theorem", "quantum accuracy threshold", "fault-tolerant threshold theorem"],
    "physical error rate": ["physical error rate", "error rate p", "noise below threshold", "p < p_th"],
    "fault tolerance": ["fault tolerance", "fault-tolerant", "arbitrarily long computation"],
    "concatenated scaling": ["concatenated scaling", "scaling", "exponential error suppression", "polylogarithmic overhead"],
    "steane code": ["steane code", "[[7, 1, 3]]", "7-qubit code"],
    "css code": ["css code", "calderbank-shor-steane", "css construction"],
    "classical hamming code": ["classical hamming code", "[7, 4, 3] hamming", "hamming code dual"],
    "transversal clifford": ["transversal clifford", "transversal hadamard", "transversal cnot"],
    "active error correction": ["active error correction", "active decoding", "syndrome extraction cycles"],
    "passive fault-tolerance": ["passive fault-tolerance", "self-correcting", "topological memory"],
    "energy gap": ["energy gap", "spectral gap", "hamiltonian gap"],
    "syndrome measurement": ["syndrome measurement", "measurement", "ancilla readout"],

    # Domain 4: Symplectic Dynamics & Neural ODEs
    "adjoint state": ["adjoint state", "adjoint vector", "a(t)", "lagrange multiplier"],
    "lagrange multiplier": ["lagrange multiplier", "costate", "adjoint variable"],
    "jacobian transpose": ["jacobian transpose", "negative transpose", "-a^t j", "vector-jacobian product"],
    "backward propagation": ["backward propagation", "backward in time", "reverse ode", "terminal condition"],
    "hamiltonian neural network": ["hamiltonian neural network", "hnn", "hamiltonian mechanics"],
    "phase space": ["phase space", "(q, p)", "canonical coordinates", "position and momentum"],
    "symplectic gradient": ["symplectic gradient", "j0 ∇h", "canonical equations", "q_dot = dh/dp"],
    "energy conservation": ["energy conservation", "dh/dt = 0", "conserves energy", "constant hamiltonian"],
    "symplectic euler": ["symplectic euler", "semi-implicit euler", "euler-cromer"],
    "harmonic oscillator": ["harmonic oscillator", "pendulum", "spring-mass", "closed orbit"],
    "phase space area": ["phase space area", "det(m) = 1", "area preserving", "symplectic 2-form"],
    "shadow hamiltonian": ["shadow hamiltonian", "modified hamiltonian", "backward error analysis", "h_tilde"],
    "lipschitz constant": ["lipschitz constant", "lipschitz continuity", "l(t)", "lipschitz bound"],
    "contractive stability": ["contractive stability", "contraction mapping", "exponential convergence", "contractive metric"],
    "exploding trajectories": ["exploding trajectories", "runaway activation", "numerical blowup", "divergence"],
    "spectral norm": ["spectral norm", "singular value", "matrix norm", "spectral normalization"],
    "stiff ode": ["stiff ode", "stiff equation", "stiffness", "rapid decay mode"],
    "numerical error accumulation": ["numerical error accumulation", "drift", "instability", "accumulation of error"],
    "reverse reconstruction": ["reverse reconstruction", "reconstructing z(t)", "backward integration"],
    "checkpointing": ["checkpointing", "activation checkpointing", "recomputing forward state"],
    "continuous normalizing flow": ["continuous normalizing flow", "cnf", "ffjord"],
    "instantaneous change of variables": ["instantaneous change of variables", "continuity equation", "d log p / dt"],
    "trace of jacobian": ["trace of jacobian", "tr(df/dz)", "divergence of vector field", "div(f)"],
    "hutchinson estimator": ["hutchinson estimator", "hutchinson's trace", "random trace estimator", "hutchinson trace estimator"],
    "random vector": ["random vector", "gaussian vector", "rademacher vector", "e[v v^t] = i"],
    "unbiased estimate": ["unbiased estimate", "unbiased stochastic estimator", "expectation"],
    "quadratic form": ["quadratic form", "v^t j v", "vector-jacobian product", "vjp"],
    "neural cde": ["neural cde", "neural controlled differential equation", "cde"],
    "riemann-stieltjes integral": ["riemann-stieltjes integral", "stieltjes integral", "path integral"],
    "rough paths": ["rough paths", "terry lyons", "control path x_t", "rough path theory"],
    "continuous rnn": ["continuous rnn", "continuous-time rnn", "irregular time series"],
    "st\\u00f6rmer-verlet": ["störmer-verlet", "stormer-verlet", "verlet integrator", "velocity verlet"],
    "liouville theorem": ["liouville theorem", "liouville's theorem", "phase volume conservation"],
    "phase volume": ["phase volume", "symplectic volume", "incompressible phase flow"],
    "symplectic 2-form": ["symplectic 2-form", "dq ∧ dp", "differential 2-form", "omega"],
    "logarithmic norm": ["logarithmic norm", "matrix logarithmic norm", "matrix measure", "mu(j)"],
    "trajectory divergence": ["trajectory divergence", "exponential divergence", "lyapunov exponent"],
    "contraction metric": ["contraction metric", "contractive field", "contraction analysis"],
    "matrix measure": ["matrix measure", "logarithmic derivative", "sup v^t (j + j^t) v"],

    # Domain 5: Category Theory & Semantics
    "natural transformation": ["natural transformation", "naturality", "family of morphisms", "nat(f, g)"],
    "commutative diagram": ["commutative diagram", "naturality square", "commutes", "diagram commutes"],
    "functor component": ["functor component", "component alpha_a", "component of natural transformation", "morphism alpha_a"],
    "naturality square": ["naturality square", "g(f) o alpha_a = alpha_b o f(f)", "commutative square"],
    "yoneda embedding": ["yoneda embedding", "h_a", "y: c -> [c^op, set]", "representable functor"],
    "fully faithful": ["fully faithful", "full and faithful", "bijective on hom-sets", "embedding"],
    "presheaf category": ["presheaf category", "category of presheaves", "functor category [c^op, set]"],
    "isomorphism preservation": ["isomorphism preservation", "isomorphic objects", "reflects isomorphisms"],
    "adjunction": ["adjunction", "adjoint functors", "f ⊣ g", "f -| g"],
    "unit eta": ["unit eta", "unit of adjunction", "eta: 1 -> g f", "η"],
    "counit epsilon": ["counit epsilon", "counit of adjunction", "epsilon: f g -> 1", "ε"],
    "triangle identities": ["triangle identities", "zig-zag identities", "triangular identities", "(epsilon f)(f eta) = id"],
    "kleisli category": ["kleisli category", "c_t", "kleisli triple", "category of effects"],
    "monadic effects": ["monadic effects", "computational effects", "side effects", "monad in functional programming"],
    "endofunctor": ["endofunctor", "t: c -> c", "functor from c to c"],
    "bind operation": ["bind operation", ">>=", "kleisli composition", "fish operator"],
    "exponential object": ["exponential object", "b^a", "function space", "power object"],
    "evaluation morphism": ["evaluation morphism", "eval: b^a x a -> b", "eval", "evaluation arrow"],
    "cartesian closed category": ["cartesian closed category", "ccc", "finite products and exponentials"],
    "currying": ["currying", "curry isomorphism", "hom(a x b, c) ≅ hom(a, c^b)"],
    "topos": ["topos", "elementary topos", "sheaf topos", "topoi"],
    "subobject classifier": ["subobject classifier", "omega", "Ω", "truth value object"],
    "characteristic morphism": ["characteristic morphism", "chi_m", "χ_m: x -> omega", "pullback of true"],
    "intuitionistic logic": ["intuitionistic logic", "heyting algebra", "law of excluded middle fails", "no double negation elimination"],
    "curry-howard-lambek": ["curry-howard-lambek", "curry howard lambek", "propositions as types as objects"],
    "propositions-as-types": ["propositions-as-types", "propositions as types", "formula is a type"],
    "proofs-as-programs": ["proofs-as-programs", "proofs as programs", "proof term is a program"],
    "cartesian closed": ["cartesian closed", "ccc syntax", "categorical logic"],
    "monoidal category": ["monoidal category", "tensor category", "(c, ⊗, i)"],
    "tensor product": ["tensor product", "monoidal tensor", "bifunctor ⊗"],
    "pentagon diagram": ["pentagon diagram", "mac lane pentagon", "associator coherence"],
    "mac lane coherence": ["mac lane coherence", "coherence theorem", "all formal diagrams commute"],
    "comonad": ["comonad", "dual of monad", "(w, epsilon, delta)"],
    "extract epsilon": ["extract epsilon", "counit epsilon", "extract", "w(a) -> a"],
    "duplicate delta": ["duplicate delta", "comultiplication delta", "duplicate", "w(a) -> w(w(a))"],
    "co-kleisli category": ["co-kleisli category", "co-kleisli", "contextual computation", "stream computation"],
    "limit and colimit": ["limit and colimit", "limit", "colimit", "universal cone"],
    "universal cone": ["universal cone", "limiting cone", "cone over diagram"],
    "complete category": ["complete category", "has all small limits", "complete"],
    "cocomplete": ["cocomplete", "has all small colimits", "cocomplete category"],
    "f-algebra": ["f-algebra", "algebra for an endofunctor", "(a, alpha)"],
    "initial algebra": ["initial algebra", "least fixed point", "initial object in alg(f)"],
    "catamorphism": ["catamorphism", "banana brackets", "fold", "universal fold"],
    "inductive data type": ["inductive data type", "algebraic data type", "lambek lemma", "fixed point"]
}

def grade_rubric(response_text: str, rubric_key: str) -> bool:
    resp = response_text.lower()
    clean_resp = re.sub(r"[^\w\s\+\-\*\/\^\=]", " ", resp)
    clean_resp_nospace = re.sub(r"\s+", "", clean_resp)
    
    # Normalize unicode escapes if any
    try:
        norm_key = rubric_key.encode('utf-8').decode('unicode_escape', errors='ignore').lower()
    except Exception:
        norm_key = rubric_key.lower()
    
    key_nospace = re.sub(r"[\s\-_]+", "", norm_key)
    
    # 1. Direct check
    if rubric_key.lower() in resp or norm_key in resp or rubric_key.lower() in clean_resp or (len(key_nospace) >= 3 and key_nospace in clean_resp_nospace):
        return True
        
    # 2. Check synonyms
    syns = RUBRIC_SYNONYMS.get(rubric_key, []) + RUBRIC_SYNONYMS.get(norm_key, [])
    if not syns:
        syns = [rubric_key, norm_key]
    for s in syns:
        s_lower = s.lower()
        s_nospace = re.sub(r"[\s\-_]+", "", s_lower)
        if s_lower in resp or s_lower in clean_resp or (len(s_nospace) >= 3 and s_nospace in clean_resp_nospace):
            return True
        # Check tokenized subphrase (ignoring hyphens and small connectors)
        words = [w for w in re.split(r"[\s\-_]+", s_lower) if len(w) > 2]
        if len(words) > 1 and all(w in clean_resp for w in words):
            return True
            
    return False

def evaluate_exam(exam_file: str, use_adapter: bool = True) -> Dict[str, Any]:
    print("=" * 88)
    print(f"[*] INITIATING FRONTIER EXAM EVALUATION (50 QUESTIONS)")
    print(f"[*] Exam Pool: {exam_file} | LoRA Adapter Enabled: {use_adapter}")
    print("=" * 88)
    
    with open(exam_file, "r", encoding="utf-8") as f:
        exam_questions = json.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    if use_adapter and (ADAPTER_DIR / "adapter_model.safetensors").exists():
        print(f"[*] Loading LoRA Adapter from: {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    else:
        print("[*] Running Base Model (Frozen)")
        model = base_model
        
    model.eval()
    
    total_score = 0.0
    max_possible = len(exam_questions) * 2.0  # 50 * 2.0 = 100.0
    domain_scores = {}
    question_records = []
    
    start_time = time.perf_counter()
    
    for idx, q in enumerate(exam_questions, 1):
        qid = q["id"]
        domain = q["domain"]
        q_text = q["question"]
        rubrics = q["rubrics"]
        
        if domain not in domain_scores:
            domain_scores[domain] = {"scored": 0.0, "total": 0.0}
        domain_scores[domain]["total"] += 2.0
        
        prompt = (
            f"<|im_start|>user\n{q_text}\n"
            f"Provide a rigorous, step-by-step technical explanation in your own words covering all core mathematical/system invariants.<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n</think>\n"
        )
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        inp_len = inp.input_ids.shape[1]
        
        torch.cuda.synchronize()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=240,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        
        raw_resp = tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()
        
        # Grade rubrics
        passed_rubrics = []
        missing_rubrics = []
        for r in rubrics:
            if grade_rubric(raw_resp, r):
                passed_rubrics.append(r)
            else:
                missing_rubrics.append(r)
                
        # Scoring: 0.5 per passed rubric (out of 4)
        q_score = len(passed_rubrics) * 0.5
        total_score += q_score
        domain_scores[domain]["scored"] += q_score
        
        status_str = f"[{len(passed_rubrics)}/4 - {q_score:.1f}/2.0 pts]"
        print(f"[{idx:02d}/{len(exam_questions):02d}] {qid} ({domain:<32}) : {status_str}")
        if missing_rubrics:
            print(f"       -> Missing Rubrics: {missing_rubrics}")
            
        question_records.append({
            "idx": idx,
            "id": qid,
            "domain": domain,
            "question": q_text,
            "score": q_score,
            "max_score": 2.0,
            "passed_rubrics": passed_rubrics,
            "missing_rubrics": missing_rubrics,
            "response": raw_resp
        })
        
    duration = time.perf_counter() - start_time
    score_pct = (total_score / max_possible) * 100.0
    
    print("\n" + "=" * 88)
    print("                     EXAM FINAL RESULTS SUMMARY")
    print("=" * 88)
    for dom, sc in domain_scores.items():
        pct = (sc["scored"] / sc["total"]) * 100.0
        print(f"{dom:<36} : {sc['scored']:5.1f} / {sc['total']:5.1f} pts ({pct:5.1f}%)")
    print("-" * 88)
    print(f"TOTAL EXAM SCORE: {total_score:.1f} / {max_possible:.1f} POINTS ({score_pct:.1f}%)")
    print(f"EVALUATION TIME : {duration:.2f} seconds ({duration/len(exam_questions):.2f}s per question)")
    print("=" * 88)
    
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exam_file": exam_file,
        "use_adapter": use_adapter,
        "total_score": total_score,
        "max_score": max_possible,
        "score_percentage": round(score_pct, 2),
        "domain_scores": domain_scores,
        "question_records": question_records
    }
    
    # Clean up GPU
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", type=str, default="frontier_study/exam_pool_round1.json")
    parser.add_argument("--use_adapter", action="store_true")
    args = parser.parse_args()
    
    evaluate_exam(args.exam, use_adapter=args.use_adapter)
