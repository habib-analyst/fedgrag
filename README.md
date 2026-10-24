# fedgrag — Federated Graph Retrieval Routing, Open Replication

An open, CPU-only, deterministic replication of the **routing / retrieval core**
of "Federated Graph Retrieval-Augmented Generation" (FedGRAG, arXiv 2610.30728,
ICLR 2027 paper) — built from the paper's tables, not from author code
(no author code was found; see `PAPER.md` for the G1 recheck record).

## What it does

Real knowledge graphs live in separate organisations that cannot share them.
FedGRAG's idea: route a query to the right clients using **lightweight
indices** (prototype sketches, no raw graph shared), then retrieve
**controlled-disclosure evidence abstracts** (hashed entity IDs, relations
kept) instead of raw triples.

This repo implements that mechanism on a synthetic distributed graph corpus
and measures it against five baselines:

| Method | Recall@5 (400 queries) | Clients contacted |
|---|---|---|
| Centralized oracle (upper bound) | 0.7075 | 10 |
| Naive broadcast-to-all | 0.7075 | 10 |
| **Federated prototype routing (this work)** | **0.5637** | **3** |
| Direct mean-vector matching | 0.4637 | 3 |
| Random routing | 0.2800 | 3 |
| Local-only (anchor client) | 0.5000 | 1 |

Gated results (preregistered before the determinative run, 10,000-sample
paired bootstrap, 95% CI):

- **Disease gate:** routing beats local-only by **+0.0638** Recall@5,
  CI [0.0388, 0.0888] — federation adds retrieval quality over answering alone.
- **Load-bearing gate:** routing beats broadcast on cost-adjusted quality
  (Recall@5 per client contacted) by **+0.1172**, CI [0.1093, 0.1250] —
  ~80% of broadcast's recall at 30% of its client cost.
- **Disclosure gate:** **0** raw entity strings in any transmitted payload
  across all strategies (verified by substring scan).

## Honest scope and limitations (read before citing)

- **No LLM is used anywhere.** Retrieval scoring uses deterministic hashed
  token vectors (a scripted policy, not a model). The paper's downstream
  generation results (EM/F1 with an LLM on HotpotQA/MuSiQue/2Wiki) are
  **not** replicated — this is a routing/retrieval replication only.
- **The corpus is synthetic** (12 topics, 2,880 triples, 10 clients, 400
  2-hop cross-client queries). Effect sizes here are not the paper's
  effect sizes; they demonstrate the mechanism, not production behaviour.
- **Propagation ablation (negative result):** the paper's graph-propagation
  step over the prototype routing graph *hurt* in the pilot (routed-client
  coverage 0.550 with propagation vs 0.625 without; recall 0.529 vs 0.575).
  The shipped router therefore uses direct prototype matching with no
  propagation rounds. This is a property of this synthetic setup, not a
  refutation of the paper's design on real data.
- Local-only recall is exactly 0.5000 by construction (the anchor client
  always holds the anchor triple and never the second hop) — the disease
  gate is routing's ability to find the *second* client, not to beat a
  strong local solver.

## Quick start (CPU only, $0, no downloads)

```bash
pip install -e .
python examples/run_experiment.py   # determinative run, seed 42
python -m pytest tests/ -q           # 16 tests
```

## Layout

- `src/fedgrag/vectors.py` — deterministic hashed token vectors
- `src/fedgrag/corpus.py` — synthetic distributed graph + cross-client queries
- `src/fedgrag/clients.py` — local graphs, prototype sketches, evidence abstracts
- `src/fedgrag/routing.py` — routing strategies + retrieval evaluation
- `src/fedgrag/stats.py` — paired bootstrap CIs
- `examples/run_experiment.py` — the gated determinative run
- `PAPER.md` — full writeup with preregistration, gates, and numbers

## License

MIT. This is an independent open replication; it is not affiliated with
the paper's authors.
