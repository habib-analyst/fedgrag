# FedGRAG Open Replication: Federated Graph Retrieval Routing Without a Shared Graph

**Paper replicated:** arXiv 2610.30728, "Federated Graph Retrieval-Augmented
Generation" (Xiaobao Wang, Renda Han, Zhiliang Xia, Meng Liu, Renxiang Guan,
Di Jin), published as an ICLR 2027 conference paper.

**Replication status: PASS — all preregistered gates passed.**
Repo staged for `habib-analyst/fedgrag` (repo creation + push by parent;
this build made no pushes and no repo creation).

## The paper's claim, from its tables

GraphRAG today assumes one central knowledge graph. In practice knowledge is
split across organisations that cannot share their graphs. FedGRAG proposes
two mechanisms: (1) **federated knowledge routing** — find the relevant
clients through lightweight knowledge indices, without a global graph; and
(2) **controlled-disclosure evidence construction** — retrieved subgraphs
are converted into evidence abstracts so raw entities never leave a client.

Paper numbers (10 clients, HotpotQA): FedGRAG R@2 19.4 / R@5 31.7 vs best
federated baseline FedMosaic 17.1 / 27.3; routing Hit@2 66.7 vs Direct
Matching 42.4 and Random Routing 31.6; communication ~50 KB per client vs
FRAG's 213 KB at 1 client. The paper's experiments used NVIDIA 4090D GPUs
and a shared LLM for generation.

## What this replication does (and does not do)

This is a **routing/retrieval replication only, CPU-only, $0, no LLM**.
Retrieval scoring uses deterministic hashed token vectors — a scripted
policy, not a model. The paper's downstream generation results (EM/F1) are
explicitly out of scope: the routing mechanism is the replicable core, and
that is what was built, gated, and measured. The corpus is fully synthetic
(12 topics, 2,880 triples, 10 clients); no gated datasets were used.

Method, in one paragraph: each client keeps its local graph private and
shares only K prototype centroids (clustered by head-topic × head-entity
group of 5; 288 prototypes total) plus, on retrieval, evidence abstracts of
the form `(topic#hash, relation, topic#hash)`. A query is routed to the top
B = 3 clients by max prototype cosine, and the top 5 retrieved triples
across those clients are scored against the 2 gold chain triples of each
cross-client 2-hop query. Baselines: centralized oracle, naive broadcast,
direct mean-vector matching, random routing, and local-only answering.

## Preregistration and gates

`PREREG.md` was locked before the determinative run (bars set from a
120-query pilot, seed 7, then frozen; the determinative run used 400 fresh
queries, seed 42, same corpus seed 123). Bootstrap: 10,000 paired resamples,
seed 20261024, percentile 95% CI.

| Gate | Bar (locked) | Measured (seed 42, n = 400) | Verdict |
|---|---|---|---|
| G0 staging | ≥12 pytest green + determinism | 16/16 green; rerun byte-identical | PASS |
| G1 author-code recheck | Any author repo → hard stop | None found (authenticated GitHub, 2026-10-24; abs page 0 links; repo search FedGRAG/fedgrag = 0; code search 18 hits all arXiv-listing/metadata, none author code) | PASS |
| Competence pre-check | Oracle Recall@5 ≥ 0.60 | 0.7075 [0.6837, 0.7312] | PASS |
| G2 disease: routing vs local-only | Paired diff ≥ 0.03, CI lower > 0 | **+0.0638, CI [0.0388, 0.0888]** | PASS |
| G3 load-bearing: routing vs broadcast, cost-adjusted | Paired diff ≥ 0.05, CI lower > 0 | **+0.1172, CI [0.1093, 0.1250]** | PASS |
| Disclosure (hard) | 0 raw entity strings transmitted | 0 leaks across all strategies | PASS |

Full determinative numbers (Recall@5, mean [95% CI]):

- Oracle 0.7075 [0.6837, 0.7312] — hit-all 0.415
- Broadcast 0.7075 [0.6837, 0.7312] — 10 clients contacted
- **Routing 0.5637 [0.5387, 0.5887]** — hit-all 0.200, routed-all 0.660, 3 clients
- Direct matching 0.4637 [0.4375, 0.4900] — routed-all 0.435
- Random 0.2800 [0.2525, 0.3075]
- Local-only 0.5000 exactly (structural: anchor triple always, second hop never)

Reading: federated routing recovers ~80% of broadcast's recall while
contacting 30% of the clients, and beats both direct matching (+0.100) and
local-only answering (+0.064). The disease is real in this setup: answering
alone caps at the anchor hop; finding the second client is where federation
pays, and prototype routing finds it 66% of the time.

## Two honest negatives

1. **Propagation hurt (pilot ablation).** The paper's graph-propagation step
   over the prototype routing graph diluted the entity-group signal in the
   pilot: routed-all 0.550 with propagation vs 0.625 without, recall 0.529 vs
   0.575 (120 pilot queries). The shipped router therefore uses direct
   prototype matching with zero propagation rounds. This is a property of
   this synthetic setup — coarse hashed vectors, small corpus — not a
   refutation of the paper's design on real embeddings and real graphs.
   It is reported here rather than hidden, and the propagation code path
   remains in the repo for re-testing.
2. **Prototype granularity matters more than the routing rule.** Topic-only
   prototypes routed only 22.5% of queries to all gold clients at B = 2;
   (topic × entity-group) prototypes reached 62.5–66% at B = 3. A federated
   index that is too coarse fails silently: it returns *plausible* clients
   (same topic) that do not hold the evidence. Index resolution is a
   first-order design choice, not a detail.

## Limitations

- Synthetic corpus and hashed vectors: effect sizes demonstrate the
  mechanism, they are not the paper's effect sizes and do not transfer to
  production graphs without re-measurement.
- No generation step: nothing here says anything about answer quality
  (EM/F1) — only about whether the right evidence reaches the server.
- The disclosure check is a string-level guarantee (no raw entity text in
  payloads). It is not a formal privacy proof: prototype centroids and
  hashed IDs can still leak distributional information in principle.
- Single corpus seed and single query seed for the determinative run;
  multi-seed robustness was not gated and is future work.

## Reproduce

```bash
pip install -e .
python examples/run_experiment.py   # seed 42, 400 queries
python -m pytest tests/ -q           # 16 tests
```

Compute: CPU only, minutes per run, $0. No training, no API calls, no
downloads (numpy + pytest only).
