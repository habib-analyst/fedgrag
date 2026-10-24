"""Synthetic distributed graph corpus.

Design (locked in PREREG):
- TOPICS: 12 synthetic domains. Each topic has 40 entities.
- Triples: (head, relation, tail) chains inside/across topics. Each triple is
  assigned to exactly ONE client (no shared global KG anywhere).
- Partition: each topic's triples are owned mainly by 2 home clients, with a
  small spillover to 1 other client, so knowledge is heterogeneous and
  cross-client by construction.
- Queries: 2-hop chains (t1, t2) whose two triples live on DIFFERENT clients
  (cross-client multi-hop, the HotpotQA-like case). Query tokens = anchor
  entity + both relations + intermediate entity + target topic token.
  Gold evidence = the 2 chain triples. Distractor triples share at most one
  query token family, so retrieval is non-vacuous but learnable.
"""
import numpy as np

TOPICS = [
    "cardio", "onco", "neuro", "renal", "pulmo", "endo",
    "immuno", "derma", "gastro", "heme", "ortho", "psych",
]
RELATIONS = ["treats", "causes", "inhibits", "binds", "regulates", "associated_with"]
N_ENTITIES_PER_TOPIC = 40


def build_corpus(n_clients=10, seed=123):
    rng = np.random.default_rng(seed)
    triples = []  # dicts: id, head, rel, tail, head_topic, tail_topic, client
    tid = 0
    for ti, topic in enumerate(TOPICS):
        home = [ti % n_clients, (ti + 3) % n_clients]
        spill = (ti + 6) % n_clients
        for i in range(N_ENTITIES_PER_TOPIC):
            for rel in RELATIONS:
                # chain: ent i -> ent (i+1 mod 40), sometimes cross-topic tail
                if rng.random() < 0.25:
                    t2 = TOPICS[(ti + 1) % len(TOPICS)]
                    tail = f"{t2}_ent_{int(rng.integers(0, N_ENTITIES_PER_TOPIC))}"
                    tail_topic = t2
                else:
                    tail = f"{topic}_ent_{(i + 1) % N_ENTITIES_PER_TOPIC}"
                    tail_topic = topic
                head = f"{topic}_ent_{i}"
                client = home[int(rng.integers(0, 2))] if rng.random() < 0.85 else spill
                triples.append({
                    "id": tid, "head": head, "rel": rel, "tail": tail,
                    "head_topic": topic, "tail_topic": tail_topic,
                    "client": int(client),
                })
                tid += 1
    return triples


def triple_tokens(t):
    return [
        (t["head"], 1.0), (t["tail"], 1.0), (t["rel"], 0.7),
        (t["head_topic"], 0.5), (t["tail_topic"], 0.5),
    ]


def build_queries(triples, n_queries, seed):
    """Cross-client 2-hop queries: find chains t1.tail == t2.head with
    t1.client != t2.client. Query tokens weight the anchor + intermediate
    strongly; the second hop is reachable only via the shared entity."""
    rng = np.random.default_rng(seed)
    by_head = {}
    for t in triples:
        by_head.setdefault(t["head"], []).append(t)
    chains = []
    for t1 in triples:
        for t2 in by_head.get(t1["tail"], []):
            if t1["client"] != t2["client"] and t1["id"] != t2["id"]:
                chains.append((t1, t2))
    idx = rng.choice(len(chains), size=min(n_queries, len(chains)), replace=False)
    queries = []
    for j, ci in enumerate(idx):
        t1, t2 = chains[int(ci)]
        qtokens = [
            (t1["head"], 1.0), (t1["rel"], 0.7), (t1["tail"], 1.0),
            (t2["rel"], 0.7), (t2["tail_topic"], 0.5),
        ]
        queries.append({
            "id": j, "tokens": qtokens, "gold": [t1["id"], t2["id"]],
            "gold_clients": sorted({t1["client"], t2["client"]}),
            "anchor_client": t1["client"],
        })
    return queries
