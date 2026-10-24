"""Routing strategies + retrieval evaluation.

Strategies (all select B clients, then retrieve top-k overall from them):
- oracle:      all triples visible (centralized upper bound; competence check)
- broadcast:   all clients (naive baseline; max cost)
- routing:     FedGRAG-analog: prototype routing graph, T rounds of
               relevance propagation over top-L prototype neighbours,
               client score = max propagated prototype score
- direct:      paper Table 2 baseline: client mean-vector similarity, no graph
- random:      paper Table 2 baseline: seeded random B clients
- local:       single anchor client only (no federation)
"""
import numpy as np

from .vectors import vec_from_tokens, cosine


def build_routing_graph(prototypes, top_l=6):
    """prototypes: list of (client_id, topic, vector). Returns adjacency as
    list of neighbour-index lists (top-L by cosine, excluding self)."""
    n = len(prototypes)
    adj = []
    for i in range(n):
        sims = [(cosine(prototypes[i][2], prototypes[j][2]), j)
                for j in range(n) if j != i]
        sims.sort(reverse=True)
        adj.append([j for _, j in sims[:top_l]])
    return adj


def route_clients(prototypes, adj, qvec, n_clients, top_b=2, rounds=2, alpha=0.7,
                  score_mode="max"):
    scores = np.array([cosine(qvec, p[2]) for p in prototypes])
    s = scores.copy()
    for _ in range(rounds):
        prop = np.array([
            np.mean([s[j] for j in adj[i]]) if adj[i] else 0.0
            for i in range(len(prototypes))
        ])
        s = alpha * scores + (1 - alpha) * prop
    per_client = {}
    for (cid, _, _), sc in zip(prototypes, s):
        per_client.setdefault(cid, []).append(float(sc))
    client_score = {}
    for cid, vals in per_client.items():
        vals.sort(reverse=True)
        if score_mode == "top2sum":
            client_score[cid] = sum(vals[:2])
        else:
            client_score[cid] = vals[0]
    ranked = sorted(client_score.items(), key=lambda x: -x[1])
    return [cid for cid, _ in ranked[:top_b]]


def direct_clients(clients, qvec, top_b=2):
    ranked = sorted(clients, key=lambda c: -cosine(qvec, c.mean_vec))
    return [c.id for c in ranked[:top_b]]


def retrieve_from(clients_by_id, selected, qvec, k=5):
    """Merge-rank retrieved triples across selected clients. Returns
    (retrieved triple ids, payload strings, n_clients_contacted)."""
    scored = []
    payloads = []
    for cid in selected:
        for t, ab in clients_by_id[cid].retrieve(qvec, k=k):
            scored.append((cosine(qvec, clients_by_id[cid].vecs[t["id"]]), t["id"]))
            payloads.append(ab)
    scored.sort(key=lambda x: -x[0])
    return [tid for _, tid in scored[:k]], payloads, len(selected)


def evaluate(queries, clients_by_id, prototypes, adj, strategy, seed=0, k=5, top_b=2,
             route_kwargs=None):
    """Per-query: recall@k, hit (all gold retrieved), routed (selected cover
    all gold clients), clients contacted, payload token cost."""
    rng = np.random.default_rng(seed)
    rows = []
    all_ids = sorted(clients_by_id)
    for q in queries:
        qvec = vec_from_tokens(q["tokens"])
        if strategy == "oracle":
            selected = all_ids
            # oracle retrieves from the pooled graph: emulate via all clients
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        elif strategy == "broadcast":
            selected = all_ids
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        elif strategy == "routing":
            selected = route_clients(prototypes, adj, qvec, len(all_ids), top_b=top_b,
                                     **(route_kwargs or {}))
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        elif strategy == "direct":
            selected = direct_clients(list(clients_by_id.values()), qvec, top_b=top_b)
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        elif strategy == "random":
            selected = sorted(rng.choice(all_ids, size=top_b, replace=False).tolist())
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        elif strategy == "local":
            selected = [q["anchor_client"]]
            retrieved, payloads, ncon = retrieve_from(clients_by_id, selected, qvec, k=k)
        else:
            raise ValueError(strategy)
        gold = set(q["gold"])
        rec = len(gold & set(retrieved)) / len(gold)
        rows.append({
            "qid": q["id"], "recall": rec,
            "hit_all": 1.0 if gold <= set(retrieved) else 0.0,
            "hit_any": 1.0 if gold & set(retrieved) else 0.0,
            "routed_all": 1.0 if set(q["gold_clients"]) <= set(selected) else 0.0,
            "clients": ncon,
            "tokens": sum(len(p.split()) for p in payloads),
            "selected": selected, "payloads": payloads,
        })
    return rows
