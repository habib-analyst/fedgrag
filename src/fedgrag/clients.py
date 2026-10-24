"""Federated clients: local graphs, lightweight index sketches, and
controlled-disclosure evidence abstracts.

Disclosure rule: a client NEVER sends a raw entity string to the server.
It sends only (a) prototype centroid vectors (floats, no entity strings)
for routing, and (b) evidence abstracts: (TYPE, hashed-id, relation,
TYPE, hashed-id). The server-side evaluator maps abstract ids back to
triple ids ONLY through a local evaluation map that is never transmitted;
the disclosure check scans every transmitted payload string for any raw
entity substring and must find zero.
"""
import hashlib
import numpy as np

from .vectors import vec_from_tokens, cosine
from .corpus import triple_tokens


def _abstract_id(entity: str, client: int) -> str:
    h = hashlib.sha256(f"{client}:{entity}".encode()).hexdigest()[:10]
    topic = entity.split("_ent_")[0]
    return f"{topic}#{h}"


def abstract_of(triple, client: int) -> str:
    return f"{_abstract_id(triple['head'], client)} {triple['rel']} {_abstract_id(triple['tail'], client)}"


class Client:
    def __init__(self, client_id, triples):
        self.id = client_id
        self.triples = triples
        self.vecs = {t["id"]: vec_from_tokens(triple_tokens(t)) for t in triples}
        # Lightweight index: K prototypes per client, clustered by
        # (head-topic, head-entity group of 5) — finer than one-per-topic,
        # so a prototype retains entity-group signal instead of washing it
        # out over a whole topic (pilot lesson: topic-only prototypes gave
        # routed_all 0.225 at B=2; grouped prototypes gave 0.333).
        self.prototypes = []  # (cluster key, vector)
        by_key = {}
        for t in triples:
            ent_n = int(t["head"].split("_ent_")[1])
            by_key.setdefault((t["head_topic"], ent_n // 5), []).append(t["id"])
        for key in sorted(by_key):
            mat = np.stack([self.vecs[i] for i in by_key[key]])
            cen = mat.mean(axis=0)
            n = np.linalg.norm(cen)
            self.prototypes.append((key, cen / n if n > 0 else cen))
        # Mean vector for the direct-matching baseline.
        if self.vecs:
            m = np.stack(list(self.vecs.values())).mean(axis=0)
            n = np.linalg.norm(m)
            self.mean_vec = m / n if n > 0 else m
        else:
            self.mean_vec = np.zeros(256)

    def retrieve(self, qvec, k=5):
        """Return top-k (triple, abstract) by cosine, abstracts only."""
        scored = sorted(
            ((cosine(qvec, v), t) for t, v in
             ((t, self.vecs[t["id"]]) for t in self.triples)),
            key=lambda x: -x[0],
        )
        out = []
        for _, t in scored[:k]:
            out.append((t, abstract_of(t, self.id)))
        return out

    def sketch_payload(self):
        """What the client shares for routing: prototype vectors only.
        Returns list of (client_id, topic, vector) — no entity strings."""
        return [(self.id, topic, vec) for topic, vec in self.prototypes]
