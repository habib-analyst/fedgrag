"""FedGRAG replication tests (CPU, deterministic, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from fedgrag.vectors import vec_from_tokens, cosine, DIM
from fedgrag.corpus import build_corpus, build_queries, TOPICS
from fedgrag.clients import Client, abstract_of
from fedgrag.routing import build_routing_graph, route_clients, evaluate
from fedgrag.stats import paired_diff_ci, mean_ci


@pytest.fixture(scope="module")
def setup():
    triples = build_corpus(n_clients=10, seed=123)
    clients = {cid: Client(cid, [t for t in triples if t["client"] == cid]) for cid in range(10)}
    protos = []
    for cid in range(10):
        protos.extend(clients[cid].sketch_payload())
    adj = build_routing_graph(protos, top_l=6)
    queries = build_queries(triples, 40, seed=7)
    return triples, clients, protos, adj, queries


def test_vector_dim_and_norm():
    v = vec_from_tokens([("a", 1.0), ("b", 0.5)])
    assert v.shape == (DIM,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_vector_deterministic():
    a = vec_from_tokens([("x", 1.0), ("y", 1.0)])
    b = vec_from_tokens([("x", 1.0), ("y", 1.0)])
    assert np.array_equal(a, b)


def test_cosine_self_is_one():
    v = vec_from_tokens([("x", 1.0)])
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_corpus_size_and_topics():
    triples = build_corpus(n_clients=10, seed=123)
    assert len(triples) == 2880
    assert set(t["head_topic"] for t in triples) == set(TOPICS)


def test_corpus_no_shared_triples_across_clients(setup):
    triples, _, _, _, _ = setup
    ids = [t["id"] for t in triples]
    assert len(ids) == len(set(ids))
    # each triple assigned to exactly one client
    assert all(isinstance(t["client"], int) for t in triples)


def test_queries_are_cross_client(setup):
    triples, _, _, _, queries = setup
    assert len(queries) > 0
    for q in queries:
        assert len(q["gold_clients"]) == 2
        assert len(q["gold"]) == 2


def test_abstract_hides_raw_entities(setup):
    triples, clients, _, _, _ = setup
    t = triples[0]
    ab = abstract_of(t, t["client"])
    assert t["head"] not in ab and t["tail"] not in ab
    assert t["rel"] in ab  # relation is shared; entities are not


def test_client_retrieve_returns_abstracts_only(setup):
    _, clients, _, _, queries = setup
    from fedgrag.vectors import vec_from_tokens as vft
    qvec = vft(queries[0]["tokens"])
    for t, ab in clients[0].retrieve(qvec, k=3):
        assert t["head"] not in ab and t["tail"] not in ab


def test_routing_selects_b_clients(setup):
    _, clients, protos, adj, queries = setup
    from fedgrag.vectors import vec_from_tokens as vft
    qvec = vft(queries[0]["tokens"])
    sel = route_clients(protos, adj, qvec, 10, top_b=3, rounds=0)
    assert len(sel) == 3 and len(set(sel)) == 3
    assert all(0 <= c < 10 for c in sel)


def test_routing_graph_adjacency_size(setup):
    _, _, protos, adj, _ = setup
    assert len(adj) == len(protos)
    assert all(len(nb) <= 6 for nb in adj)


def test_evaluate_rows_have_required_fields(setup):
    _, clients, protos, adj, queries = setup
    rows = evaluate(queries[:5], clients, protos, adj, "routing", seed=7, top_b=3,
                    route_kwargs=dict(rounds=0, score_mode="max"))
    assert len(rows) == 5
    for r in rows:
        assert 0.0 <= r["recall"] <= 1.0
        assert r["clients"] == 3


def test_oracle_beats_random_on_pilot_queries(setup):
    _, clients, protos, adj, queries = setup
    o = evaluate(queries, clients, protos, adj, "oracle", seed=7)
    r = evaluate(queries, clients, protos, adj, "random", seed=7, top_b=3)
    assert np.mean([x["recall"] for x in o]) > np.mean([x["recall"] for x in r])


def test_local_recall_structurally_half(setup):
    _, clients, protos, adj, queries = setup
    rows = evaluate(queries, clients, protos, adj, "local", seed=7)
    # anchor client always holds the anchor triple, never the second hop
    assert all(r["recall"] == 0.5 for r in rows)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=50)
    b = rng.normal(0.4, 0.1, size=50)
    d, lo, hi = paired_diff_ci(a, b, n_boot=1000)
    assert lo <= d <= hi


def test_mean_ci_sane():
    m, lo, hi = mean_ci([0.5] * 20, n_boot=1000)
    assert m == 0.5 and lo == 0.5 and hi == 0.5


def test_query_vectors_deterministic(setup):
    _, _, _, _, queries = setup
    from fedgrag.vectors import vec_from_tokens as vft
    assert np.array_equal(vft(queries[0]["tokens"]), vft(queries[0]["tokens"]))
