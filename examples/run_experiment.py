"""Determinative gated run (seed 42, 400 queries). Evaluated ONCE vs PREREG."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedgrag.corpus import build_corpus, build_queries
from fedgrag.clients import Client
from fedgrag.routing import build_routing_graph, evaluate
from fedgrag.stats import paired_diff_ci, mean_ci

OUT = Path(__file__).resolve().parents[1] / "results"


def run():
    triples = build_corpus(n_clients=10, seed=123)
    clients = {cid: Client(cid, [t for t in triples if t["client"] == cid]) for cid in range(10)}
    protos = []
    for cid in range(10):
        protos.extend(clients[cid].sketch_payload())
    adj = build_routing_graph(protos, top_l=6)
    queries = build_queries(triples, 400, seed=42)
    RK = dict(rounds=0, score_mode="max")
    res = {}
    per_strat_rows = {}
    for strat in ["oracle", "broadcast", "routing", "direct", "random", "local"]:
        kw = RK if strat == "routing" else None
        rows = evaluate(queries, clients, protos, adj, strat, seed=42, top_b=3, route_kwargs=kw)
        per_strat_rows[strat] = rows
        rec = [r["recall"] for r in rows]
        m, lo, hi = mean_ci(rec)
        res[strat] = {"recall": m, "lo": lo, "hi": hi,
                      "hit_all": sum(r["hit_all"] for r in rows) / len(rows),
                      "hit_any": sum(r["hit_any"] for r in rows) / len(rows),
                      "routed_all": sum(r["routed_all"] for r in rows) / len(rows),
                      "mean_clients": sum(r["clients"] for r in rows) / len(rows),
                      "mean_tokens": sum(r["tokens"] for r in rows) / len(rows),
                      "rec_per_client": sum(r["recall"] / r["clients"] for r in rows) / len(rows)}
    # paired gates
    r = [x["recall"] for x in per_strat_rows["routing"]]
    l = [x["recall"] for x in per_strat_rows["local"]]
    g2 = paired_diff_ci(r, l)
    rq = [x["recall"] / x["clients"] for x in per_strat_rows["routing"]]
    bq = [x["recall"] / x["clients"] for x in per_strat_rows["broadcast"]]
    g3 = paired_diff_ci(rq, bq)
    # disclosure scan across all strategies' payloads
    raw = {t["head"] for t in triples} | {t["tail"] for t in triples}
    leaks = 0
    for rows in per_strat_rows.values():
        for row in rows:
            for p in row["payloads"]:
                if any(e in p for e in raw):
                    leaks += 1
    res["gates"] = {
        "competence_oracle_recall": res["oracle"]["recall"],
        "competence_pass": res["oracle"]["recall"] >= 0.60,
        "G2_diff": g2[0], "G2_lo": g2[1], "G2_hi": g2[2],
        "G2_pass": bool(g2[0] >= 0.03 and g2[1] > 0),
        "G3_diff": g3[0], "G3_lo": g3[1], "G3_hi": g3[2],
        "G3_pass": bool(g3[0] >= 0.05 and g3[1] > 0),
        "disclosure_leaks": leaks,
        "disclosure_pass": leaks == 0,
    }
    res["meta"] = {"n_queries": len(queries), "n_triples": len(triples),
                   "n_prototypes": len(protos), "seed": 42, "top_b": 3, "k": 5}
    OUT.mkdir(exist_ok=True)
    (OUT / "experiment.json").write_text(json.dumps(res, indent=2))
    for strat, v in res.items():
        if isinstance(v, dict) and "recall" in v:
            print(f"{strat:9s} recall={v['recall']:.4f} [{v['lo']:.4f},{v['hi']:.4f}] "
                  f"hit_all={v['hit_all']:.3f} routed={v['routed_all']:.3f} "
                  f"rec/client={v['rec_per_client']:.4f}")
    print("GATES:", json.dumps(res["gates"], indent=2))
    return res


if __name__ == "__main__":
    run()
