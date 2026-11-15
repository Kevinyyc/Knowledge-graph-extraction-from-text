"""Gather the gold graph and the graphs of several runs into one self-contained HTML viewer.

The viewer has two levels:
- document: the entities of one document, the gold facts and the facts a run kept;
- corpus: the merged graph (nodes keyed as in build.py), around one entity.

Each fact is correct (gold and predicted), missed (gold only) or wrong (predicted only).
"""
import json
from pathlib import Path

from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.graph.build import build_graph, node_key

TEMPLATE = Path(__file__).with_name("viewer.html")


def _prf(n_correct, n_predicted, n_gold):
    precision = n_correct / n_predicted if n_predicted else 0.0
    recall = n_correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if n_correct else 0.0
    return {"precision": round(100 * precision, 2), "recall": round(100 * recall, 2),
            "f1": round(100 * f1, 2)}


def _triples(graph):
    return {(edge["head"], edge["relation"], edge["tail"]) for edge in graph["edges"]}


def viewer_data(docs, relation_names, runs, split="dev"):
    """Return the data of the viewer.

    runs maps a run name to its facts {(document index, head id, tail id, relation code): score}.
    """
    gold = gold_facts(docs)
    gold_graph = build_graph(docs, dict.fromkeys(gold, 1.0), relation_names)
    run_graphs = {name: build_graph(docs, facts, relation_names) for name, facts in runs.items()}

    # Corpus nodes: one table shared by the gold graph and every run graph.
    nodes = {}
    for graph in (gold_graph, *run_graphs.values()):
        for node in graph["nodes"]:
            nodes.setdefault(node["id"], node)
    node_ids = sorted(nodes)
    position = {key: index for index, key in enumerate(node_ids)}

    def edges(graph, with_score):
        return [[position[edge["head"]], edge["relation"], position[edge["tail"]], edge["support"]]
                + ([round(edge["score"], 3)] if with_score else []) for edge in graph["edges"]]

    run_entries = []
    for name, facts in runs.items():
        n_correct = len(gold & set(facts))
        graph_correct = len(_triples(gold_graph) & _triples(run_graphs[name]))
        run_entries.append({
            "name": name,
            "facts": _prf(n_correct, len(facts), len(gold)),
            "graph": _prf(graph_correct, len(run_graphs[name]["edges"]),
                          len(gold_graph["edges"])),
            "n_facts": len(facts), "n_edges": len(run_graphs[name]["edges"]),
        })

    by_document = [[[] for _ in docs] for _ in runs]
    for run_index, facts in enumerate(runs.values()):
        for (index, head, tail, relation), score in sorted(facts.items()):
            by_document[run_index][index].append([head, tail, relation, round(score, 3)])

    documents = []
    for index, doc in enumerate(docs):
        documents.append({
            "title": doc["title"],
            "sents": [" ".join(tokens) for tokens in doc["sents"]],
            "entities": [{"name": entity["mentions"][0]["name"], "type": entity["type"],
                          "node": position.get(node_key(doc, {"id": entity_id, **entity})),
                          "sents": sorted({mention["sent_id"] for mention in entity["mentions"]})}
                         for entity_id, entity in enumerate(doc["entities"])],
            "gold": [[relation["h"], relation["t"], relation["r"], relation.get("evidence", [])]
                     for relation in doc["relations"]],
            "runs": [by_document[run_index][index] for run_index in range(len(runs))],
        })

    return {
        "split": split,
        "relations": relation_names,
        "runs": run_entries,
        "documents": documents,
        "corpus": {
            "nodes": [{"id": key, "label": nodes[key]["label"], "type": nodes[key]["type"],
                       "n_documents": len(nodes[key]["documents"])} for key in node_ids],
            "gold": edges(gold_graph, with_score=False),
            "runs": [edges(graph, with_score=True) for graph in run_graphs.values()],
        },
    }


def render_viewer(data):
    """Return the HTML page of the viewer, with its data inlined."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__VIEWER_DATA__", payload)
