"""Experiment D: can the fact and graph edge scores be trusted?

For each run, facts are put into 10 score bins (every fact above 0.02 in
scores_dev_seed0.pt), and the share of gold facts in a bin is compared with its mean score.
ECE is the gap between the two, weighted by the number of facts per bin. The edges of the
graph built by build_graph.py (<run>/graph/dev_graph.json) are binned the same way.

Dev only. Writes results/calibration/calibration.json, read by make_figures.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kg_extraction.data.linked_docred import load_split
from kg_extraction.evaluation.graph import triples
from kg_extraction.evaluation.relations import gold_facts
from kg_extraction.graph.build import build_graph


def reliability(scores, correct, n_bins=10):
    """Return per-bin counts, mean scores and precisions, and the expected calibration error."""
    scores, correct = np.asarray(scores), np.asarray(correct, dtype=float)
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            mean, precision = scores[mask].mean(), correct[mask].mean()
            ece += mask.sum() / len(scores) * abs(mean - precision)
            rows.append({"low": index / n_bins, "high": (index + 1) / n_bins,
                         "count": int(mask.sum()), "mean_score": round(float(mean), 4),
                         "precision": round(float(precision), 4)})
    return {"bins": rows, "ece": round(float(100 * ece), 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--runs", nargs="+", default=[
        "results/doc-tune-ens+old", "results/doc-tune4", "results/doc-full",
        "results/promptore-classifier"])
    parser.add_argument("--output", default="results/calibration/calibration.json")
    args = parser.parse_args()

    dev = load_split(args.data_dir, "dev")
    gold = gold_facts(dev)
    gold_edges = triples(build_graph(dev, dict.fromkeys(gold, 1.0)))
    report = {}
    for run in map(Path, args.runs):
        facts = torch.load(run / "scores_dev_seed0.pt", weights_only=False)
        report[run.name] = {"facts": reliability(list(facts.values()),
                                                 [fact in gold for fact in facts])}
        graph_path = run / "graph" / "dev_graph.json"
        if graph_path.exists():
            edges = json.loads(graph_path.read_text(encoding="utf-8"))["edges"]
            report[run.name]["edges"] = reliability(
                [edge["max_score"] for edge in edges],
                [(edge["head"], edge["relation"], edge["tail"]) in gold_edges for edge in edges])
        print(run.name, {key: value["ece"] for key, value in report[run.name].items()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
