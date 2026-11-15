"""Experiment B: how many labels the PromptORE methods (M1, M2) need.

The detector, the relation classifiers (M2) and the cluster naming (M1) are fitted on the
cached PromptORE train sample of a random share of the train documents, the same share
that train_doc_model.py --train-fraction draws with the same seed. The fact threshold,
and k for M1, are chosen on dev, as in compare_methods.py, and only dev is scored.

Writes results/label_efficiency/promptore.json, one row per method, fraction and seed.
The document model rows come from the train reports of train_doc_model.py runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kg_extraction.data.linked_docred import load_relation_names, load_split, train_subset
from kg_extraction.evaluation.relations import gold_facts, relation_scores, train_fact_names
from kg_extraction.pairs.candidates import allowed_type_pairs
from kg_extraction.relations.cache import cached_embeddings
from kg_extraction.relations.clustering import ClusterTyper
from kg_extraction.relations.probe import (
    RelationProbe, best_threshold, facts_above, pair_instances, pair_list, sample_train_pairs)


def dev_scores(docs, pairs, scores, relations, train_names):
    threshold = best_threshold(pairs, scores, relations, gold_facts(docs))
    facts = facts_above(pairs, scores, relations, threshold)
    result = relation_scores(docs, facts, train_names)
    return {"f1": result["f1"], "ign_f1": result["ign_f1"], "threshold": round(threshold, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/Linked-Re-DocRED")
    parser.add_argument("--embeddings-dir", default="data/processed/embeddings")
    parser.add_argument("--output", default="results/label_efficiency/promptore.json")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k-values", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    args = parser.parse_args()

    docs = {split: load_split(args.data_dir, split) for split in ("train", "dev")}
    relations = sorted(load_relation_names(args.data_dir))
    train_names = train_fact_names(docs["train"])
    allowed = allowed_type_pairs(docs["train"])

    # The seed 0 train sample of compare_methods.py and its cached embeddings.
    all_pairs = pair_list(docs["train"], allowed)
    kept, _ = sample_train_pairs(all_pairs, args.negatives_per_positive, 0)
    train_pairs = [all_pairs[index] for index in kept]
    embeddings = cached_embeddings(Path(args.embeddings_dir) / "train_sample.npy",
                                   pair_instances(docs["train"], train_pairs))
    dev_pairs = pair_list(docs["dev"], allowed)
    dev_embeddings = cached_embeddings(Path(args.embeddings_dir) / "dev_isolated.npy",
                                       pair_instances(docs["dev"], dev_pairs))
    unrelated_per_doc = np.bincount([pair[0] for pair in all_pairs if not pair[3]],
                                    minlength=len(docs["train"]))

    output = Path(args.output)
    rows = json.loads(output.read_text()) if output.exists() else []
    done = {(row["method"], row["fraction"], row["seed"]) for row in rows}
    for fraction in args.fractions:
        for seed in args.seeds if fraction < 1 else args.seeds[:1]:
            if ("M2", fraction, seed) in done:
                continue
            chosen = set(train_subset(len(docs["train"]), fraction, seed))
            rows_kept = [index for index, pair in enumerate(train_pairs) if pair[0] in chosen]
            labels = [train_pairs[index][3] for index in rows_kept]
            sampled_unrelated = sum(1 for codes in labels if not codes)
            negative_weight = (unrelated_per_doc[sorted(chosen)].sum()
                               / max(sampled_unrelated, 1))
            subset = embeddings[rows_kept]
            probe = RelationProbe(relations).fit(subset, labels, negative_weight)
            base = {"fraction": fraction, "seed": seed, "train_docs": len(chosen),
                    "train_facts": sum(len(codes) for codes in labels)}
            rows.append({"method": "M2", **base, **dev_scores(
                docs["dev"], dev_pairs, probe.scores(dev_embeddings), relations, train_names)})

            related = np.array([bool(codes) for codes in labels])
            detection = probe.detector.predict_proba(dev_embeddings)[:, 1]
            best = None
            for k in args.k_values:
                if k > related.sum() // 2:
                    continue
                typer = ClusterTyper(relations, k, seed).fit(
                    subset[related], [codes for codes in labels if codes])
                result = dev_scores(docs["dev"], dev_pairs,
                                    detection[:, None] * typer.scores(dev_embeddings),
                                    relations, train_names)
                if best is None or result["f1"] > best["f1"]:
                    best = {**result, "k": k}
            rows.append({"method": "M1", **base, **best})
            print(json.dumps(rows[-2:]), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
