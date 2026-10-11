"""TF-IDF + logistic regression on the same splits as the neural models.

    python src/baseline.py                     # tokens are words and punctuation marks
    python src/baseline.py --words-only        # tokens are words only; markdown and punctuation are dropped
    python src/baseline.py --held-out coding   # RQ3: train on the other three tasks, test on coding

Reads   data/splits/{train,val,test}.jsonl
Writes  results/baseline.json (or baseline_words_only.json)
        results/runs/baseline_answer_no_<task>/metrics.json when a task is held out

It gives the number the CNN and the LSTM have to beat. Three inputs are tried, as RQ2 asks:
the prompt alone, the answer alone, and both together.

The prompt alone must score exactly one third. Every prompt is in the data three times, once per
model, with identical text; whatever the classifier predicts for that text is right for one of the
three rows and wrong for the other two. A score above one third would mean the split leaked.

The strength of the regularisation (C) is chosen on the validation split. The test split is used
once, for the numbers that are reported.

Needs scikit-learn.
"""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import evaluate

ROOT = Path(__file__).resolve().parents[1]

INPUTS = {"prompt only": lambda r: r["llm_input"],
          "answer only": lambda r: r["llm_output"],
          "prompt + answer": lambda r: r["llm_input"] + "\n\n" + r["llm_output"]}
WITH_PUNCTUATION = r"\w+|[^\w\s]"     # a word, or one punctuation or markdown character
WORDS_ONLY = r"\b\w\w+\b"             # words of two or more characters (scikit-learn's default)
C_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)    # validation accuracy peaks inside this range
TOP_FEATURES = 15


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit(train_texts, train_labels, val_texts, val_labels, token_pattern):
    """The vectoriser and the classifier with the best validation accuracy over C_VALUES."""
    vectoriser = TfidfVectorizer(lowercase=True, token_pattern=token_pattern, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)
    x_train, x_val = vectoriser.fit_transform(train_texts), vectoriser.transform(val_texts)
    best = None
    for c in C_VALUES:
        classifier = LogisticRegression(C=c, max_iter=3000).fit(x_train, train_labels)
        accuracy = accuracy_score(val_labels, classifier.predict(x_val))
        if best is None or accuracy > best[0]:
            best = (accuracy, c, classifier)
    return vectoriser, best[2], best[1], best[0]


def score(vectoriser, classifier, rows, texts, labels, names):
    """The same numbers as for the neural models, from the function they are scored with."""
    predicted = list(classifier.predict(vectoriser.transform(texts)))
    return evaluate.score(labels, predicted, [r["task"] for r in rows], names)


def strongest_features(vectoriser, classifier, names):
    """For each model, the tokens whose presence pushes the prediction towards it most."""
    features = vectoriser.get_feature_names_out()
    out = {}
    for name, weights in zip(classifier.classes_, classifier.coef_):
        top = weights.argsort()[::-1][:TOP_FEATURES]
        out[name] = [[features[i], round(float(weights[i]), 2)] for i in top]
    return {n: out[n] for n in names}


def held_out_run(task, train, val, test, names, token_pattern):
    """RQ3 for the baseline: fit on the answers of the other tasks, test on the answers of the held-out task.

    Writes metrics.json in the same form as evaluate.evaluate_model, so the run shows up in the same
    table and figure as the neural models.
    """
    text_of = INPUTS["answer only"]
    seen = [[r for r in rows if r["task"] != task] for rows in (train, val)]
    unseen = [r for r in test if r["task"] == task]
    label = lambda rows: [r["llm_name"] for r in rows]
    vectoriser, classifier, c, _ = fit([text_of(r) for r in seen[0]], label(seen[0]),
                                       [text_of(r) for r in seen[1]], label(seen[1]), token_pattern)
    run = f"baseline_answer_no_{task}"
    metrics = {"run": run, "classifier": "TF-IDF baseline", "input": "answer", "models": names,
               "chance": round(1 / len(names), 4),
               "train_tasks": sorted({r["task"] for r in train} - {task}), "test_tasks": [task], "C": c,
               "val": score(vectoriser, classifier, seen[1], [text_of(r) for r in seen[1]], label(seen[1]), names),
               "test": score(vectoriser, classifier, unseen, [text_of(r) for r in unseen], label(unseen), names)}
    (evaluate.RUNS / run).mkdir(parents=True, exist_ok=True)
    (evaluate.RUNS / run / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--words-only", action="store_true", help="drop punctuation and markdown characters")
    ap.add_argument("--held-out", help="train on the other tasks and test on this one (RQ3)")
    args = ap.parse_args()

    train, val, test = (read_jsonl(args.splits_dir / f"{s}.jsonl") for s in ("train", "val", "test"))
    names = sorted({r["llm_name"] for r in train})
    labels = {name: [r["llm_name"] for r in rows] for name, rows in (("train", train), ("val", val), ("test", test))}
    token_pattern = WORDS_ONLY if args.words_only else WITH_PUNCTUATION

    if args.held_out:
        m = held_out_run(args.held_out, train, val, test, names, token_pattern)
        print(f"{m['run']}: validation accuracy on the seen tasks {m['val']['accuracy']:.3f}, "
              f"test accuracy on {args.held_out} {m['test']['accuracy']:.3f} (chance {m['chance']:.3f})")
        return

    results = {"tokens": "words only" if args.words_only else "words and punctuation", "models": names,
               "chance": round(1 / len(names), 4), "rows": {"train": len(train), "val": len(val), "test": len(test)},
               "inputs": {}}
    for name, text_of in INPUTS.items():
        texts = {split: [text_of(r) for r in rows] for split, rows in (("train", train), ("val", val), ("test", test))}
        vectoriser, classifier, c, _ = fit(texts["train"], labels["train"], texts["val"], labels["val"], token_pattern)
        results["inputs"][name] = {"C": c, "features": len(vectoriser.get_feature_names_out()),
                                   "val": score(vectoriser, classifier, val, texts["val"], labels["val"], names),
                                   "test": score(vectoriser, classifier, test, texts["test"], labels["test"], names)}
        if name == "answer only":
            results["strongest_features_answer_only"] = strongest_features(vectoriser, classifier, names)

    out = ROOT / "results" / ("baseline_words_only.json" if args.words_only else "baseline.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"TF-IDF + logistic regression, tokens: {results['tokens']}; chance is {results['chance']:.3f}")
    print(f"\n{'input':16s} {'C':>5s} {'features':>9s} {'val acc':>8s} {'test acc':>9s} {'test macro F1':>14s}")
    for name, r in results["inputs"].items():
        print(f"{name:16s} {r['C']:5g} {r['features']:9d} {r['val']['accuracy']:8.3f} {r['test']['accuracy']:9.3f} "
              f"{r['test']['macro_f1']:14.3f}")
    answer = results["inputs"]["answer only"]["test"]
    print("\nanswer only, test split. Confusion matrix (rows: true model, columns: predicted model):")
    print(" " * 14 + "".join(f"{n:>14s}" for n in names))
    for n, row in zip(names, answer["confusion_matrix"]):
        print(f"{n:14s}" + "".join(f"{v:14d}" for v in row))
    print("accuracy per task:", answer["accuracy_per_task"])
    print("\nstrongest features per model (answer only):")
    for n, feats in results["strongest_features_answer_only"].items():
        print(f"  {n}: " + ", ".join(repr(f) for f, _ in feats))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
