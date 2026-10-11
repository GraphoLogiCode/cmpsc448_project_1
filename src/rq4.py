"""RQ4: what distinguishes the LLMs?

    python src/rq4.py                  # everything; about 15 minutes, most of it the LSTM
    python src/rq4.py --skip-lstm      # about 3 minutes

Reads   data/splits/{train,val,test}.jsonl
Writes  results/rq4.json

Three parts:

  1. Style per model: length, vocabulary, lexical diversity and formatting of each LLM's answers,
     measured on the training split.

  2. Style features alone: a logistic regression that sees only 16 numbers per answer (length,
     punctuation rates, markdown counts, ...) and no words. It answers "how far do structural signals
     go on their own?". A second one sees the length only.

  3. Ablations: one kind of signal is removed from every answer, the classifiers are trained again on
     the changed text, and the drop in test accuracy shows how much they relied on it.

         full              the answer as it is
         no markup         markdown and LaTeX symbols removed, list markers removed
         words only        every punctuation mark and symbol removed
         structure only    every word replaced by the same placeholder; punctuation and markup kept
         first 100 tokens  the answer cut to its first 100 tokens, so length says little
         without opening   the first 30 tokens removed
         opening only      only the first 30 tokens

As everywhere else, settings are chosen on the validation split and the test split is read once per
trained classifier.
"""
import argparse
import contextlib
import io
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import baseline
import evaluate
from data import make_loaders, read_split, tokenize
from models import LSTMClassifier, TextCNN
from train import fit

ROOT = Path(__file__).resolve().parents[1]
OPENING = 30
WORD = re.compile(r"\w+")
# List markers and quote marks at the start of a line, LaTeX commands and delimiters, and the markdown
# characters * # `. Underscores, dollar signs and bars are left alone: they are also ordinary content.
MARKUP = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+|^\s*>\s?|\\[A-Za-z]+|\\[()\[\]]|[*#`]")


def first_tokens(text, start, stop):
    return " ".join(tokenize(text)[start:stop])


ABLATIONS = {
    "full": lambda t: t,
    "no markup": lambda t: MARKUP.sub(" ", t),
    "words only": lambda t: " ".join(WORD.findall(t)),
    "structure only": lambda t: WORD.sub("w", t),
    "first 100 tokens": lambda t: first_tokens(t, 0, 100),
    "without opening": lambda t: first_tokens(t, OPENING, None),
    "opening only": lambda t: first_tokens(t, 0, OPENING),
}

# The numbers the style-only classifier sees. Rates are per 100 tokens, so they do not repeat the length.
FEATURES = ["tokens", "share of tokens that are punctuation", "mean word length", "distinct tokens in the first 100",
            "mean sentence length", "lines per 100 tokens", "bold markers", "heading lines", "bullet lines",
            "numbered lines", "code fences", "LaTeX markers", "colons", "exclamation marks", "question marks",
            "curly quotes or long dashes"]


def style_features(text):
    tokens = tokenize(text)
    words = WORD.findall(text)
    n = max(len(tokens), 1)
    per_100 = lambda count: 100 * count / n
    sentences = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    return [len(tokens),
            sum(not WORD.fullmatch(t) for t in tokens) / n,
            statistics.mean(len(w) for w in words) if words else 0.0,
            len(set(tokens[:100])),
            len(words) / max(len(sentences), 1),
            per_100(text.count("\n") + 1),
            per_100(text.count("**")),
            per_100(len(re.findall(r"(?m)^#{1,6} ", text))),
            per_100(len(re.findall(r"(?m)^\s*[-*+] ", text))),
            per_100(len(re.findall(r"(?m)^\s*\d+[.)] ", text))),
            per_100(text.count("```")),
            per_100(len(re.findall(r"\\\(|\\\[|\\frac|\\boxed|\\text", text))),
            per_100(text.count(":")),
            per_100(text.count("!")),
            per_100(text.count("?")),
            per_100(len(re.findall("[‘’“”—]", text)))]


def style_by_model(train, names):
    """Part 1: averages per LLM over its training answers."""
    out = {}
    for name in names:
        rows = [r for r in train if r["llm_name"] == name]
        texts = [r["llm_output"] for r in rows]
        token_lists = [tokenize(t) for t in texts]
        features = np.array([style_features(t) for t in texts])
        counts = Counter(t for tokens in token_lists for t in tokens if WORD.fullmatch(t))
        total = sum(counts.values())
        share = lambda pattern: round(sum(bool(re.search(pattern, t)) for t in texts) / len(texts), 4)
        out[name] = {
            "answers": len(texts),
            "mean_tokens": round(float(features[:, 0].mean()), 1),
            "median_tokens": float(np.median(features[:, 0])),
            "distinct_words": len(counts),
            "distinct_words_per_10000_words": round(10000 * len(counts) / total, 1),
            "words_used_once": sum(c == 1 for c in counts.values()),
            "mean_distinct_tokens_in_first_100": round(float(features[features[:, 0] >= 100, 3].mean()), 1),
            "mean_word_length": round(float(features[:, 2].mean()), 2),
            "mean_sentence_length_words": round(float(features[:, 4].mean()), 1),
            "share_with_bold": share(r"\*\*[^*\n]+\*\*"), "share_with_heading": share(r"(?m)^#{1,6} "),
            "share_with_bullets": share(r"(?m)^\s*[-*+] "), "share_with_numbered_list": share(r"(?m)^\s*\d+[.)] "),
            "share_with_code_fence": share(r"```"), "share_with_latex": share(r"\\\(|\\\[|\\frac|\\boxed"),
            "share_with_curly_quotes": share("[‘’“”]"),
            "share_cut_off_at_token_limit": round(sum(r["finish"] == "length" for r in rows) / len(rows), 4),
        }
    return out


def feature_classifiers(splits, names):
    """Part 2: logistic regression on the style features, and on the length alone."""
    x = {s: np.array([style_features(r["llm_output"]) for r in rows]) for s, rows in splits.items()}
    y = {s: [r["llm_name"] for r in rows] for s, rows in splits.items()}
    tasks = {s: [r["task"] for r in rows] for s, rows in splits.items()}
    out = {}
    for label, columns in (("all style features", list(range(len(FEATURES)))), ("length only", [0])):
        scaler = StandardScaler().fit(x["train"][:, columns])
        best = None
        for c in baseline.C_VALUES:
            model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x["train"][:, columns]), y["train"])
            val = evaluate.score(y["val"], list(model.predict(scaler.transform(x["val"][:, columns]))), tasks["val"], names)
            if best is None or val["accuracy"] > best[1]["accuracy"]:
                best = (model, val, c)
        model, val, c = best
        test = evaluate.score(y["test"], list(model.predict(scaler.transform(x["test"][:, columns]))), tasks["test"], names)
        out[label] = {"C": c, "val": val, "test": test}
        if len(columns) > 1:
            # With standardised inputs, a larger weight means the feature moves the prediction more.
            out[label]["strongest_features"] = {
                name: [[FEATURES[i], round(float(weights[i]), 2)] for i in np.argsort(-np.abs(weights))[:5]]
                for name, weights in zip(model.classes_, model.coef_)}
    return out


def ablation(name, transform, splits, names, device, with_lstm):
    """Part 3: train every classifier on the transformed answers; return validation and test scores."""
    out = {}
    texts = {s: [transform(r["llm_output"]) for r in rows] for s, rows in splits.items()}
    labels = {s: [r["llm_name"] for r in rows] for s, rows in splits.items()}
    vectoriser, classifier, _, _ = baseline.fit(texts["train"], labels["train"], texts["val"], labels["val"],
                                                baseline.WITH_PUNCTUATION)
    out["TF-IDF baseline"] = {s: baseline.score(vectoriser, classifier, splits[s], texts[s], labels[s], names)
                              for s in ("val", "test")}

    for label, model_class in (("CNN", TextCNN), ("LSTM", LSTMClassifier)):
        if label == "LSTM" and not with_lstm:
            continue
        torch.manual_seed(448)
        data = make_loaders("answer", transform=transform)
        model = model_class(len(data["vocab"]), len(names)).to(device)
        with contextlib.redirect_stdout(io.StringIO()):          # hide the per-epoch lines of fit()
            fit(model, data["train"], data["val"], device, epochs=20, patience=3, lr=1e-3, clip=5.0,
                weights_path=ROOT / "models" / "rq4_tmp.pt")
        out[label] = {}
        for s in ("val", "test"):
            predicted = [names[i] for i in evaluate.predict(model, data[s], device)]
            out[label][s] = evaluate.score(labels[s], predicted, [r["task"] for r in splits[s]], names)
    (ROOT / "models" / "rq4_tmp.pt").unlink(missing_ok=True)
    print(f"{name:18s} " + "  ".join(f"{k} {v['test']['accuracy']:.3f}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-lstm", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {s: read_split(s) for s in ("train", "val", "test")}
    names = sorted({r["llm_name"] for r in splits["train"]})

    results = {"models": names, "chance": round(1 / len(names), 4), "opening_tokens": OPENING,
               "style_by_model": style_by_model(splits["train"], names),
               "style_feature_names": FEATURES,
               "feature_classifiers": feature_classifiers(splits, names)}
    for label, r in results["feature_classifiers"].items():
        print(f"{label:20s} test accuracy {r['test']['accuracy']:.3f}")
    print("\ntest accuracy after each ablation:")
    results["ablations"] = {name: ablation(name, transform, splits, names, device, not args.skip_lstm)
                            for name, transform in ABLATIONS.items()}

    out = ROOT / "results" / "rq4.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
