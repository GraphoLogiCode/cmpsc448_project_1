"""Score a classifier's predictions, save them, and compare all classifiers in one table.

    python src/evaluate.py        # table of the baseline and of every run in results/runs

Use from train.py, after training and with the best weights loaded:

    from evaluate import evaluate_model
    metrics = evaluate_model(model, data, run_name="cnn_answer", classifier="CNN", mode="answer", device=device)

That writes, to results/runs/<run_name>/:
    predictions_val.jsonl, predictions_test.jsonl    one row per example: prompt_id, task, llm_name, predicted
    metrics.json                                     accuracy, macro F1, per-model scores, confusion matrix,
                                                     accuracy per task, for both splits

Every classifier is scored by the same function, score(), so the numbers are comparable.
The model is called as model(tokens, lengths) and must return one score per class for each row;
tokens is moved to the device, lengths stays on the CPU.
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
BASELINE_INPUTS = {"prompt only": "prompt", "answer only": "answer", "prompt + answer": "both"}


def score(labels, predicted, tasks, names):
    """All reported numbers for one split. labels and predicted are model names, tasks the task of each row."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=names, zero_division=0)
    return {"rows": len(labels),
            "accuracy": round(accuracy_score(labels, predicted), 4),
            "macro_f1": round(f1_score(labels, predicted, labels=names, average="macro", zero_division=0), 4),
            "per_model": {n: {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4)}
                          for n, p, r, f in zip(names, precision, recall, f1)},
            "confusion_matrix": confusion_matrix(labels, predicted, labels=names).tolist(),
            "accuracy_per_task": {t: round(accuracy_score([l for l, k in zip(labels, tasks) if k == t],
                                                          [p for p, k in zip(predicted, tasks) if k == t]), 4)
                                  for t in sorted(set(tasks))}}


def predict(model, loader, device):
    """The predicted class index for every row of the loader, in the loader's order."""
    import torch

    model.eval()
    predicted = []
    with torch.no_grad():
        for tokens, lengths, _ in loader:
            predicted += model(tokens.to(device), lengths).argmax(dim=1).tolist()
    return predicted


def evaluate_model(model, data, run_name, classifier, mode, device, splits=("val", "test")):
    """Predict the given splits, save predictions and metrics under results/runs/<run_name>, return the metrics.

    data is what make_loaders() returned. classifier is a display name such as "CNN" or "LSTM";
    mode is the input the model was trained on ("answer", "prompt" or "both").
    """
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    names = data["names"]
    metrics = {"run": run_name, "classifier": classifier, "input": mode, "models": names,
               "chance": round(1 / len(names), 4),
               "train_tasks": data.get("train_tasks"), "test_tasks": data.get("test_tasks")}
    for split in splits:
        rows = data["rows"][split]
        predicted = [names[i] for i in predict(model, data[split], device)]
        assert len(predicted) == len(rows), "the loader and the rows have different lengths"
        with open(run_dir / f"predictions_{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for row, p in zip(rows, predicted):
                f.write(json.dumps({"prompt_id": row["prompt_id"], "task": row["task"],
                                    "llm_name": row["llm_name"], "predicted": p}) + "\n")
        metrics[split] = score([r["llm_name"] for r in rows], predicted, [r["task"] for r in rows], names)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def collect(results_dir=ROOT / "results"):
    """Every scored classifier as a list of dicts: classifier, input, run, val, test, models, chance."""
    out = []
    baseline = results_dir / "baseline.json"
    if baseline.exists():
        b = json.loads(baseline.read_text(encoding="utf-8"))
        for name, mode in BASELINE_INPUTS.items():
            out.append({"classifier": "TF-IDF baseline", "input": mode, "run": "baseline", "models": b["models"],
                        "chance": b["chance"], "val": b["inputs"][name]["val"], "test": b["inputs"][name]["test"]})
    for metrics_file in sorted((results_dir / "runs").glob("*/metrics.json")):
        out.append(json.loads(metrics_file.read_text(encoding="utf-8")))
    return out


def main():
    results = collect()
    if not results:
        raise SystemExit("nothing to show: run src/baseline.py or train a model first")
    tasks = sorted(results[0]["test"]["accuracy_per_task"])
    print(f"chance is {results[0]['chance']:.3f}; accuracy per task is on the test split\n")
    print(f"{'classifier':16s} {'input':7s} {'run':16s} {'val acc':>8s} {'test acc':>9s} {'macro F1':>9s}  "
          + "  ".join(f"{t:>8s}" for t in tasks))
    for r in results:
        per_task = r["test"]["accuracy_per_task"]
        print(f"{r['classifier']:16s} {r['input']:7s} {r['run']:16s} {r['val']['accuracy']:8.3f} "
              f"{r['test']['accuracy']:9.3f} {r['test']['macro_f1']:9.3f}  "
              + "  ".join(f"{per_task[t]:8.3f}" if t in per_task else f"{'-':>8s}" for t in tasks))


if __name__ == "__main__":
    main()
