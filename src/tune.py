"""Compare settings of the two models on the VALIDATION split. The test split is never read here.

    python src/tune.py                  # every trial
    python src/tune.py --only lstm      # trials whose name contains "lstm"

Appends one line per trial to results/tuning.jsonl: the best validation accuracy and loss, the epoch
they were reached in, the number of parameters and the time.

Each trial changes one thing about a starting point (embedding 128, dropout 0.5, learning rate 0.001,
batch 32, 512 tokens; CNN: 128 filters of widths 2 to 5; LSTM: one bidirectional layer of 128 units,
max over time). The "final" trials repeat the settings that train.py uses with three more seeds, to
show how much the score moves between runs that differ only in the random seed.
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import torch

from data import make_loaders
from models import LSTMClassifier, TextCNN
from train import fit

ROOT = Path(__file__).resolve().parents[1]
CNN = dict(emb_dim=128, n_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.5)
LSTM = dict(emb_dim=128, hidden=128, layers=1, dropout=0.5, pool="max")
FIT = dict(max_len=512, batch_size=32, lr=1e-3, patience=3, epochs=20, seed=448)

# name, model class, model settings, training settings
TRIALS = [
    ("cnn starting point", TextCNN, CNN, {}),
    ("cnn 700 tokens", TextCNN, CNN, {"max_len": 700}),
    ("cnn widths 1 to 5", TextCNN, {**CNN, "kernel_sizes": (1, 2, 3, 4, 5)}, {}),
    ("cnn widths 3 to 5", TextCNN, {**CNN, "kernel_sizes": (3, 4, 5)}, {}),
    ("cnn 256 filters", TextCNN, {**CNN, "n_filters": 256}, {}),
    ("cnn embedding 256", TextCNN, {**CNN, "emb_dim": 256}, {}),
    ("cnn dropout 0.3", TextCNN, {**CNN, "dropout": 0.3}, {}),
    ("cnn learning rate 0.0005", TextCNN, CNN, {"lr": 5e-4}),
    ("cnn batch 64", TextCNN, CNN, {"batch_size": 64}),
    ("lstm starting point", LSTMClassifier, LSTM, {}),
    ("lstm last memory instead of max", LSTMClassifier, {**LSTM, "pool": "last"}, {}),
    ("lstm 700 tokens", LSTMClassifier, LSTM, {"max_len": 700}),
    ("lstm 256 units", LSTMClassifier, {**LSTM, "hidden": 256}, {}),
    ("lstm 2 layers", LSTMClassifier, {**LSTM, "layers": 2}, {}),
    ("lstm dropout 0.3", LSTMClassifier, {**LSTM, "dropout": 0.3}, {}),
    ("lstm learning rate 0.002", LSTMClassifier, LSTM, {"lr": 2e-3}),
    ("lstm embedding 256", LSTMClassifier, {**LSTM, "emb_dim": 256}, {}),
    ("lstm patience 5", LSTMClassifier, LSTM, {"patience": 5, "epochs": 30}),
] + [(f"final cnn seed {s}", TextCNN, CNN, {"seed": s}) for s in (1, 2, 3)] \
  + [(f"final lstm seed {s}", LSTMClassifier, {**LSTM, "hidden": 256}, {"seed": s}) for s in (1, 2, 3)]


def trial(name, model_class, model_settings, changes, device):
    s = {**FIT, **changes}
    torch.manual_seed(s["seed"])
    data = make_loaders("answer", s["max_len"], s["batch_size"], seed=s["seed"])
    model = model_class(len(data["vocab"]), len(data["names"]), **model_settings).to(device)
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()):          # hide the per-epoch lines of fit()
        history, best = fit(model, data["train"], data["val"], device, s["epochs"], s["patience"], s["lr"], clip=5.0)
    return {"name": name, "val_acc": best["val_acc"], "val_loss": best["val_loss"], "best_epoch": best["epoch"],
            "epochs": len(history), "seconds": round(time.time() - started, 1),
            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="run only the trials whose name contains this text")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = ROOT / "results" / "tuning.jsonl"
    for name, model_class, model_settings, changes in TRIALS:
        if args.only not in name:
            continue
        row = trial(name, model_class, model_settings, changes, device)
        print(f"{name:34s} val acc {row['val_acc']:.4f}  loss {row['val_loss']:.4f}  best epoch "
              f"{row['best_epoch']:2d} of {row['epochs']:2d}  {row['seconds']:6.1f} s  {row['parameters']:,} parameters",
              flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
