"""Train one model and keep its best epoch.

    python src/train.py --model cnn                        # RQ1: the answer alone
    python src/train.py --model lstm --mode prompt         # RQ2: the prompt alone ("both" for prompt + answer)
    python src/train.py --model cnn --held-out coding      # RQ3: train on the other three tasks, test on coding
    python src/train.py --model lstm --overfit 50          # check: can the model memorise 50 training rows?

Reads   data/splits/{train,val,test}.jsonl through data.py
Writes  models/<run>.pt                    the weights of the best epoch
        results/runs/<run>/history.json    loss and accuracy on the training and validation split, per epoch
        results/runs/<run>/config.json     every setting of the run, the model size and the time it took
        results/runs/<run>/metrics.json    scores on validation and test, written by evaluate.py
        results/runs/<run>/predictions_{val,test}.jsonl

<run> is <model>_<mode>, with _no_<task> appended when a task is held out.

How it trains:
  - cross-entropy loss and the Adam optimiser
  - gradients are clipped, which keeps the LSTM stable
  - after every epoch the model is scored on the validation split; the epoch with the best validation
    accuracy is kept (lower validation loss breaks a tie)
  - training stops early once validation accuracy has not improved for --patience epochs
  - the test split is used once, at the end, with the kept weights
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import MODES, collate, make_loaders
from evaluate import RUNS, evaluate_model
from models import LSTMClassifier, TextCNN

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"cnn": ("CNN", TextCNN), "lstm": ("LSTM", LSTMClassifier)}
TASKS = ("coding", "factual", "math", "writing")


def run_epoch(model, loader, loss_fn, device, optimizer=None, clip=None):
    """One pass over a loader: trains if an optimizer is given, otherwise only measures.

    Returns the mean loss per row and the accuracy.
    """
    model.train(optimizer is not None)
    total_loss, correct, rows = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            scores = model(tokens, lengths)
            loss = loss_fn(scores, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += int((scores.argmax(dim=1) == labels).sum())
            rows += len(labels)
    return total_loss / rows, correct / rows


def fit(model, train_loader, val_loader, device, epochs, patience, lr, clip, weights_path=None):
    """Train with early stopping. Returns the per-epoch history and the entry of the best epoch.

    If weights_path is given, the best epoch's weights are saved there and loaded back at the end.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history, best, stale = [], None, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer, clip)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                 "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
        history.append(entry)
        # Compare the recorded (rounded) numbers with each other, so that equal accuracies count as a tie.
        improved = best is None or (entry["val_acc"], -entry["val_loss"]) > (best["val_acc"], -best["val_loss"])
        print(f"  epoch {epoch:2d}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}{'  (best so far)' if improved else ''}", flush=True)
        if improved:
            best, stale = entry, 0
            if weights_path is not None:
                torch.save(model.state_dict(), weights_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"  stopping: no better validation accuracy for {patience} epochs")
                break
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return history, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--mode", choices=MODES, default="answer", help="which text the model reads")
    ap.add_argument("--held-out", choices=TASKS, help="train on the other tasks and test on this one (RQ3)")
    ap.add_argument("--run", help="name of the run; the default is built from the model, the mode and the held-out task")
    ap.add_argument("--epochs", type=int, default=20, help="most epochs to train")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=5.0, help="largest allowed gradient norm")
    ap.add_argument("--max-len", type=int, default=512, help="tokens kept per text")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=448)
    ap.add_argument("--overfit", type=int, metavar="N",
                    help="train on the first N training rows only and report how well they are memorised; writes nothing")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tasks = [t for t in TASKS if t != args.held_out] if args.held_out else None
    test_tasks = [args.held_out] if args.held_out else None
    data = make_loaders(args.mode, args.max_len, args.batch_size, train_tasks=train_tasks, test_tasks=test_tasks,
                        seed=args.seed)
    display_name, model_class = MODELS[args.model]
    model = model_class(len(data["vocab"]), len(data["names"]), emb_dim=args.emb_dim, dropout=args.dropout).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    if args.overfit:
        # A model that cannot memorise a handful of rows has a bug; a healthy one gets close to 100%.
        few = DataLoader(Subset(data["train"].dataset, range(args.overfit)), args.batch_size, shuffle=True,
                         collate_fn=collate)
        print(f"{display_name}: memorising {args.overfit} training rows ({parameters:,} parameters, {device})")
        history, _ = fit(model, few, few, device, args.epochs, patience=args.epochs, lr=args.lr, clip=args.clip)
        # The "val" numbers are the same rows scored with dropout switched off, which is what counts here.
        print(f"accuracy on those rows after {len(history)} epochs: {history[-1]['val_acc']:.3f}")
        return

    run = args.run or f"{args.model}_{args.mode}" + (f"_no_{args.held_out}" if args.held_out else "")
    run_dir = RUNS / run
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = ROOT / "models" / f"{run}.pt"
    weights.parent.mkdir(exist_ok=True)

    print(f"{run}: {display_name} on {args.mode}, {parameters:,} parameters, vocabulary {len(data['vocab']):,}, "
          f"{len(data['rows']['train'])} training rows, {device}")
    started = time.time()
    history, best = fit(model, data["train"], data["val"], device, args.epochs, args.patience, args.lr, args.clip,
                        weights)
    seconds = round(time.time() - started, 1)

    (run_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    config = {**vars(args), "run": run, "classifier": display_name, "parameters": parameters,
              "vocabulary": len(data["vocab"]), "rows": {s: len(data["rows"][s]) for s in ("train", "val", "test")},
              "epochs_run": len(history), "best_epoch": best["epoch"], "seconds": seconds, "device": device,
              "torch": torch.__version__}
    (run_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    metrics = evaluate_model(model, data, run_name=run, classifier=display_name, mode=args.mode, device=device)
    print(f"{run}: best epoch {best['epoch']} of {len(history)}, {seconds} s; "
          f"validation accuracy {metrics['val']['accuracy']:.3f}, test accuracy {metrics['test']['accuracy']:.3f} "
          f"(chance {metrics['chance']:.3f})")
    print(f"  test accuracy per task: {metrics['test']['accuracy_per_task']}")


if __name__ == "__main__":
    main()
