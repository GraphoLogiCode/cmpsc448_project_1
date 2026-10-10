"""Turn the raw responses into train / validation / test files.

    python src/build_dataset.py

Reads   data/prompts/prompts.jsonl
        data/responses/<model>.jsonl (trial files are skipped)
Writes  data/splits/{train,val,test}.jsonl
        data/splits/split_report.json     what was changed or removed, and how many rows each split holds

Steps:
  1. Split by prompt. Within each task a fixed random 70% / 15% / 15% of the prompts go to
     train / val / test, and every answer to a prompt goes to the same split as the prompt.
     The split is drawn from the full prompt list, so it does not change when the cleaning does.
  2. Strip whitespace from both ends of every answer. Every Mistral answer starts with a space,
     which would otherwise give the label away in the first character.
  3. Drop an answer that is empty, or that contains its own model's or maker's name when the prompt
     did not mention it ("As an AI developed by Alibaba Cloud ...").
  4. Drop a prompt altogether if any model lost its answer in step 3, so every remaining prompt
     has one answer from every model and the classes stay exactly balanced.

Kept on purpose: answers cut off at the token limit (see the finish column), refusals, and the few
answers that are identical across models. The task column is kept so one task can be held out (RQ3).
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")

# The names an answer must not give away, by model. Same patterns as in check_responses.py.
OWN_NAMES = {"llama3.1:8b": r"\bllama\b|\bmeta ai\b|\bmeta\b(?= (?:ai|platforms))",
             "qwen2.5:7b": r"\bqwen\b|\balibaba\b|\btongyi\b",
             "mistral:7b": r"\bmistral\b"}
COLUMNS = ("prompt_id", "task", "llm_name", "llm_family", "llm_input", "llm_output", "finish")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assign_splits(prompts, val_share, test_share, seed):
    """prompt id -> split name, drawn separately inside each task."""
    split_of = {}
    for task in sorted({p["task"] for p in prompts}):
        ids = sorted(p["id"] for p in prompts if p["task"] == task)
        random.Random(f"{seed}-{task}").shuffle(ids)
        n_val, n_test = round(len(ids) * val_share), round(len(ids) * test_share)
        for i, prompt_id in enumerate(ids):
            split_of[prompt_id] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return split_of


def names_itself(row):
    """True if the answer contains its own model's or maker's name and the prompt did not."""
    pattern = OWN_NAMES[row["llm_name"]]
    return bool(re.search(pattern, row["llm_output"], re.I)) and not re.search(pattern, row["llm_input"], re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "prompts.jsonl")
    ap.add_argument("--responses-dir", type=Path, default=ROOT / "data" / "responses")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    ap.add_argument("--val", type=float, default=0.15, help="share of the prompts that goes to validation")
    ap.add_argument("--test", type=float, default=0.15, help="share of the prompts that goes to the test set")
    ap.add_argument("--seed", type=int, default=448)
    args = ap.parse_args()

    prompts = read_jsonl(args.prompts)
    split_of = assign_splits(prompts, args.val, args.test, args.seed)

    files = sorted(f for f in args.responses_dir.glob("*.jsonl") if not f.stem.endswith("_trial"))
    rows = [row for f in files for row in read_jsonl(f)]
    models = sorted({r["llm_name"] for r in rows})
    assert set(models) == set(OWN_NAMES), f"models in the data ({models}) and in OWN_NAMES differ"
    assert all(r["prompt_id"] in split_of for r in rows), "a response refers to a prompt that is not in the prompt file"

    stripped, removed = Counter(), []
    answers = {}                                    # prompt id -> {model: row}
    for r in rows:
        clean = r["llm_output"].strip()
        stripped[r["llm_name"]] += clean != r["llm_output"]
        r = {**r, "llm_output": clean}
        if not clean:
            removed.append({"prompt_id": r["prompt_id"], "llm_name": r["llm_name"], "why": "empty answer"})
        elif names_itself(r):
            removed.append({"prompt_id": r["prompt_id"], "llm_name": r["llm_name"], "why": "names its own model or maker"})
        else:
            assert r["llm_name"] not in answers.setdefault(r["prompt_id"], {}), "two answers from one model to one prompt"
            answers[r["prompt_id"]][r["llm_name"]] = r

    complete = {pid: by_model for pid, by_model in answers.items() if len(by_model) == len(models)}
    incomplete = sorted(set(split_of) - set(complete))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        ids = sorted(pid for pid in complete if split_of[pid] == split)
        with open(args.out_dir / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for pid in ids:
                for model in models:
                    f.write(json.dumps({c: complete[pid][model][c] for c in COLUMNS}, ensure_ascii=False) + "\n")
        kept = [complete[pid][m] for pid in ids for m in models]
        counts[split] = {"prompts": len(ids), "rows": len(kept),
                         "rows_per_model": dict(Counter(r["llm_name"] for r in kept)),
                         "prompts_per_task": dict(Counter(complete[pid][models[0]]["task"] for pid in ids)),
                         "cut_off_rows": sum(r["finish"] == "length" for r in kept)}

    # The properties the later experiments rely on.
    written = {split: read_jsonl(args.out_dir / f"{split}.jsonl") for split in SPLITS}
    ids_in = {split: {r["prompt_id"] for r in rs} for split, rs in written.items()}
    assert not (ids_in["train"] & ids_in["val"] or ids_in["train"] & ids_in["test"] or ids_in["val"] & ids_in["test"]), \
        "a prompt is in two splits"
    assert all(len(set(c["rows_per_model"].values())) == 1 for c in counts.values()), "classes are not balanced"
    assert all(r["llm_output"] == r["llm_output"].strip() and r["llm_output"] for rs in written.values() for r in rs)

    report = {"seed": args.seed, "shares": {"val": args.val, "test": args.test}, "models": models,
              "raw_rows": len(rows), "answers_changed_by_stripping_whitespace": dict(stripped),
              "answers_removed": removed, "prompts_removed_because_an_answer_was_removed": incomplete,
              "splits": counts, "rows_written": sum(c["rows"] for c in counts.values())}
    (args.out_dir / "split_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"raw rows: {len(rows)} from {len(models)} models")
    print(f"answers changed by stripping whitespace: {dict(stripped)}")
    print(f"answers removed: {len(removed)}")
    for item in removed:
        print(f"   {item['prompt_id']}  {item['llm_name']}  ({item['why']})")
    print(f"prompts removed because an answer was removed: {len(incomplete)} {incomplete}")
    print(f"\n{'split':6s} {'prompts':>7s} {'rows':>6s} {'per model':>9s} {'cut off':>8s}  prompts per task")
    for split, c in counts.items():
        print(f"{split:6s} {c['prompts']:7d} {c['rows']:6d} {c['rows'] // len(models):9d} {c['cut_off_rows']:8d}  "
              f"{dict(sorted(c['prompts_per_task'].items()))}")
    print(f"\nwrote {report['rows_written']} rows to {args.out_dir}; no prompt is in two splits; classes are balanced")


if __name__ == "__main__":
    main()
