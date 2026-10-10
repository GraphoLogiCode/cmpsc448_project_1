"""Send every prompt to every LLM and save the answers.

    python src/generate_responses.py                # all prompts, all models
    python src/generate_responses.py --limit 20     # a short trial, to check the output and the speed

Reads   data/prompts/prompts.jsonl
Writes  data/responses/<model>.jsonl     one row per prompt: llm_name, llm_input, llm_output, ...
        data/responses/settings.json     the settings every model was run with

The models run locally through Ollama. Every model gets the same system prompt, temperature, seed
and token limit, so for one prompt the only thing that differs between the rows is the model.
An answer that reaches the token limit is kept and marked with finish = "length".

A run can be stopped and started again: prompts that already have an answer in the output file
are skipped. Uses only the standard library.
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {"llama3.1:8b": "Llama (Meta)", "qwen2.5:7b": "Qwen (Alibaba)", "mistral:7b": "Mistral (Mistral AI)"}
# Sent to every model, so that no model falls back on the system prompt it was packaged with.
SYSTEM_PROMPT = "You are a helpful assistant."
# num_predict is the longest answer in tokens. In a trial of 40 prompts per model, 600 cut off 10% of the
# answers of two models and 35% of the third; 300 cut off 50% to 75%. Shortening answers later is free,
# lengthening them is not.
OPTIONS = {"temperature": 0.7, "seed": 448, "num_predict": 600, "num_ctx": 4096}
ATTEMPTS = 3


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call(host, path, payload=None, timeout=600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(host + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask(host, model, prompt):
    """One answer from one model. Returns the row fields that come from the model."""
    payload = {"model": model, "stream": False, "options": OPTIONS,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            reply = call(host, "/api/chat", payload)
            return {"llm_output": reply["message"]["content"], "finish": reply.get("done_reason", ""),
                    "output_tokens": reply.get("eval_count", 0),
                    "seconds": round(reply.get("total_duration", 0) / 1e9, 2)}
        except Exception as error:   # connection reset, timeout, model still loading
            if attempt == ATTEMPTS:
                raise
            print(f"  retrying {model} after: {error}", flush=True)
            time.sleep(5 * attempt)


def run_model(host, model, family, prompts, out_file, workers):
    done = {row["prompt_id"] for row in read_jsonl(out_file)} if out_file.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{model}: {len(done)} answers already saved, {len(todo)} to generate", flush=True)
    if not todo:
        return
    started, finished, failed = time.time(), 0, 0

    def one(prompt):
        answer = ask(host, model, prompt["text"])
        return {"llm_name": model, "llm_family": family, "prompt_id": prompt["id"], "task": prompt["task"],
                "llm_input": prompt["text"], **answer}

    with open(out_file, "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(one, p) for p in todo]):
            try:
                row = future.result()
            except Exception as error:
                failed += 1
                print(f"  gave up on one prompt: {error}", flush=True)
                continue
            # Answers arrive here one at a time, so each is on disk before the next is handled.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            finished += 1
            if finished % 50 == 0 or finished == len(todo):
                per_answer = (time.time() - started) / finished
                print(f"  {model}: {finished}/{len(todo)}  {per_answer:.2f} s per answer  "
                      f"about {(len(todo) - finished) * per_answer / 60:.0f} min left", flush=True)
    print(f"{model}: finished in {(time.time() - started) / 60:.1f} min, {failed} prompts failed", flush=True)


def summary(out_file):
    rows = read_jsonl(out_file)
    tokens = [r["output_tokens"] for r in rows]
    return {"rows": len(rows), "empty": sum(not r["llm_output"].strip() for r in rows),
            "hit_token_limit": sum(r["finish"] == "length" for r in rows),
            "median_output_tokens": statistics.median(tokens) if tokens else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "prompts.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "responses")
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="Ollama model names")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--workers", type=int, default=4, help="requests sent to Ollama at the same time")
    ap.add_argument("--limit", type=int, help="use only the first N prompts of each task (for a trial)")
    ap.add_argument("--max-tokens", type=int, default=OPTIONS["num_predict"], help="longest answer, in tokens")
    args = ap.parse_args()
    OPTIONS["num_predict"] = args.max_tokens

    prompts = read_jsonl(args.prompts)
    if args.limit:
        seen, kept = {}, []
        for p in prompts:
            seen[p["task"]] = seen.get(p["task"], 0) + 1
            if seen[p["task"]] <= args.limit:
                kept.append(p)
        prompts = kept
    args.out_dir.mkdir(parents=True, exist_ok=True)

    installed = {m["name"]: m for m in call(args.host, "/api/tags")["models"]}
    missing = [m for m in args.models if m not in installed]
    if missing:
        raise SystemExit(f"not installed in Ollama: {', '.join(missing)} (run: ollama pull <name>)")

    settings = {"system_prompt": SYSTEM_PROMPT, "options": OPTIONS, "workers": args.workers,
                "prompts": len(prompts), "ollama_version": call(args.host, "/api/version")["version"],
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "models": {m: {"family": MODELS.get(m, ""), "digest": installed[m]["digest"],
                               "parameters": installed[m]["details"]["parameter_size"],
                               "quantization": installed[m]["details"]["quantization_level"]} for m in args.models}}
    settings_file = args.out_dir / ("settings.json" if not args.limit else "settings_trial.json")
    settings_file.write_text(json.dumps(settings, indent=1), encoding="utf-8")

    for model in args.models:
        suffix = "" if not args.limit else "_trial"
        out_file = args.out_dir / f"{model.replace(':', '_')}{suffix}.jsonl"
        run_model(args.host, model, MODELS.get(model, ""), prompts, out_file, args.workers)
        settings["models"][model]["result"] = summary(out_file)
        print(f"{model}: {settings['models'][model]['result']}", flush=True)
    settings["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    settings_file.write_text(json.dumps(settings, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
