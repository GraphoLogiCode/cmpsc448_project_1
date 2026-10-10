"""Quality checks on the generated responses.

    python src/check_responses.py

Reads   data/responses/<model>.jsonl (trial files are skipped)
Writes  results/response_stats.json, and prints the same numbers

It answers the questions that decide how build_dataset.py has to clean the data: is every prompt
answered by every model, how many answers are empty or cut off, how long they are, and how often
an answer gives its author away by naming a model or company.
"""
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["coding", "math", "factual", "writing"]

# Names that would let a classifier read the label off the text.
NAMES = {"Llama / Meta": r"\bllama\b|\bmeta ai\b|\bmeta\b(?= (?:ai|platforms))",
         "Qwen / Alibaba": r"\bqwen\b|\balibaba\b|\btongyi\b",
         "Mistral": r"\bmistral\b",
         "other model or company": r"\bopenai\b|\bchatgpt\b|\bgpt-?[34]\b|\banthropic\b|\bclaude\b|\bgemini\b|\bbard\b"}
AI_PHRASE = r"\bas an ai\b|\bi(?: a|')m an ai\b|\blanguage model\b|\bai assistant\b"
REFUSAL = r"\bi can(?:'|no)t (?:help|assist|provide|fulfill|answer)\b|\bi(?: a|')m sorry,? but\b|\bi(?: a|')m (?:not able|unable) to\b"
FORMATS = {"bold (**)": r"\*\*[^*\n]+\*\*", "heading (#)": r"(?m)^#{1,6} ", "bullet list": r"(?m)^\s*[-*] ",
           "numbered list": r"(?m)^\s*\d+[.)] ", "code block": r"```", "LaTeX": r"\\\(|\\\[|\\frac|\\boxed|\$[^$\n]+\$"}
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def share(rows, test):
    return round(sum(bool(test(r)) for r in rows) / len(rows), 4) if rows else 0.0


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def describe(rows):
    """Numbers for one group of answers (one model, or one model and task)."""
    text = lambda r: r["llm_output"]
    words = [len(text(r).split()) for r in rows]
    tokens = [r["output_tokens"] for r in rows]
    return {
        "rows": len(rows),
        "empty": sum(not text(r).strip() for r in rows),
        "cut_off_at_token_limit": share(rows, lambda r: r["finish"] == "length"),
        "tokens_median": statistics.median(tokens), "tokens_mean": round(statistics.mean(tokens), 1),
        "words_median": statistics.median(words), "words_p10": percentile(words, 0.10),
        "words_p90": percentile(words, 0.90),
        "format": {name: share(rows, lambda r: re.search(p, text(r))) for name, p in FORMATS.items()},
        "names": {name: sum(bool(re.search(p, text(r), re.I)) for r in rows) for name, p in NAMES.items()},
        "says_it_is_an_ai": sum(bool(re.search(AI_PHRASE, text(r), re.I)) for r in rows),
        "opens_with_refusal": sum(bool(re.search(REFUSAL, text(r)[:200], re.I)) for r in rows),
        "contains_cjk": sum(bool(CJK.search(text(r))) for r in rows),
    }


def main():
    files = sorted(f for f in (ROOT / "data" / "responses").glob("*.jsonl") if not f.stem.endswith("_trial"))
    data = {}
    for f in files:
        rows = read_jsonl(f)
        data[rows[0]["llm_name"]] = rows
    stats = {"models": {}, "across_models": {}}

    for model, rows in data.items():
        ids = Counter(r["prompt_id"] for r in rows)
        entry = {"all": describe(rows), "duplicate_prompt_ids": sum(c - 1 for c in ids.values()),
                 "by_task": {t: describe([r for r in rows if r["task"] == t]) for t in TASKS
                             if any(r["task"] == t for r in rows)},
                 "seconds_per_answer_median": statistics.median(r["seconds"] for r in rows),
                 "most_common_openings": Counter(" ".join(r["llm_output"].split()[:2]) for r in rows).most_common(5)}
        stats["models"][model] = entry

    answered = [{r["prompt_id"] for r in rows if r["llm_output"].strip()} for rows in data.values()]
    by_prompt = {}
    for model, rows in data.items():
        for r in rows:
            by_prompt.setdefault(r["prompt_id"], {})[model] = " ".join(r["llm_output"].split())
    identical = sum(1 for answers in by_prompt.values()
                    if len(answers) > 1 and len(set(answers.values())) < len(answers))
    stats["across_models"] = {"prompts_answered_by_every_model": len(set.intersection(*answered)) if answered else 0,
                              "prompts_where_two_models_gave_the_same_text": identical}

    out = ROOT / "results" / "response_stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")

    for model, entry in stats["models"].items():
        a = entry["all"]
        print(f"\n== {model}: {a['rows']} rows, {a['empty']} empty, {entry['duplicate_prompt_ids']} duplicate prompt ids, "
              f"{a['cut_off_at_token_limit']:.1%} cut off, median {a['words_median']:.0f} words "
              f"({a['tokens_median']:.0f} tokens), {entry['seconds_per_answer_median']:.2f} s per answer")
        print(f"   {'task':8s} {'rows':>5s} {'cut off':>8s} {'median words':>13s} {'p10':>5s} {'p90':>5s}  "
              + "  ".join(f"{name:>13s}" for name in FORMATS))
        for task, d in entry["by_task"].items():
            print(f"   {task:8s} {d['rows']:5d} {d['cut_off_at_token_limit']:8.1%} {d['words_median']:13.0f} "
                  f"{d['words_p10']:5d} {d['words_p90']:5d}  "
                  + "  ".join(f"{d['format'][name]:13.1%}" for name in FORMATS))
        print(f"   {'all':8s} {a['rows']:5d} {a['cut_off_at_token_limit']:8.1%} {a['words_median']:13.0f} "
              f"{a['words_p10']:5d} {a['words_p90']:5d}  " + "  ".join(f"{a['format'][name]:13.1%}" for name in FORMATS))
        print(f"   names mentioned (answers): {a['names']} | says it is an AI: {a['says_it_is_an_ai']} | "
              f"opens with a refusal: {a['opens_with_refusal']} | contains CJK characters: {a['contains_cjk']}")
        print(f"   most common first two words: {entry['most_common_openings']}")
    print(f"\nacross models: {stats['across_models']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
