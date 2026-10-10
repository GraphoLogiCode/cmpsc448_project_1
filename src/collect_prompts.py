"""Build data/prompts/prompts.jsonl: the prompts that every LLM will answer.

    python src/collect_prompts.py

Four task types, the same number of prompts each:
  coding   C++ functions from four MIT-licensed game engines, wrapped in one fixed review instruction
  math     GSM8K test questions
  factual  Dolly-15k, category open_qa
  writing  Dolly-15k, category creative_writing

Each output row has: id, task, source, license, origin, text.
data/prompts/SOURCES.md says where the input files come from and under which licence.

The coding prompts need two folders from the research project, the extracted functions (--units-dir)
and the engine source trees (--engines-root). The other two inputs are in data/sources/.
"""
import argparse
import json
import random
import re
import statistics
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Engines whose top-level licence file is MIT, with the copyright holder their source files name.
# Esoterica's files carry no header; its repository licence covers them.
# The functions were extracted by the research project.
ENGINES = {"godot": r"godot engine|juan linietsky", "ogre": r"torus knot", "Torque": r"garagegames",
           "Esoterica": None}
DATED_COPYRIGHT = re.compile(r"(copyright|\(c\)|©)[^\n]{0,12}\d{4}", re.I)
VENDOR_HINTS = ("thirdparty", "third_party", "3rdparty", "external", "vendor", "/lib/", "contrib")
CODE_INSTRUCTION = ("Review the following C++ function from a game engine. "
                    "Explain briefly what it does and point out any problems you see.")
CODE_LINES = (8, 60)          # shortest and longest function, in lines
CODE_MAX_CHARS = 2500
FUNCTIONS_PER_FILE = 2
FACTUAL_CHARS = (15, 300)     # shortest and longest instruction, in characters
WRITING_CHARS = (15, 600)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def key(text):
    """Texts that differ only in case or spacing count as the same prompt."""
    return " ".join(text.lower().split())


def own_code_only(source_file, holder):
    """True if the file's header names the engine's own copyright holder and nobody else.

    Engines carry some files adapted from other projects (Bullet, rapidjson, ...). Those keep their
    authors' notices, which the engine's licence file does not reproduce, so they are left out.
    """
    if holder is None:
        return True
    head = source_file.read_text(encoding="utf-8", errors="replace")[:4000]
    notices = [line for line in head.splitlines() if DATED_COPYRIGHT.search(line)]
    return bool(notices) and all(re.search(holder, line, re.I) for line in notices)


def coding_prompts(units_dir, engines_root, n, seed):
    per_engine, rest = divmod(n, len(ENGINES))
    assert rest == 0, f"--per-task must be a multiple of {len(ENGINES)}"
    out = []
    for engine, holder in ENGINES.items():
        rows = read_jsonl(units_dir / engine / "units.jsonl")
        random.Random(f"{seed}-coding-{engine}").shuffle(rows)
        picked, seen, per_file, own = [], set(), Counter(), {}
        for r in rows:
            path = r["path"].replace("\\", "/")
            lines = r["end_line"] - r["start_line"] + 1
            # The stored code starts at the function name; put the first line's indent back, then remove
            # the indent the whole function shares.
            code = textwrap.dedent(r["indent"] + r["code"]).strip("\n")
            if not CODE_LINES[0] <= lines <= CODE_LINES[1] or len(code) > CODE_MAX_CHARS:
                continue
            if any(hint in "/" + path.lower() for hint in VENDOR_HINTS):
                continue
            if key(code) in seen or per_file[path] >= FUNCTIONS_PER_FILE:
                continue
            if path not in own:
                own[path] = own_code_only(engines_root / engine / path, holder)
            if not own[path]:
                continue
            seen.add(key(code))
            per_file[path] += 1
            picked.append({"source": engine, "license": "MIT",
                           "origin": f"{path}:{r['start_line']}-{r['end_line']} {r['name']}",
                           "text": f"{CODE_INSTRUCTION}\n\n```cpp\n{code}\n```"})
            if len(picked) == per_engine:
                break
        assert len(picked) == per_engine, f"{engine}: only {len(picked)} functions pass the filters"
        out += picked
    return out


def sample(candidates, n, seed, name):
    """n distinct prompts from (origin, text) pairs, in a fixed random order."""
    unique, seen = [], set()
    for origin, text in candidates:
        if key(text) not in seen:
            seen.add(key(text))
            unique.append((origin, text))
    assert len(unique) >= n, f"{name}: only {len(unique)} distinct prompts, {n} wanted"
    return random.Random(f"{seed}-{name}").sample(unique, n)


def math_prompts(sources_dir, n, seed):
    rows = read_jsonl(sources_dir / "gsm8k" / "test.jsonl")
    candidates = [(f"test.jsonl line {i}", r["question"].strip()) for i, r in enumerate(rows, 1)]
    return [{"source": "gsm8k", "license": "MIT", "origin": origin, "text": text}
            for origin, text in sample(candidates, n, seed, "math")]


def dolly_prompts(sources_dir, category, chars, n, seed, name):
    rows = read_jsonl(sources_dir / "dolly" / "databricks-dolly-15k.jsonl")
    candidates = [(f"databricks-dolly-15k.jsonl line {i}, category {category}", r["instruction"].strip())
                  for i, r in enumerate(rows, 1)
                  if r["category"] == category and not r["context"].strip()
                  and chars[0] <= len(r["instruction"].strip()) <= chars[1]]
    return [{"source": "dolly-15k", "license": "CC BY-SA 3.0", "origin": origin, "text": text}
            for origin, text in sample(candidates, n, seed, name)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--units-dir", type=Path, default=ROOT.parent / "DeepSeek_Fine_Tuning" / "data" / "collect",
                    help="the research project's folder of extracted functions (<engine>/units.jsonl)")
    ap.add_argument("--engines-root", type=Path, default=Path("D:/dev/game engine source code"),
                    help="the engine source trees the functions were extracted from (<engine>/<path>)")
    ap.add_argument("--sources-dir", type=Path, default=ROOT / "data" / "sources")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "prompts" / "prompts.jsonl")
    ap.add_argument("--per-task", type=int, default=500)
    ap.add_argument("--seed", type=int, default=448)
    args = ap.parse_args()

    tasks = {
        "coding": coding_prompts(args.units_dir, args.engines_root, args.per_task, args.seed),
        "math": math_prompts(args.sources_dir, args.per_task, args.seed),
        "factual": dolly_prompts(args.sources_dir, "open_qa", FACTUAL_CHARS, args.per_task, args.seed, "factual"),
        "writing": dolly_prompts(args.sources_dir, "creative_writing", WRITING_CHARS, args.per_task, args.seed,
                                 "writing"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for task, prompts in tasks.items():
            for i, p in enumerate(prompts, 1):
                row = {"id": f"{task}-{i:04d}", "task": task, **p}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{'task':8s} {'prompts':>7s} {'median chars':>12s} {'shortest':>8s} {'longest':>8s}  sources")
    for task, prompts in tasks.items():
        lengths = [len(p["text"]) for p in prompts]
        sources = ", ".join(f"{s} {c}" for s, c in Counter(p["source"] for p in prompts).items())
        print(f"{task:8s} {len(prompts):7d} {statistics.median(lengths):12.0f} {min(lengths):8d} {max(lengths):8d}  "
              f"{sources}")
    print(f"wrote {sum(len(p) for p in tasks.values())} prompts to {args.out}")


if __name__ == "__main__":
    main()
