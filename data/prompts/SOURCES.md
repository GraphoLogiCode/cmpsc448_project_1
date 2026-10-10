# Where the prompts come from

`prompts.jsonl` holds 2,000 prompts, 500 per task type. It is built by `src/collect_prompts.py` with seed 448; running the script again gives the same file. Each row has `id`, `task`, `source`, `license`, `origin` and `text`. `origin` names the file and lines (or the line of the source file) a prompt was taken from.

| Task | Rows | Source | Licence | How rows were chosen |
| --- | --- | --- | --- | --- |
| coding | 500 | C++ functions from four game engines, 125 each | MIT | 8 to 60 lines, at most 2,500 characters, at most 2 functions per file |
| math | 500 | GSM8K, test split | MIT | random sample of its 1,319 questions |
| factual | 500 | Dolly-15k, category `open_qa` | CC BY-SA 3.0 | instruction of 15 to 300 characters |
| writing | 500 | Dolly-15k, category `creative_writing` | CC BY-SA 3.0 | instruction of 15 to 600 characters |

Prompts that differ only in case or spacing were treated as duplicates and kept once.

## Coding prompts

Every coding prompt is the same instruction followed by one function:

> Review the following C++ function from a game engine. Explain briefly what it does and point out any problems you see.

The functions were extracted for a separate research project on code smells in game engines. Source trees, as cloned on 7 October 2026:

| Engine | Repository | Commit | Licence file (copy in `licenses/`) |
| --- | --- | --- | --- |
| Godot | https://github.com/godotengine/godot | `3ea0cf3e` | `godot-LICENSE.txt` |
| OGRE | https://github.com/OGRECave/ogre | `b74c0615` | `ogre-LICENSE.txt` |
| Torque3D | https://github.com/TorqueGameEngines/Torque3D | `4c44642a` | `Torque-LICENSE.md` |
| Esoterica | https://github.com/BobbyAnguelov/Esoterica | `10f3f616` | `Esoterica-LICENSE.md` |

Engines carry some files adapted from other projects, which keep their own authors' notices. A function is used only if its file's header names the engine's own copyright holder and nobody else (Esoterica's files have no header; its repository licence covers them). Folders with vendored libraries are skipped. The check reads file headers; it is not a legal review of every file.

## Downloaded files

| File | From | Bytes | SHA-256 starts with |
| --- | --- | --- | --- |
| `data/sources/gsm8k/test.jsonl` | https://github.com/openai/grade-school-math (`grade_school_math/data/test.jsonl`) | 749,738 | `3730d312f6e34405` |
| `data/sources/dolly/databricks-dolly-15k.jsonl` | https://huggingface.co/datasets/databricks/databricks-dolly-15k | 13,085,339 | `2df9083338b4abd6` |

Both were downloaded on 10 October 2026 and are not kept in git.

## Attribution

- **Dolly-15k:** Copyright (2023) Databricks, Inc. Licensed under CC BY-SA 3.0. Rows with source `dolly-15k` remain under that licence.
- **GSM8K:** OpenAI, MIT licence.
- **Engines:** MIT licence; the four licence files are in `licenses/`.

## Known limits

- The task of each prompt is the category its source gave it. Nobody checked the rows by hand: some `open_qa` rows ask for an opinion, and some `creative_writing` rows are plain questions.
- Prompt length differs strongly by task (median 612 characters for coding, 41 for factual).
