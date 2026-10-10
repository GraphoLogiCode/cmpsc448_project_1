# Building the dataset: LLMs, prompts and generated responses

Technical report for review, 10 October 2026. These are working notes on how the dataset was built and what is in it. They are not the course report. Where I could not check something, the text says so.

## 1. Summary

The raw dataset now exists: 6,000 answers, from three LLM families to the same 2,000 prompts.

| | Result |
| --- | --- |
| LLMs | `llama3.1:8b` (Meta), `qwen2.5:7b` (Alibaba), `mistral:7b` (Mistral AI), run locally with Ollama |
| Prompts | 2,000: 500 each for coding, math, factual and writing |
| Answers | 6,000: every model answered every prompt; none empty, none failed, no duplicates |
| Generation time | 65 minutes on one RTX 5090 |
| Files | `data/responses/<model>.jsonl` (11.2 MB in total), settings in `data/responses/settings.json` |

With three balanced classes, guessing gives 33.3% accuracy.

The raw answers are not fit for training as they are. Section 6 lists what has to be cleaned; the most important item is that **every Mistral answer begins with a space character**, which would let a classifier read the label from the first character. `src/build_dataset.py` now does that cleaning and writes the splits to `data/splits/`: 4,194 training, 900 validation and 900 test rows.

Nothing here has been committed to git.

## 2. The LLMs

### What was chosen

You chose three families. All were pulled from Ollama's registry on 10 October 2026.

| Ollama name | Maker | Parameters | Build | Download | Digest (start) | Licence stated on its Ollama page |
| --- | --- | --- | --- | --- | --- | --- |
| `llama3.1:8b` | Meta | 8.0B | Q4_K_M | 4.9 GB | `46e0c10c039e` | Not named there. Meta's Llama 3.1 community licence, as far as I know (not verified today). |
| `qwen2.5:7b` | Alibaba | 7.6B | Q4_K_M | 4.7 GB | `845dbda0ea48` | Apache 2.0 |
| `mistral:7b` | Mistral AI | 7.2B | Q4_K_M | 4.4 GB | `6577803aa9a0` | Apache; the tag is version 0.3 |

Q4_K_M is a 4-bit compressed build of the model. The answers are those of these compressed local builds, which may differ from what the same models produce through a hosted service.

### Why these three

- **Three different makers**, so the labels are real families and not versions of one model.
- **Similar size** (7B to 8B). A classifier that separates a 3B model from a 30B one may only be learning answer quality.
- **Plain chat models.** None writes out a reasoning stage before its answer.
- **Widely used.** Each is the most downloaded model of its family on Ollama (120M, 42M and 34M pulls).

### What was considered and left out

| Model | Why not |
| --- | --- |
| `gemma3:12b` (Google, 8.2 GB) | A good fourth family. Left out because you chose three. |
| `phi4:14b` (Microsoft, 9.1 GB) | Twice the size of the others. |
| `deepseek-r1` (7b, 14b, …) | Ollama lists the small tags as distilled onto Qwen models, so they would blur with the Qwen class. |
| `qwen3`, `qwen3.5`, `gpt-oss` | Reasoning models that produce a thinking stage. I did not test them. |
| `deepseek-coder` | The base model of your research, but built for code only. |

Three is the assignment's minimum, so the dataset has no spare class. All three models produced usable output, so that risk did not materialise.

## 3. The prompts

`data/prompts/prompts.jsonl` holds 2,000 prompts, built by `src/collect_prompts.py` with a fixed seed. Running the script again gives the identical file.

| Task | Rows | Source | Licence | Median length |
| --- | --- | --- | --- | --- |
| coding | 500 | C++ functions from Godot, OGRE, Torque3D and Esoterica (125 each), from your research data | MIT | 612 characters |
| math | 500 | GSM8K test questions | MIT | 225 |
| factual | 500 | Dolly-15k, category `open_qa` | CC BY-SA 3.0 | 41 |
| writing | 500 | Dolly-15k, category `creative_writing` | CC BY-SA 3.0 | 69 |

Each coding prompt is one fixed sentence ("Review the following C++ function from a game engine. Explain briefly what it does and point out any problems you see.") followed by the function. This is the link to your code-smell research.

**Licence filtering of the code.** These four are the engines whose top-level licence file is plain MIT. Of the other six in your research, Spring is GPL, Defold has its own licence, and I did not confirm Cafu, Nebula, cocos2d-x and Irrlicht. Inside the four, some files are adapted from other projects and keep other authors' notices. The script uses a function only if its file header names the engine's own copyright holder and nobody else. That removes 23 Godot files, 92 OGRE files and 465 Torque files from the pool. It is a header check, not a legal review.

**Checks on the prompts**

- 2,000 rows, 500 per task, unique ids, no two prompts with the same text.
- No email addresses or phone numbers in any prompt. I read samples of each task, not every row.

**Limits**

- The task labels are the sources' own categories, and Dolly's are loose. Some `open_qa` rows ask for an opinion, and some `creative_writing` rows are plain questions.
- Dolly rows stay under CC BY-SA 3.0, so the attribution in `data/prompts/SOURCES.md` must stay with the file.

Full provenance and download checksums are in `data/prompts/SOURCES.md`.

## 4. Generating the answers

### How it works

`src/generate_responses.py` sends each prompt to each model through Ollama's chat interface on the local machine and appends the answer to that model's file. It uses only the Python standard library. If it is stopped, running it again skips the prompts that already have an answer.

Each row has these fields:

| Field | Meaning |
| --- | --- |
| `llm_name` | The Ollama model name; this is the label to predict |
| `llm_family` | The family and maker |
| `prompt_id`, `task` | Which prompt, and its task type |
| `llm_input` | The prompt text |
| `llm_output` | The model's answer, exactly as returned |
| `finish` | `stop` if the model ended the answer itself, `length` if it was cut off at the token limit |
| `output_tokens`, `seconds` | Length of the answer in tokens, and how long that request took |

### Settings

Every model was run with identical settings. They are recorded in `data/responses/settings.json`.

| Setting | Value | Why |
| --- | --- | --- |
| System prompt | "You are a helpful assistant." | See below |
| Temperature | 0.7 | Sampling gives natural variety |
| Seed | 448, the same for every request | Makes the run repeatable in principle; I did not test that a second run gives the same text |
| Longest answer | 600 tokens | See below |
| Context window | 4,096 tokens | The longest prompt plus answer fits |
| Answers per prompt | One | |
| Requests in parallel | 4 | Speed only |
| Software | Ollama 0.32.4 | |

**Why an explicit system prompt.** If no system prompt is sent, Ollama uses the one each model was packaged with. I checked all three:

| Model | Packaged system prompt |
| --- | --- |
| `llama3.1:8b` | none |
| `qwen2.5:7b` | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." |
| `mistral:7b` | none |

Without an explicit prompt, Qwen alone would have been told its own name in every request. Sending the same neutral sentence to all three removes that difference.

**Why 600 tokens.** The report I wrote before generation suggested 300. Two short trials changed that:

| Token limit | Prompts per model | Llama cut off | Qwen cut off | Mistral cut off |
| --- | --- | --- | --- | --- |
| 300 | 20 | 50% | 75% | 65% |
| 600 | 40 | 10% | 35% | 10% |

At 300 most answers lost their ending. Answers can be shortened later at no cost, but not lengthened, so I used 600.

### The run

All times are local, 10 October 2026.

| Step | Duration | Notes |
| --- | --- | --- |
| Download the three models | 23 minutes | 14.0 GB |
| `llama3.1:8b`, 2,000 prompts | 20.6 minutes | 0.62 s per answer |
| `qwen2.5:7b`, 2,000 prompts | 26.6 minutes | 0.80 s per answer |
| `mistral:7b`, 2,000 prompts | 18.0 minutes | 0.54 s per answer |

The run started at 12:52 am and ended at 1:57 am. No request failed and none had to be retried. The three models wrote 2.0 million tokens in total.

I started the Ollama server myself for this run, because the Ollama app would not start from my shell, and stopped it afterwards.

## 5. What is in the answers

The numbers in this section come from `src/check_responses.py`, which also writes them to `results/response_stats.json`. I looked at the individual rows behind every count of names, refusals and non-English text.

### Completeness

| Check | Result |
| --- | --- |
| Rows per model | 2,000 each |
| Rows per model and task | 500 each |
| Empty answers | 0 |
| Prompts answered more than once by a model | 0 |
| Prompts answered by all three models | 2,000 of 2,000 |

### Length and cut-off

| | Llama | Qwen | Mistral |
| --- | --- | --- | --- |
| Median answer, words | 197 | 254 | 208 |
| Median answer, tokens | 281 | 372 | 320 |
| Total words written | 410,714 | 517,143 | 420,611 |
| Cut off at 600 tokens | 86 (4.3%) | 598 (29.9%) | 112 (5.6%) |

By task, median words with the share cut off in brackets:

| Task | Llama | Qwen | Mistral |
| --- | --- | --- | --- |
| coding | 227 (1.0%) | 373 (91.0%) | 302 (6.0%) |
| math | 106 (0.2%) | 175 (0.8%) | 130 (0.6%) |
| factual | 164 (2.0%) | 172 (4.0%) | 103 (2.0%) |
| writing | 318 (14.0%) | 313 (23.8%) | 268 (13.8%) |

Qwen writes the longest answers, and its code reviews almost always run past 600 tokens. For 1,346 of the 2,000 prompts no model was cut off; for 29 all three were.

### Formatting

Share of each model's answers that use the format at least once:

| Format | Llama | Qwen | Mistral |
| --- | --- | --- | --- |
| Bold (`**text**`) | 51.4% | 63.9% | 2.4% |
| Headings (`#`) | 0.7% | 34.5% | 0.1% |
| Bullet list | 24.9% | 52.0% | 9.8% |
| Numbered list | 53.0% | 71.8% | 52.3% |
| Code block | 7.8% | 21.6% | 0.7% |
| LaTeX maths | 9.1% | 26.1% | 7.1% |

The differences are large and they depend on the task. Llama uses bold in every code review but in 3% of its maths answers. Qwen uses LaTeX in every maths answer and headings in 99% of its code reviews. Mistral writes almost entirely plain text.

### Character-level habits

| Habit | Llama | Qwen | Mistral |
| --- | --- | --- | --- |
| Answer begins with a space | 0 | 1 | 2,000 |
| Answer ends with whitespace | 6 | 89 | 2 |
| Uses curly quotes | 0 | 457 | 1 |
| Uses an em dash | 1 | 84 | 13 |
| Contains Chinese, Japanese or Korean characters | 0 | 13 | 0 |

### Most common opening words

| Model | First two words (count of 2,000) |
| --- | --- |
| Llama | "Let's break" (255), "To find" (150), "**Function Review**" (118) |
| Qwen | "To determine" (466), "The provided" (180), "### Explanation" (135) |
| Mistral | "This C++" (448), "First, let's" (177), "To find" (161) |

### Does an answer name its author?

| | Llama | Qwen | Mistral |
| --- | --- | --- | --- |
| Names its own maker or model as its author | 0 | 1 | 0 |
| Refers to itself as an AI ("As an AI…") | 0 | 12 | 3 |
| Mentions any model or AI company name | 3 | 10 | 6 |

- **One real leak.** Asked "What is your political stance?", Qwen answered "As an AI developed by Alibaba Cloud…", despite the neutral system prompt (prompt `factual-0322`).
- **The other name mentions are about the topic, not the author:** the star sign Gemini in horoscopes, the painter Claude Monet, Alibaba as a cloud provider, OpenAI in an answer about foundation models.

### Other things found

- **Refusals.** Nine answers open with a refusal phrase (Llama 4, Mistral 3, Qwen 2). Two of Llama's are bare refusals with no content ("I can't fulfill that request."), both on writing prompts.
- **Chinese text.** Thirteen Qwen answers contain Chinese characters part-way through an English answer, all on factual or writing prompts. No prompt contains Chinese.
- **Identical answers.** For two prompts ("What is the capital of Arizona?", "what is the capital of China?") two models gave exactly the same sentence. Those rows cannot be told apart by any classifier.
- **One very short answer:** Llama answered a factual prompt with "Mrs. Doubtfire."

## 6. What has to be cleaned before training

These are decisions for `src/build_dataset.py`. The raw files should stay as they are, so that every cleaning step can be reported.

| Issue | Rows | Suggested handling | If ignored |
| --- | --- | --- | --- |
| Leading space on Mistral answers | 2,000 | Strip whitespace from both ends of every answer | A character-level model identifies Mistral from the first character. This comes from how the model is served, not from its writing. |
| Answer names its maker | 1 | Remove the row, or replace the name | A direct label leak, though a single row |
| "As an AI…" self-reference | 15 | Keep, and report the count | Not a name, but concentrated in Qwen |
| Cut-off answers | 796 | Keep and report; for a length-controlled experiment, cut every answer to the same number of tokens | "Ends mid-sentence" marks Qwen's code reviews |
| Refusals | 9 | Keep; they are real model behaviour | Negligible |
| Identical answers across models | 4 rows | Keep or drop; say which | Negligible |

Then split **by prompt**, so that all three answers to one prompt land in the same split.

### What `src/build_dataset.py` does (added later on 10 October)

The script now exists and has been run. It follows the table above:

| Step | Result |
| --- | --- |
| Assign each prompt to a split, 70% / 15% / 15% within each task, from the full prompt list with seed 448 | 1,400 / 300 / 300 prompts before cleaning |
| Strip whitespace from both ends of every answer | Changed 2,000 Mistral, 90 Qwen and 6 Llama answers |
| Drop an answer that is empty, or that contains its own model's or maker's name when the prompt did not | 2 Qwen answers: `factual-0322` (the leak) and `writing-0282` (names Alibaba as an online shop) |
| Drop a prompt if any model lost its answer | 2 prompts, so 6 rows in total |

| Split | Prompts | Rows | Rows per model | Rows cut off at the token limit |
| --- | --- | --- | --- | --- |
| train | 1,398 | 4,194 | 1,398 | 557 |
| validation | 300 | 900 | 300 | 106 |
| test | 300 | 900 | 300 | 132 |

The second dropped answer is not a real leak, but the rule is deliberately simple: any answer carrying its own label's name goes. The cost is one extra prompt.

Checks, made by the script and again by a separate check against the raw files: no prompt is in two splits, every prompt has exactly three answers, the classes are exactly balanced in every split and task, no answer begins or ends with whitespace, and all 5,994 answers are otherwise identical to the raw text. Running the script twice gives the same files.

Cut-off answers, refusals and the two pairs of identical answers were kept. What was changed and removed is recorded in `data/splits/split_report.json`.

## 7. What this means for the research questions

The table gives expectations from looking at the data, written before any classifier was trained.

| Question | What the data suggests |
| --- | --- |
| RQ1 | The models differ visibly in formatting, opening words and length, so I expect the task to be easy and accuracy well above 33.3%. A simple baseline may already do well, which makes the comparison between the baseline, the CNN and the LSTM worth showing. |
| RQ2 | Every model received the same prompts, so a prompt-only classifier must score at chance. Anything higher means a leak, usually a split by row. |
| RQ3 | Formatting changes with the task (Llama's bold, Qwen's LaTeX and headings). Holding out one task is therefore a real test of whether a classifier learned the model or the task's formatting. Coding, from your research, is the natural task to hold out. |
| RQ4 | Section 5 gives the leads: markdown, opening words, length, curly quotes. The analysis itself is still to do: remove one signal at a time (strip the markdown, cut answers to equal length, drop the first sentence) and measure how much accuracy falls. |

### First numbers from the baseline (added later on 10 October)

`src/baseline.py` trains TF-IDF + logistic regression on the splits. Accuracy on the 900 test rows:

| Input | Words and punctuation as tokens | Words only |
| --- | --- | --- |
| Prompt only | 33.3% | 33.3% |
| Answer only | 88.3% | 83.4% |
| Prompt + answer | 87.8% | 81.3% |

- **The prompt alone is exactly at chance**, as the design requires: the same prompt text appears once per model, so any prediction is right for one row in three.
- **The answer alone identifies the model 88% of the time**, so the task is easy, as expected.
- **Dropping punctuation and markdown costs about five points**, so most of the signal is in the words.
- **It depends on the task:** 100% on coding, 93% on math, 84% on writing and 76% on factual answers (answer only, with punctuation).

These are the numbers the CNN and the LSTM have to beat. Interpreting them for RQ2 and RQ4 is still to do.

## 8. Limits of this dataset

- **Three classes, the minimum allowed.**
- **Compressed local builds.** The fingerprints are those of 4-bit Ollama builds with one system prompt and one temperature. They may not match the same models served elsewhere.
- **One answer per prompt**, so nothing can be said about how much one model varies between runs.
- **English prompts only**, from four sources.
- **Loose task labels** for the two Dolly tasks (section 3).
- **Reproducibility of the answers is untested.** The seed is fixed, but four requests ran in parallel and I did not run the generation twice.
- **The outputs come from open-weight models run locally.** I did not review the three model licences for terms about publishing outputs; Qwen 7B and Mistral 7B are stated as Apache 2.0, and I did not verify Llama's licence text.

## 9. Open datasets found, as an alternative

These already contain LLM answers. I read their pages and file listings on Hugging Face and downloaded none of them. With the generated dataset complete they are no longer needed as a fallback, but they could serve as a second dataset.

| Dataset | Contents | Licence and access | Size |
| --- | --- | --- | --- |
| `lmarena-ai/arena-human-preference-55k` | Over 55,000 Chatbot Arena battles in which two LLMs answer the same user prompt. Over 70 models, including GPT-4, Claude 2, Llama 2, Gemini and Mistral. | Apache 2.0, open | 184 MB |
| `lmarena-ai/arena-human-preference-140k` | 135,634 votes with full conversations, an `is_code` flag and task tags. | Prompts CC BY 4.0; model outputs under each provider's terms. Open | 1.61 GB |
| `liamdugan/raid` | Text from 11 generators in 11 genres. | MIT, open | 11.8 GB for the training file |
| `lmsys/lmsys-chat-1m` | A large set of real conversations with many LLMs. | Gated: needs your name, email and affiliation, and a custom licence | not checked |

In the Arena data the prompts are written by real users and each is answered by two models only, so a prompt-only classifier may beat chance there. That would be a different experiment from this one.

## 10. How to repeat it

```bash
ollama pull llama3.1:8b
```

```bash
ollama pull qwen2.5:7b
```

```bash
ollama pull mistral:7b
```

```bash
python src/collect_prompts.py
```

```bash
python src/generate_responses.py
```

```bash
python src/check_responses.py
```

`collect_prompts.py` needs the two prompt source files in `data/sources/` and the research project's folders (see `data/prompts/SOURCES.md`). The other two scripts need only Python and a running Ollama server.

## 11. What is left

1. Write the baseline, then the CNN and the LSTM, and run RQ1 and RQ2.
2. Run RQ3 and RQ4 if time allows.
3. Write the course report. The scripts for collecting prompts, generating answers, checking them and building the splits were written by me; read them so that you can explain every step.

## Sources

- Ollama library: https://ollama.com/library and the pages for `llama3.1`, `qwen2.5`, `mistral`, `gemma3`, `phi4`, `deepseek-r1`
- https://huggingface.co/datasets/openai/gsm8k and https://github.com/openai/grade-school-math
- https://huggingface.co/datasets/databricks/databricks-dolly-15k
- https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k
- https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k
- https://huggingface.co/datasets/liamdugan/raid
- https://huggingface.co/datasets/lmsys/lmsys-chat-1m
