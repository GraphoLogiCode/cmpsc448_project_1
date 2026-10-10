# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448, Project 1 (midterm). Deadline: 11:59 pm, 11 October 2026.

Team: _fill in the leader and members; the report PDF must list them_

## Question

Given a piece of text, can a classifier tell which LLM wrote it? Two models are required: a CNN and an RNN (LSTM).

| | Question | Status |
| --- | --- | --- |
| RQ1 (required) | Identify the LLM from its output | not started |
| RQ2 (required) | Compare input only, output only, and input + output | not started |
| RQ3 (+5) | Train on some task types, test on another | not started |
| RQ4 (+5) | Which characteristics distinguish the LLMs | not started |

The full assignment is in `assignment/` (kept out of git).

## Layout

```
src/
  collect_prompts.py      build data/prompts/prompts.jsonl from the four prompt sources
  generate_responses.py   send the same prompts to every LLM, save what each writes
  check_responses.py      quality checks on the answers; writes results/response_stats.json
  build_dataset.py        clean the responses, tag the task type, split by prompt
  data.py                 tokeniser, vocabulary, padding, DataLoader
  models.py               the CNN and the LSTM
  train.py                training loop with validation and early stopping
  evaluate.py             accuracy, macro F1, confusion matrix
  baseline.py             TF-IDF + logistic regression
  experiments.py          the runs for RQ2, RQ3 and RQ4
data/
  prompts/                prompts.jsonl, SOURCES.md (where each prompt comes from) and the licence files
  sources/                the downloaded prompt sources (kept out of git)
  responses/              one file per LLM: (llm_name, llm_input, llm_output), and settings.json
  splits/                 train / validation / test
docs/
  technical_report.md     how the dataset was built and what is in it
results/                  metrics, confusion matrices, figures
report/                   the report PDF
```

`collect_prompts.py`, `generate_responses.py`, `check_responses.py`, `build_dataset.py` and `baseline.py` are implemented. The raw dataset is in `data/responses/` (6,000 answers), the cleaned splits are in `data/splits/` (4,194 train, 900 validation, 900 test rows), and the baseline's results are in `results/baseline.json`. The other files under `src/` (`data.py`, `models.py`, `train.py`, `evaluate.py`, `experiments.py`) are stubs that describe their job.

## Dataset

Rows of `(llm_name, llm_input, llm_output)` from three LLM families, 2,000 rows per LLM. How it was built and what the checks found is in [docs/technical_report.md](docs/technical_report.md).

- **Prompts:** 2,000 prompts, 500 for each of four task types (coding, math, factual, writing). Sources, licences and selection rules are in [data/prompts/SOURCES.md](data/prompts/SOURCES.md).
- **LLMs:** three families, run locally with Ollama 0.32.4 on 10 October 2026: `llama3.1:8b` (Meta), `qwen2.5:7b` (Alibaba) and `mistral:7b` (Mistral AI), each as a 4-bit (Q4_K_M) build.
- **Generation settings:** the same for every model: system prompt "You are a helpful assistant.", temperature 0.7, seed 448, at most 600 tokens per answer, one answer per prompt. The exact values and model digests are in `data/responses/settings.json`.
- **Cleaning and split.** `build_dataset.py` strips whitespace from every answer (every raw Mistral answer begins with a space), removes the two prompts where Qwen's answer contains its maker's name, and splits by prompt, 70% / 15% / 15% within each task. What it changed is listed in `data/splits/split_report.json`. The files in `data/responses/` stay raw.

Four rules, each of which protects one of the research questions:

1. **Every LLM answers the same prompts.** Then the prompt alone cannot reveal the LLM, and input-only accuracy in RQ2 has to sit at chance. If it does not, something leaked.
2. **Split by prompt, not by row.** Otherwise the same prompt is in training and in the test set, and the test score is too high.
3. **Same settings for every LLM.** A different system prompt or token limit would be a fingerprint of the setup, not of the model.
4. **Remove self-identification** ("I am Qwen…") before training, and report how often it occurred.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```
