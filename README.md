

CMPSC 448, Project 1 (midterm). Deadline: 11:59 pm, 11 October 2026.

Team: Benjamin Park

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
  generate_responses.py   send the same prompts to every LLM, save what each writes
  build_dataset.py        clean the responses, tag the task type, split by prompt
  data.py                 tokeniser, vocabulary, padding, DataLoader
  models.py               the CNN and the LSTM
  train.py                training loop with validation and early stopping
  evaluate.py             accuracy, macro F1, confusion matrix
  baseline.py             TF-IDF + logistic regression
  experiments.py          the runs for RQ2, RQ3 and RQ4
data/
  prompts/                the prompts, with their task type and source
  responses/              one file per LLM: (llm_name, llm_input, llm_output)
  splits/                 train / validation / test
results/                  metrics, confusion matrices, figures
report/                   the report PDF
```

Every file under `src/` is an empty stub that describes its job. Nothing is implemented yet.

## Dataset

Rows of `(llm_name, llm_input, llm_output)` from at least three LLM families, with about the same number of rows per LLM. The assignment asks for the sources to be documented, so fill these in as they are decided:

- **Prompts:** _where they come from, their licence, how they were selected_
- **LLMs:** _name, version and family of each, and how it was run_
- **Generation settings:** _system prompt, temperature, token limit_

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
