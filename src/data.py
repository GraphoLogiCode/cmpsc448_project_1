"""Text to tensors: tokeniser, vocabulary, padding and data loaders.

    python src/data.py            # prints the vocabulary size, token counts and one batch, as a check

Use from train.py:

    from data import make_loaders
    data = make_loaders(mode="answer", max_len=512, batch_size=32)
    for tokens, lengths, labels in data["train"]:
        ...                       # tokens: (batch, time) ids, 0 = padding
                                  # lengths: (batch,) number of real tokens in each row
                                  # labels: (batch,) index into data["names"]

mode chooses the input, as RQ2 asks: "answer" (the LLM's output), "prompt" (the user's input) or
"both" (the prompt, a <sep> token, then the answer).

For RQ3, train_tasks and test_tasks restrict the splits to some task types, for example
make_loaders(train_tasks=["math", "factual", "writing"], test_tasks=["coding"]).

The vocabulary is built from the training split only, so validation and test words that were never
seen in training become <unk>, as they would for new text.
"""
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
MODES = ("answer", "prompt", "both")
PAD, UNK, SEP = 0, 1, 2
SPECIALS = ["<pad>", "<unk>", "<sep>"]
# A token is a word, or one punctuation or markdown character. Same rule as the TF-IDF baseline.
TOKEN = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return TOKEN.findall(text.lower())


def read_split(split, tasks=None):
    """The rows of one split, optionally only those of the given task types."""
    with open(ROOT / "data" / "splits" / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if tasks is None or r["task"] in tasks]


def row_tokens(row, mode, max_len, transform=None):
    """The tokens of one row for the chosen input, cut to max_len.

    transform, if given, rewrites the answer text first (used by rq4.py to remove one kind of signal).
    """
    answer = transform(row["llm_output"]) if transform else row["llm_output"]
    if mode == "answer":
        tokens = tokenize(answer)
    elif mode == "prompt":
        tokens = tokenize(row["llm_input"])
    elif mode == "both":
        # The prompt gets at most a third of the room, so a long prompt cannot push the answer out.
        tokens = tokenize(row["llm_input"])[:max_len // 3] + ["<sep>"] + tokenize(answer)
    else:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    return tokens[:max_len] or ["<unk>"]      # never an empty row: the LSTM cannot pack a text of length 0


class Vocabulary:
    """Token <-> id. Tokens seen fewer than min_count times in training map to <unk>."""

    def __init__(self, token_lists, min_count=2):
        counts = Counter(t for tokens in token_lists for t in tokens if t not in SPECIALS)
        kept = sorted((t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t))
        self.tokens = SPECIALS + kept
        self.ids = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        return [self.ids.get(t, UNK) for t in tokens]

    def decode(self, ids):
        return [self.tokens[i] for i in ids]


class TextDataset(Dataset):
    """One item per row: (token ids, label index)."""

    def __init__(self, rows, vocab, names, mode, max_len, transform=None):
        self.items = [(torch.tensor(vocab.encode(row_tokens(r, mode, max_len, transform)), dtype=torch.long),
                       names.index(r["llm_name"])) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    """Pad the rows of a batch to the longest one. Returns tokens, lengths, labels."""
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    tokens = torch.full((len(ids), int(lengths.max())), PAD, dtype=torch.long)
    for i, x in enumerate(ids):
        tokens[i, :len(x)] = x
    return tokens, lengths, torch.tensor(labels, dtype=torch.long)


def make_loaders(mode="answer", max_len=512, batch_size=32, min_count=2, train_tasks=None, test_tasks=None, seed=448,
                 transform=None):
    """Loaders for the three splits, with the vocabulary, the class names and the rows behind them.

    Returns a dict with keys "train", "val", "test" (DataLoader), "vocab", "names" and "rows"
    (the rows of each split, in the order the validation and test loaders yield them).
    train_tasks applies to the training and validation splits, test_tasks to the test split;
    both are kept in the dict so the evaluation can record them.
    """
    rows = {"train": read_split("train", train_tasks), "val": read_split("val", train_tasks),
            "test": read_split("test", test_tasks)}
    names = sorted({r["llm_name"] for r in read_split("train")})
    vocab = Vocabulary((row_tokens(r, mode, max_len, transform) for r in rows["train"]), min_count)
    sets = {s: TextDataset(rows[s], vocab, names, mode, max_len, transform) for s in SPLITS}
    shuffle = torch.Generator().manual_seed(seed)
    return {"train": DataLoader(sets["train"], batch_size, shuffle=True, generator=shuffle, collate_fn=collate),
            "val": DataLoader(sets["val"], batch_size, shuffle=False, collate_fn=collate),
            "test": DataLoader(sets["test"], batch_size, shuffle=False, collate_fn=collate),
            "vocab": vocab, "names": names, "rows": rows, "train_tasks": train_tasks, "test_tasks": test_tasks}


def main():
    for mode in MODES:
        full = [len(row_tokens(r, mode, 10**9)) for r in read_split("train")]
        full.sort()
        data = make_loaders(mode=mode)
        tokens, lengths, labels = next(iter(data["train"]))
        unknown = sum(int((t == UNK).sum()) for t, _ in data["val"].dataset.items)
        total = sum(len(t) for t, _ in data["val"].dataset.items)
        print(f"\nmode = {mode}")
        print(f"  vocabulary: {len(data['vocab']):,} tokens | classes: {data['names']}")
        print(f"  tokens per training row: median {full[len(full) // 2]}, 90th percentile {full[int(len(full) * 0.9)]}, "
              f"longest {full[-1]} | rows longer than 512: {sum(n > 512 for n in full) / len(full):.1%}")
        print(f"  unknown tokens in validation: {unknown / total:.1%}")
        print(f"  one batch: tokens {tuple(tokens.shape)}, lengths {tuple(lengths.shape)}, labels {tuple(labels.shape)}")
        print(f"  first row starts: {' '.join(data['vocab'].decode(tokens[0, :14].tolist()))}")


if __name__ == "__main__":
    main()
