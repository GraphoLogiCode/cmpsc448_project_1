"""The two required models: a convolutional and a recurrent text classifier.

    TextCNN          embedding -> 1-D convolutions of several widths -> max over time -> dropout -> linear
    LSTMClassifier   embedding -> bidirectional LSTM -> max over time of its outputs -> dropout -> linear

Both are called the same way:

    scores = model(tokens, lengths)      # shape (batch, n_classes); the largest score is the prediction

    tokens   LongTensor (batch, time) of token ids, on the model's device; id 0 is padding
    lengths  LongTensor (batch,) with the number of real tokens in each row

They return raw scores, not probabilities: CrossEntropyLoss applies the softmax itself.

Both models ignore the padding when they pool over time, so the prediction for a text does not
depend on how long the other texts in its batch are.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextCNN(nn.Module):
    """A convolution of width k looks at k neighbouring tokens at a time, so each filter learns to detect
    one kind of k-token pattern. Taking the maximum over all positions asks "does this pattern occur
    anywhere in the text?", which makes the model independent of where the pattern is and of the length."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, kernel_sizes=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, tokens, lengths):
        widest = max(conv.kernel_size[0] for conv in self.convs)
        if tokens.size(1) < widest:                      # a batch of texts shorter than the widest kernel
            tokens = F.pad(tokens, (0, widest - tokens.size(1)))
        lengths = lengths.to(tokens.device)
        x = self.embedding(tokens).transpose(1, 2)       # (batch, emb_dim, time): Conv1d wants channels second
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))                      # (batch, n_filters, windows)
            # A window is real if all of its k tokens are real. Windows that reach into the padding are set
            # to 0, which can never beat a real window because the ReLU output is never negative.
            last_real = (lengths - conv.kernel_size[0]).clamp(min=0)
            window = torch.arange(h.size(2), device=h.device)
            in_padding = window.unsqueeze(0) > last_real.unsqueeze(1)          # (batch, windows)
            h = h.masked_fill(in_padding.unsqueeze(1), 0.0)
            pooled.append(h.max(dim=2).values)           # (batch, n_filters): strongest match of each filter
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


class LSTMClassifier(nn.Module):
    """An LSTM reads the tokens in order and carries a memory from one token to the next, so its output at
    each position depends on everything read so far. A bidirectional LSTM reads the text both ways, so each
    position knows what comes before and after it.

    pool chooses how the outputs become one vector per text: "max" takes the maximum over time of every
    output dimension, "last" takes the final memory of each direction. On validation "max" was better
    (see results/tuning.jsonl), which fits texts of several hundred tokens: the final memory has to
    carry everything, while the maximum can pick up a signal from any position."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=256, layers=1, dropout=0.5, pool="max"):
        super().__init__()
        assert pool in ("max", "last")
        self.pool = pool
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, tokens, lengths):
        x = self.dropout(self.embedding(tokens))
        # Packing tells the LSTM where each row really ends, so it does not run over the padding.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (memory, _) = self.lstm(packed)
        if self.pool == "last":
            # memory[-2] is the forward direction after the last token, memory[-1] the backward one after the first.
            return self.out(self.dropout(torch.cat([memory[-2], memory[-1]], dim=1)))
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)           # (batch, time, 2 * hidden)
        position = torch.arange(outputs.size(1), device=outputs.device)
        in_padding = position.unsqueeze(0) >= lengths.to(outputs.device).unsqueeze(1)
        outputs = outputs.masked_fill(in_padding.unsqueeze(2), float("-inf"))
        return self.out(self.dropout(outputs.max(dim=1).values))
