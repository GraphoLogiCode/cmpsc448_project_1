"""The two required models. Graded as your own work: write it yourself and be able to explain every line.

TextCNN  embedding -> 1-D convolutions of several widths -> max over time -> dropout -> linear
LSTM     embedding -> (bidirectional) LSTM over packed sequences -> last or pooled hidden state -> dropout -> linear

What the rest of the code expects from each model class:

    model = TextCNN(vocab_size=len(data["vocab"]), n_classes=len(data["names"]), ...)
    scores = model(tokens, lengths)      # shape (batch, n_classes); the largest score is the prediction

    tokens   LongTensor (batch, time) of token ids, on the model's device; id 0 is padding
    lengths  LongTensor (batch,) with the number of real tokens in each row, on the CPU
             (the CNN can ignore it; pack_padded_sequence in the LSTM needs it)

Return raw scores, not probabilities: CrossEntropyLoss applies the softmax itself.
"""
