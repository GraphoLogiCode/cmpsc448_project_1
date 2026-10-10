"""The two required models. Graded as your own work: write it yourself and be able to explain every line.

TextCNN  embedding -> 1-D convolutions of several widths -> max over time -> dropout -> linear
LSTM     embedding -> (bidirectional) LSTM over packed sequences -> last or pooled hidden state -> dropout -> linear

Both take a batch of token ids and return one score per LLM.
"""
