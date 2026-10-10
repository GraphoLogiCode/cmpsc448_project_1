"""Text to tensors.

  - a tokeniser (word level or character level)
  - a vocabulary built from the training split only, with <pad> and <unk>
  - a Dataset that returns token ids and the label, for one of three inputs:
    the prompt, the answer, or both (RQ2)
  - a collate function that pads a batch and returns the true lengths (the LSTM needs them)
"""
