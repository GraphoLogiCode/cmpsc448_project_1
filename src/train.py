"""Train one model and keep its best epoch. Graded as your own work.

  - cross-entropy loss, Adam, gradient clipping (it matters for the LSTM)
  - after every epoch: loss and accuracy on the validation split
  - save the weights of the epoch with the best validation score; stop when it no longer improves
  - write the settings and the per-epoch numbers to results/, for the training details in the report
  - fix the random seed
"""
