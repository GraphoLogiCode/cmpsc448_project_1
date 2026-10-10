"""Train one model and keep its best epoch. Graded as your own work.

  - cross-entropy loss, Adam, gradient clipping (it matters for the LSTM)
  - after every epoch: loss and accuracy on the validation split
  - save the weights of the epoch with the best validation score; stop when it no longer improves
  - fix the random seed

What is already there for you:

    from data import make_loaders           # batches of (tokens, lengths, labels); see data.py
    from evaluate import evaluate_model     # predictions and metrics.json; see evaluate.py

What the figures and the comparison table need from a run named <run>, for example "cnn_answer":

    models/<run>.pt                     the best weights: torch.save(model.state_dict(), ...)
    results/runs/<run>/history.json     one entry per epoch, written with json.dump:
                                        [{"epoch": 1, "train_loss": 0.95, "train_acc": 0.61,
                                          "val_loss": 0.80, "val_acc": 0.65}, ...]
                                        accuracies are shares between 0 and 1
    results/runs/<run>/metrics.json     written for you by
                                        evaluate_model(model, data, run_name=<run>, classifier="CNN",
                                                       mode="answer", device=device)
                                        Call it after loading the best weights back into the model.

Then `python src/evaluate.py` prints the comparison table and `python src/plot_results.py` draws the figures.

Runs to make:
    RQ1   mode="answer", for the CNN and for the LSTM
    RQ2   mode="prompt" and mode="both", for each model
    RQ3   make_loaders(train_tasks=[three tasks], test_tasks=[the fourth]), once per held-out task

Two checks before a full run:
  - train on 50 rows only: the model should reach close to 100% on them, or it has a bug
  - train with mode="prompt": validation accuracy must stay at one third, because every prompt
    appears once per model with identical text
"""
