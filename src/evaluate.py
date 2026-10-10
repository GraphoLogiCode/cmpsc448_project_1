"""Score a trained model on the test split.

Writes to results/: accuracy, macro F1, precision and recall per LLM, and the confusion matrix.
Print the chance level (1 / number of LLMs) beside every score.
"""
