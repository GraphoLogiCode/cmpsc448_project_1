"""The runs behind each research question.

RQ1  answer only, CNN and LSTM
RQ2  prompt only / answer only / prompt + answer
RQ3  train on all tasks but one, test on the held-out task
RQ4  compare length, vocabulary, lexical diversity and formatting per LLM; then remove one signal
     (strip the markdown, cut answers to equal length) and measure the drop in accuracy
"""
