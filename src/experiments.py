"""The runs behind each research question. This file holds no code; it lists the commands.

RQ1  the answer alone
         python src/baseline.py
         python src/train.py --model cnn
         python src/train.py --model lstm

RQ2  the prompt alone, and prompt + answer (the baseline run above covers all three inputs)
         python src/train.py --model cnn  --mode prompt
         python src/train.py --model cnn  --mode both
         python src/train.py --model lstm --mode prompt
         python src/train.py --model lstm --mode both

RQ3  train on all tasks but one, test on that one; once for each of coding, factual, math, writing
         python src/baseline.py --held-out coding
         python src/train.py --model cnn  --held-out coding
         python src/train.py --model lstm --held-out coding

RQ4  what distinguishes the models: style statistics, a style-only classifier, and ablations
         python src/rq4.py

After any run:   python src/evaluate.py   and   python src/plot_results.py
"""
