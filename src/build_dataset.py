"""Turn the raw responses into train / validation / test files.

Reads   data/responses/*.jsonl
Writes  data/splits/{train,val,test}.jsonl

Steps:
  - drop empty and failed answers, and keep the number of rows per LLM balanced
  - remove self-identification ("I am Qwen ...") and count how often it occurred
  - split by prompt_id, so all answers to one prompt land in the same split
  - keep the task column; RQ3 holds one task out
"""
