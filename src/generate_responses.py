"""Send every prompt to every LLM and save the answers.

Reads   data/prompts/prompts.jsonl        one prompt per line: id, task, source, text
Writes  data/responses/<llm_name>.jsonl   one row per prompt: llm_name, llm_input, llm_output, prompt_id, task

To decide before writing this:
  - which LLMs (at least three families) and how they are run
  - one system prompt, temperature and token limit, used for every LLM
  - what to do with an answer that hits the token limit (keep and flag it, or drop it)
"""
