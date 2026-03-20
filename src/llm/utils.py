import ast

def build_medmcqa_prompt(item, PROMPT_TEMPLATE):
    return PROMPT_TEMPLATE.format(
            question=item["question"]
        )
