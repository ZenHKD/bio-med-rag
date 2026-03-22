
def build_medmcqa_prompt(item, PROMPT_TEMPLATE):
    return PROMPT_TEMPLATE.format(
            question=item["question"]
        )

def build_medqa_prompt(item, template):

    options = []

    options.append(f"A. {item['option_A']}")
    options.append(f"B. {item['option_B']}")
    options.append(f"C. {item['option_C']}")
    options.append(f"D. {item['option_D']}")

    if item.get("option_E") not in [None, "None", ""]:
        options.append(f"E. {item['option_E']}")

    options_text = "\n".join(options)

    prompt = template.format(
        question=item["question"],
        options=options_text
    )

    return prompt

