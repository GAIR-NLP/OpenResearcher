from llm.chat_llm import chat
from config import category_list, field_selector_model

prompt = """You need to determine the academic field category that needs to be searched based on the user's query.
There are a total of 7 categories, which are Computer Science, Economics, Electrical Engineering and Systems Science, Mathematics, Physics, Quantitative Biology, Quantitative Finance, Statistics.
Based on the user's query, you should determine which one or more academic field categories should be searched. You only need to reply with the category.
If there are multiple academic field categories, please separate them with commas.
Note that the categories you reply with must match the original category names exactly.

User Query:
{query_str}

Category(ies):
"""

def validate_and_process_categories(llm_output):
    class_names = llm_output.split(", ")
    res = []
    for class_name in class_names:
        if class_name in category_list:
            res.append(class_name)
    return res


def field_selector(query_str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str)}
    ]
    completion = chat(messages, model = field_selector_model)
    response = ""
    for chunk in completion:
        response += chunk
    return validate_and_process_categories(response)