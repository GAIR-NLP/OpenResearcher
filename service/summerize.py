from llm.chat_llm import chat
from config import summerize_model

prompt = """You need to answer the user's question in detail and accurately based on the user's query, the sub-queries given to you, and the responses to the sub-queries.
You only need to answer the user's question; there is no need to mention "based on the given information."

User Query:
{query_str}

Sub-Queries and Responses:
{context}

Answer:
"""

def summerize(query_str, sub_res_list):
    context = ""
    for sub_query, sub_response in sub_res_list:
        context += f"""sub-query:
{sub_query}
sub-response:
{sub_response}

"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str, context=context)}
    ]
    response = chat(messages, model = summerize_model)
    return response
    