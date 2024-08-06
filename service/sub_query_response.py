from llm.chat_llm import chat
from config import sub_query_response_model

prompt = """You need to answer the user's query based on the given context.
You need to answer the user's query in detail and accurately.
You only need to answer to the user's query without mentioning "based on the given context content."

User Query:
{query_str}

Context Content:
{context}

Answer:
"""


def sub_query_response(query_str, context_list):
    context = "\n\n".join(context_list)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str, context=context)}
    ]
    completion = chat(messages, model = sub_query_response_model)
    for chunk in completion:
        yield chunk
    