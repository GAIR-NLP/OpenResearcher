from llm.chat_llm import chat
from config import extractor_model

prompt = """You now have a text.
Based on the user's query, you need to filter out the text content that might be useful for answering the user's question.
Note that if any part of the text might help in answering the user's question, you need to extract it and express it in a clearer form (be careful not to lose any useful information).
Your reply should only include text that might be useful for answering the user's question.
If there is no relevant information in the given context that can answer the user's question, you only need to reply "No relevant information can answer the user's question", do not try to answer the question yourself.

User Query:
{query_str}

Origin Text:
{content}

Helpful Text:
"""


def extractor_paper(query_str, content):
    id, content = content.split("\n", 1)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str, content=content)}
    ]
    completion = chat(messages, model = extractor_model)
    yield id + "\n"
    for chunk in completion:
        yield chunk

def extractor_internet(query_str, content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str, content=content)}
    ]
    completion = chat(messages, model = extractor_model)
    for chunk in completion:
        yield chunk
    