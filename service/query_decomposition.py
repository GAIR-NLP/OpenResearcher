import re
from llm.chat_llm import chat
from config import query_decomposition_model

query_decomposition_system_prompt = """You are an expert at understanding queries and breaking them down into multiple simple sub-queries."""
query_decomposition_user_prompt = """You need to understand the user's intent based on the queries they pose, and decompose the user's queries into sub-queries for more effective handling by the RAG pipeline. 
Here is an example you can refer to, and you just need to respond to each sub-queries line by line. (Sub queries can only have a maximum of 5)
Note that each sub-question should not contain any pronouns; full name need to be provided. Each sub-query MUST NOT depend on the results of previous sub-queries, ensuring that each sub-query is independent.

[Example Begin]
Query:
Provide a summary of the latest developments in quantum computing and their potential impact on cybersecurity.

Sub-queries:
1. Retrieve information on recent advances in quantum computing.
2. Retrieve information on the impact of quantum computing on cybersecurity.

[Example End]

Query:
{query_str}

Sub-queries:
"""


def query_decomposition(query_str):
    messages = [
        {"role": "system", "content": query_decomposition_system_prompt},
        {"role": "user", "content": query_decomposition_user_prompt.format(query_str=query_str)}
    ]
    completion = chat(messages, model = query_decomposition_model)
    response = ""
    for chunk in completion:
        response += chunk
    sub_queries = re.findall(r'\d+\.\s+(.*)', response)
    return sub_queries
    