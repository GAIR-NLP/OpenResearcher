from llm.chat_llm import chat
from config import query_router_model

prompt = """You are an intelligent assistant tasked with analyzing user queries in the context of their conversation history with an AI assistant.
Your goal is to categorize user's latest query into one of three types based on its complexity and the resources needed to answer it accurately.

1. Simple Question: A query that you can answer directly using your general knowledge or according the conversation history.

2. Complex Question: A query that requires in-depth information or specific data that would benefit from using a RAG (Retrieval-Augmented Generation) system to provide a comprehensive and accurate answer.

3. Paper-Specific Question: A question about the content of a specific academic paper, which would require accessing and analyzing the full text of that paper to provide an accurate response.

For each question, follow these steps:
1. Carefully read and understand the question.
2. Consider the depth of knowledge required to answer it accurately.
3. Determine if it requires specific paper content or broad information retrieval.
4. Categorize the question into one of the three types.

If it's the Simple Question, respond with 'LLM';
if it's the Complex Question, respond with 'RAG';
if it's the Paper-Specific Question, respond with 'CHAT'.

You can ONLY respond with one of these three options.

---
Conversation history:
{multi_turn_content}

---
User's latest Query:
{query_str}

Query Type:
"""


def query_router(query_str, history_messages):
    multi_turn_content = ""
    for message in history_messages:
        if message['role'] == "user":
            multi_turn_content += "User: " + message['content'] + "\n"
        else:
            multi_turn_content += "Assistant: " + message['content'] + "\n"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(query_str=query_str,
                                                  multi_turn_content=multi_turn_content)}
    ]
    response = ""
    completion = chat(messages, model = query_router_model)
    for chunk in completion:
        response += chunk
    if response == "LLM":
        return 0
    if response == "CHAT":
        return 2
    #  0 = llm, 1 = rag, 2 = chat with paper, if can not decision, use rag
    return 1
    