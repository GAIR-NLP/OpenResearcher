from llm.chat_llm import chat
from config import self_critic_model

critic_prompt = """Given a user's query, an assistant's response, determine if the assistant's response effectively answers the user's query.
If yes, ONLY reply with '[YES]'; otherwise, reply with '[NO]' and provide a detailed reason.

User's Query:
{query_str}

Assistant's Response:
{response}

Judgment:
"""

refine_prompt = """Below, this assistant's response did not effectively answer the user's query.
The reasons are provided below.
Please use the user's query, the assistant's response and its critic, and the relevant content to provide a revised answer to the user's query.

User's Query:
{query_str}

Relevant Content:
{context}

Assistant's Response:
{response}

Critic:
{critic}

Refined Response:
"""


def self_critic(query_str, response):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": critic_prompt.format(query_str=query_str,
                                                  response=response)}
    ]
    completion = chat(messages, model = self_critic_model)
    chat_message = ""
    skip_str_1 = "[YES]"
    same_idx_1 = 0
    skip_str_2 = "[NO]"
    same_idx_2 = 0
    for chunk in completion:
        idx_1 = 0
        while same_idx_1 < len(skip_str_1) and idx_1 < len(chunk) and skip_str_1[same_idx_1] == chunk[idx_1]:
            same_idx_1 += 1
            idx_1 += 1
        idx_2 = 0
        while same_idx_2 < len(skip_str_2) and idx_2 < len(chunk) and skip_str_2[same_idx_2] == chunk[idx_2]:
            same_idx_2 += 1
            idx_2 += 1
        chunk = chunk[max(idx_1, idx_2):]
        chat_message += chunk
    if same_idx_1 >= 5:
        return ""
    return chat_message


def self_refine(query_str, context, response, critic):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": refine_prompt.format(query_str=query_str, 
                                                  context=context,
                                                  response=response,
                                                  critic=critic)}
    ]
    completion = chat(messages, model = self_critic_model)
    for chunk in completion:
        yield chunk
