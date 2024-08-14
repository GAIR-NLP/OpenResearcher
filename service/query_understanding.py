from llm.chat_llm import chat
from config import query_understanding_model

assitant_prompt = """You are an literature assistant. Your task is to ask question to user to get better understanding of user query. To provide the best assistance, follow these steps when interacting with users:

[RULE]
1. May need to ask for the specific academic field or subject area they are interested in, or inquire about the time range for the literature search (e.g., recent years, last decade, specific period).
2. Request additional details or context regarding their inquiry to narrow down the search (e.g., specific theories, key terms, notable authors).
3. Make sure your questions are clear and concise.
4. If you need to ask the user a question, your response should start with '[NEED MORE INFORMATION]', followed by your question. 
5. If you feel you have gathered enough information, please reply with '[DONE]' only.
6. Please ask only the necessary questions to get enough information to proceed.
7. Only can ask 1 questions.

Note that "assistant" in the conversation history refers to you, meaning that if the assistant has already asked N questions, it indicates that you have used N questioning opportunities.

[Example Begin]

User: What is PPO?
Your: [NEED MORE INFORMATION]I need more information to better answer your question. Could you provide the academic field of 'PPO' you are inquiring about, or please explain the question in more detail?
User: Computer field
Your: [DONE]

[Example End]

[Example Begin]

User: I am seeking recommendations for RAG (Retrieval Augmented Generation) benchmarks specifically in the domain of general scientific literature. I am interested in benchmarks that evaluate both retrieval effectiveness and generation quality.
Your: [DONE]

[Example End]

---

Conversation History:
{multi_turn_content}

---
Your Question or [Done]:
"""

query_rewrite_prompt = """You are tasked with analyzing the conversation history between a user and an AI assistant, and then rewriting the user's query to incorporate relevant context from the conversation. 
Your goal is to create a more informative and context-aware query that will help the AI provide a more accurate and helpful response.
Please note that you cannot miss any query information from the user.
Your response must only contain the new query phrased from the user's perspective.

Here is an example you can refer to:
[Example Begin]
Conversation history:
User: Recommend some datasets for verifying the correctness of model responses to factual questions.
Assistant: Could you specify the academic field or domain you are interested in for these datasets?
User: NLP in computer science

Rewrited query:
Recommend some datasets for verifying the correctness of model responses to factual questions in the field of NLP(Natural Language Processing).
[Example End]
---
Conversation history:
{multi_turn_content}

Rewrited query:
"""

def split_last_newline(input_string):
    input_string = input_string.strip()
    last_newline_index = input_string.rfind('\n')
    if last_newline_index != -1:
        return input_string[:last_newline_index]
    else:
        return input_string


class Multi_Turn_Query_Understanding:
    def __init__(self) -> None:
        pass
        
    def query_understanding_chat(self, history_messages):
        multi_turn_content = ""
        for message in history_messages:
            if message['role'] == "user":
                multi_turn_content += "User: " + message['content'] + "\n"
            else:
                multi_turn_content += "Assistant: " + message['content'] + "\n"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": assitant_prompt.format(
                                        multi_turn_content=multi_turn_content)
            }
        ]
        completion = chat(messages, model = query_understanding_model, stop=["\n"])
        chat_message = ""
        skip_str_1 = "[NEED MORE INFORMATION]"
        same_idx_1 = 0
        skip_str_2 = "[DONE]"
        same_idx_2 = 0
        for chunk in completion:
            chat_message += chunk
            idx_1 = 0
            while same_idx_1 < len(skip_str_1) and idx_1 < len(chunk) and skip_str_1[same_idx_1] == chunk[idx_1]:
                same_idx_1 += 1
                idx_1 += 1
            idx_2 = 0
            while same_idx_2 < len(skip_str_2) and idx_2 < len(chunk) and skip_str_2[same_idx_2] == chunk[idx_2]:
                same_idx_2 += 1
                idx_2 += 1
            chunk = chunk[max(idx_1, idx_2):]
            yield chunk
    
    def query_rewrite_according_messages(self, history_messages):
        multi_turn_content = ""
        for message in history_messages:
            if 'role' not in message or 'content' not in message or message['role'] is None or message['content'] is None:
                continue
            if message['role'] == "user":
                multi_turn_content += "User: " + message['content'] + "\n"
            else:
                multi_turn_content += "Assistant: " + message['content'] + "\n"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query_rewrite_prompt.format(multi_turn_content=split_last_newline(multi_turn_content))}
        ]
        completion = chat(messages, model = query_understanding_model)
        for chunk in completion:
            yield chunk
