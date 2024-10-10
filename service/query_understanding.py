from llm.chat_llm import chat
from config import query_understanding_model

assistant_prompt = """You are a literature assistant. Your task is to ask a question to the user to get a better understanding of their query. To provide the best assistance, follow these steps when interacting with users:

[RULE]
1. Ask for the specific academic field or subject area they are interested in, or inquire about the time range for the literature search (e.g., recent years, last decade, specific period).
2. Request additional details or context regarding their inquiry to narrow down the search (e.g., specific theories, key terms, notable authors).
3. Make sure your questions are clear and concise.
4. If you need to ask the user a question, start your response with '[NEED MORE INFORMATION]', followed by your question.
5. If you have gathered enough information, reply with '[DONE]' only.
6. Ask only the necessary questions to gather sufficient information to proceed.
7. Ask only one question at a time.

Note: "Assistant" in the conversation history refers to you, meaning that if the assistant has already asked N questions, it indicates that you have used N questioning opportunities.

[Example Begin]

User: What is PPO?
Your: [NEED MORE INFORMATION] I need more information to better answer your question. Could you provide the academic field of 'PPO' you are inquiring about, or please explain the question in more detail?
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
Please note that you cannot omit any query information from the user.
Your response must only contain the new query phrased from the user's perspective.

Here is an example you can refer to:
[Example Begin]
Conversation history:
User: Recommend some datasets for verifying the correctness of model responses to factual questions.
Assistant: Could you specify the academic field or domain you are interested in for these datasets?
User: NLP in computer science

Rewritten query:
Recommend some datasets for verifying the correctness of model responses to factual questions in the field of NLP (Natural Language Processing).
[Example End]
---
Conversation history:
{multi_turn_content}

Rewritten query:
"""

def split_last_newline(input_string):
    """Splits the input string at the last newline and returns the string without the last line."""
    input_string = input_string.strip()
    last_newline_index = input_string.rfind('\n')
    return input_string[:last_newline_index] if last_newline_index != -1 else input_string


class MultiTurnQueryUnderstanding:
    def __init__(self):
        pass
        
    def _build_conversation_history(self, history_messages):
        """Builds a string representation of the conversation history."""
        return "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history_messages
            if msg.get('role') and msg.get('content')
        )

    def query_understanding_chat(self, history_messages):
        """Processes the conversation history and generates a question or a '[DONE]' response."""
        multi_turn_content = self._build_conversation_history(history_messages)
        formatted_prompt = assistant_prompt.format(multi_turn_content=multi_turn_content)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ]

        completion = chat(messages, model=query_understanding_model, stop=["\n"])
        chat_message = ""
        for chunk in completion:
            chat_message += chunk
            if chat_message.startswith("[DONE]"):
                return "[DONE]"
            if chat_message.startswith("[NEED MORE INFORMATION]"):
                return chat_message.strip()
        return chat_message.strip()

    def query_rewrite_according_messages(self, history_messages):
        """Rewrites the user's query based on the conversation history."""
        multi_turn_content = split_last_newline(self._build_conversation_history(history_messages))
        formatted_prompt = query_rewrite_prompt.format(multi_turn_content=multi_turn_content)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ]

        completion = chat(messages, model=query_understanding_model)
        return "".join(completion).strip()

