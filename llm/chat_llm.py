from openai import OpenAI
from config import openai_api_base_url, openai_api_key

def chat(messages, model=None, stop=None):
    return chat_deepseek_stream(messages, model, stop)

def chat_deepseek_stream(messages, model = "deepseek-chat", stop = None):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base_url,
    )
    completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        stop=stop
    )
    for chunk in completion:
        yield chunk.choices[0].delta.content