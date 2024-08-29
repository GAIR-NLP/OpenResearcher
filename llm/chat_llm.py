from openai import OpenAI
from config import openai_api_base_url, openai_api_key

def chat(messages, model=None, stop=None):
    return chat_openai_api_stream(messages, model, stop)

def chat_openai_api_stream(messages, model = "deepseek-chat", stop = None):
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
        if chunk.choices[0].delta is None or chunk.choices[0].delta.content is None:
            continue
        yield chunk.choices[0].delta.content