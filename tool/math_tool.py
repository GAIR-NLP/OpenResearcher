from llama_index.core.tools import FunctionTool
import nest_asyncio
nest_asyncio.apply()
from inspect import signature
from llm.chat_llm import chat_openai_api_stream
from config import math_tool_model

def solve_math(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    generator = chat_openai_api_stream(messages, model=math_tool_model)
    response = ""
    for gen in generator:
        response += gen
    return response

fn = solve_math
name = fn.__name__
docstring = fn.__doc__
description = f"{name}{signature(fn)}\n{docstring}"
math_tool = FunctionTool.from_defaults(fn=fn,
                                    name=name,
                                    description=f"""Math problem helper. Below is the detailed information about the function:{description}""")

