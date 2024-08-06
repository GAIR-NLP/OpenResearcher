from llama_index.core.tools import FunctionTool
import nest_asyncio
nest_asyncio.apply()
from inspect import signature
from llm.chat_llm import chat
from config import code_tool_model

def solve_code(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    generator = chat(messages, model=code_tool_model)
    response = ""
    for gen in generator:
        response += gen
    return response

fn = solve_code
name = fn.__name__
docstring = fn.__doc__
description = f"{name}{signature(fn)}\n{docstring}"
code_tool = FunctionTool.from_defaults(fn=fn,
                                    name=name,
                                    description=f"""Code problem helper. Below is the detailed information about the function:{description}""")

