from llama_index.core.tools import FunctionTool
import nest_asyncio
nest_asyncio.apply()
from inspect import signature
from service.query_internet import query_internet

def search_internet(query: str) -> str:
    return query_internet(query)

fn = search_internet
name = fn.__name__
docstring = fn.__doc__
description = f"{name}{signature(fn)}\n{docstring}"
internet_tool = FunctionTool.from_defaults(fn=fn,
                                    name=name,
                                    description=f"""You can search everything from internet by ask this tool.""")

