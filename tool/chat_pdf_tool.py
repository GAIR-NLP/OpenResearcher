import os
import nest_asyncio
nest_asyncio.apply()
from config import base_paper_path
from llama_index.core.tools import FunctionTool
import html2text
from llm.chat_llm import chat
from inspect import signature
from config import chat_pdf_tool_model

CHAT_PDF_PROMPT = "You are an expert in analyzing and summarizing papers, skilled at answering user questions based on the content of the paper."
prompt_format = """Please understand and analyze the content of the paper, and then answer user questions based on the content of the paper.
[Question]
{prompt}

[Paper Content]
{paper_content}

[Your Answer]
"""

def get_paper_content(id: str) -> str:
    html_path = os.path.join(base_paper_path, id[0:4], id, "doc.html")
    if not os.path.exists(html_path):
        return "Paper not found error."
    with open(html_path, encoding='utf-8') as file:
        content = file.read()
    if content is None:
        return "Paper not found error."
    content = html2text.html2text(content)
    content = content.split("## References", 1)[0]
    content = content.split("## REFERENCES", 1)[0]
    content = content.split("## REFERENCE", 1)[0]
    content = content.split("## Reference", 1)[0]
    content = content.split("## Acknowledgment", 1)[0]
    return content

def chat_with_pdf(id: str, query: str) -> str:
    paper_content = get_paper_content(id)
    messages=[
            {"role": "system", "content": CHAT_PDF_PROMPT},
            {"role": "user", "content": prompt_format.format(prompt=query, 
                                                             paper_content=paper_content)}
    ]
    response = ""
    completion = chat(messages, model = chat_pdf_tool_model)
    for chunk in completion:
        response += chunk
    return response

fn = chat_with_pdf
name = fn.__name__
docstring = fn.__doc__
description = f"{name}{signature(fn)}\n{docstring}"
chat_pdf_tool = FunctionTool.from_defaults(fn=fn,
                                            name=name,
                                            description=f"""If you have a paper id(format like this: xxxx.xxxxx) and want to get information of this paper by asking questions, use this tool. Note that you need to know the id exactly. Below is the detailed information about the function:{description}""")