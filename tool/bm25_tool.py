from llama_index.core.tools import RetrieverTool
import nest_asyncio
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.tools import ToolMetadata
from utils.qdrant_helper import *
nest_asyncio.apply()
from init import hybrid_retriever

class CustomRetriever(BaseRetriever):
    def __init__(self, hybrid_retriever):
        super().__init__()
        self.hybrid_retriever = hybrid_retriever

    def _retrieve(self, query):
        return self.hybrid_retriever.custom_retrieve_bm25(query.query_str)

custom_retriever = CustomRetriever(hybrid_retriever=hybrid_retriever)

bm25_tool = RetrieverTool(retriever=custom_retriever,
                          metadata=ToolMetadata(
                                name="bm25_tool",
                                description="Obtain the ID, title, abstract, author, and other information of relevant papers based on keywords.",
                            ),
                          )
