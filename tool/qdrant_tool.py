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
        return self.hybrid_retriever.custom_retrieve_vector(query)

custom_retriever = CustomRetriever(hybrid_retriever=hybrid_retriever)


qdrant_tool = RetrieverTool(retriever=custom_retriever,
                          metadata=ToolMetadata(
                                name="qdrant_tool",
                                description="Useful for searching every arxiv paper using dense embedding. But you need to ask question as detailed as possible.",
                            ),
                          )
