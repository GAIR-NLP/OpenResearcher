import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import Response
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncBM25Strategy
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import elastic_search_index_name, elastic_search_url
from llama_index.core.schema import NodeWithScore
from config import embed_model_path
import pickle

class AsyncESRetriever:
    def __init__(self, es_url: str, index_name: str, embed_model_name: str, similarity_top_k: int):
        self.es_url = es_url
        self.index_name = index_name
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name, 
                                                device='cuda', 
                                                embed_batch_size=45, 
                                                trust_remote_code=True)
        self.similarity_top_k = similarity_top_k
        self.lock = asyncio.Lock()
        elastic_search_bm25 = ElasticsearchStore(
            es_url=es_url,
            index_name=index_name,
            retrieval_strategy=AsyncBM25Strategy(),
        )
        index = VectorStoreIndex.from_vector_store(
                elastic_search_bm25,
                embed_model=self.embed_model,
        )
        self.retriever = index.as_retriever(similarity_top_k=self.similarity_top_k)

    async def retrieve(self, query: str, **kwargs) -> List[NodeWithScore]:
        async with self.lock:
            nodes = await self.retriever.aretrieve(query, **kwargs)
            return nodes

class RetrieveRequest(BaseModel):
    query: str
    additional_params: Dict[str, Any] = {}

app = FastAPI()

retriever = AsyncESRetriever(
    es_url=elastic_search_url,
    index_name=elastic_search_index_name,
    embed_model_name=embed_model_path,
    similarity_top_k=80
)

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    nodes = await retriever.retrieve(request.query, **request.additional_params)
    pickled_data = pickle.dumps(nodes)
    return Response(content=pickled_data, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12350)