import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import BaseRetriever
from config import category_list, qdrant_month_list, qdrant_collection_prefix
from typing import List
from fastapi.responses import Response
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from utils.qdrant_helper import *
from qdrant_client import QdrantClient, AsyncQdrantClient
from config import qdrant_host, qdrant_port, qdrant_collection_prefix
from qdrant_client import QdrantClient
import pickle
from utils.qdrant_helper import *
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class AsyncQdrantRetriever:
    def __init__(
        self,
        embed_model,
        vector_store,
        similarity_top_k,
        hybrid_top_k
    ):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.similarity_top_k = similarity_top_k
        self.hybrid_top_k = hybrid_top_k
        self.lock = asyncio.Lock()

    async def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async with self.lock:
            emb = await self.embed_model.aget_text_embedding_batch([query_bundle.query_str])
            q = VectorStoreQuery(
                query_embedding=emb[0],
                similarity_top_k=self.similarity_top_k,
                query_str=query_bundle.query_str,
                hybrid_top_k=self.hybrid_top_k,
                mode='hybrid',
            )
            result = await self.vector_store.aquery(q)
            nodes = [NodeWithScore(node=node, score=score) 
                     for node, score in zip(result.nodes, result.similarities)]
            return nodes

class AsyncRetrieverManager:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        embed_model_path,
        similarity_top_k: int,
        hybrid_top_k: int,
        qdrant_collection_prefix: str,
        category_list: List[str],
        qdrant_month_list: List[str],
        sparse_doc_vectors,
        sparse_query_vectors,
        reciprocal_rank_fusion,
        insert_batch_size: int
    ):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.aclient = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_path, 
                                                device='cuda', 
                                                embed_batch_size=45, 
                                                trust_remote_code=True)
        self.similarity_top_k = similarity_top_k
        self.hybrid_top_k = hybrid_top_k
        self.retrievers_dict = {}
        
        for category in category_list:
            for month in qdrant_month_list:
                collection_name = f"{qdrant_collection_prefix}_{category}_{month}".replace(" ", "_").lower()
                vector_store = QdrantVectorStore(
                    collection_name=collection_name,
                    client=self.client,
                    aclient=self.aclient,
                    enable_hybrid=True,
                    batch_size=insert_batch_size,
                    sparse_doc_fn=sparse_doc_vectors,
                    sparse_query_fn=sparse_query_vectors,
                    hybrid_fusion_fn=reciprocal_rank_fusion,
                )
                retriever = AsyncQdrantRetriever(
                    embed_model=self.embed_model,
                    vector_store=vector_store,
                    similarity_top_k=similarity_top_k,
                    hybrid_top_k=hybrid_top_k
                )
                self.retrievers_dict[collection_name] = retriever

    async def retrieve(self, query: str, collection_name: str) -> List[NodeWithScore]:
        retriever = self.retrievers_dict.get(collection_name)
        if not retriever:
            raise ValueError(f"No retriever found for collection: {collection_name}")
        return await retriever.retrieve(QueryBundle(query))

class RetrieveRequest(BaseModel):
    query: str
    collection_name: str

app = FastAPI()

from config import category_list, qdrant_month_list, embed_model_path



retriever_manager = AsyncRetrieverManager(
    qdrant_host=qdrant_host,
    qdrant_port=qdrant_port,
    embed_model_path=embed_model_path,
    similarity_top_k=30,
    hybrid_top_k=30,
    qdrant_collection_prefix=qdrant_collection_prefix,
    category_list=category_list,
    qdrant_month_list=qdrant_month_list, 
    sparse_doc_vectors=sparse_doc_vectors,
    sparse_query_vectors=sparse_query_vectors,
    reciprocal_rank_fusion=reciprocal_rank_fusion,
    insert_batch_size=100
)

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    nodes = await retriever_manager.retrieve(request.query, request.collection_name)
    pickled_data = pickle.dumps(nodes)
    return Response(content=pickled_data, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12351)