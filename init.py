from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from config import rerank_model_path, embed_model_path
from service.hybrid_retriever import HybridRetriever
from service.qdrant_retriever import QdrantRetriever
from service.elastic_search_retriever import ElasticSearchRetriever

embed_model = HuggingFaceEmbedding(model_name=embed_model_path, device='cuda', embed_batch_size=45, trust_remote_code=True)

qdrant_retriever = QdrantRetriever(qdrant_api_url="http://localhost:12351/retrieve")

elastic_search_retriever = ElasticSearchRetriever(elastic_search_api_url="http://localhost:12350/retrieve")

flag_embedding_reranker = FlagEmbeddingReranker(model=rerank_model_path, top_n=10)
hybrid_retriever = HybridRetriever(qdrant_retriever=qdrant_retriever,
                                   elastic_search_retriever=elastic_search_retriever,
                                   node_postprocessors=[
                                       flag_embedding_reranker
                                    ]
                                   )

