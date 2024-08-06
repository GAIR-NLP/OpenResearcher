from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import BaseRetriever
from config import category_list, qdrant_month_list, qdrant_collection_prefix
from typing import List
from llama_index.core.base.base_retriever import BaseRetriever
from utils.qdrant_helper import *
from config import qdrant_collection_prefix
from service.field_selector import field_selector
from service.date_selector import date_selector
import concurrent.futures
import pickle
import requests

def get_retrieve_nodes(retriever, query):
    return retriever.retrieve(query)

class SingleQdrantRetriever(BaseRetriever):
    def __init__(
        self,
        embed_model,
        vector_store,
        similarity_top_k,
        hybrid_top_k
    ):
        """Init params."""
        super().__init__()
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.similarity_top_k = similarity_top_k
        self.hybrid_top_k = hybrid_top_k
        

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        emb = self.embed_model.get_text_embedding_batch([query_bundle.query_str])
        q = VectorStoreQuery(
            query_embedding=emb[0],
            similarity_top_k=self.similarity_top_k,
            query_str=query_bundle.query_str,
            hybrid_top_k=self.hybrid_top_k,
            mode='hybrid',
        )
        result = self.vector_store.query(q)
        nodes = []
        for i in range(len(result.nodes)):
            nodes.append(NodeWithScore(node=result.nodes[i], 
                                       score=result.similarities[i]))
        return nodes

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        qdrant_api_url
        # embed_model,
        # similarity_top_k,
        # hybrid_top_k,
        # node_postprocessors,
        # insert_batch_size=10,
    ):
        """Init params."""
        super().__init__()
        self.qdrant_api_url = qdrant_api_url
        # client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # aclient = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        # self.node_postprocessors = node_postprocessors
        # self.embed_model = embed_model
        # self.similarity_top_k = similarity_top_k
        # self.hybrid_top_k = hybrid_top_k
        # self.retrievers_dict = {}
        # for category in category_list:
        #     for month in qdrant_month_list:
        #         collection_name = "{}_{}_{}".format(qdrant_collection_prefix, category, month).replace(" ", "_").lower()
        #         vector_store = QdrantVectorStore(
        #             collection_name,
        #             client=client,
        #             aclient=aclient,
        #             enable_hybrid=True,
        #             batch_size=insert_batch_size,
        #             sparse_doc_fn=sparse_doc_vectors,
        #             sparse_query_fn=sparse_query_vectors,
        #             hybrid_fusion_fn=reciprocal_rank_fusion,
        #         )
        #         retriever = SingleQdrantRetriever(embed_model=embed_model,
        #                                             vector_store=vector_store,
        #                                             similarity_top_k=similarity_top_k,
        #                                             hybrid_top_k=hybrid_top_k)
        #         self.retrievers_dict[collection_name] = retriever

    def _retrieve(self, query, **kwargs):
        nodes = []
        query_str = query.query_str
        need_categories = field_selector(query_str)
        if len(need_categories) == 0:
            need_categories = category_list
        need_months = date_selector(query_str, range_type="qdrant")
        if need_months is None:
            need_months = qdrant_month_list
        cur_retrievers = []
        for category in need_categories:
            for month in need_months:
                collection_name = "{}_{}_{}".format(qdrant_collection_prefix, category, month).replace(" ", "_").lower()
                cur_retrievers.append(self.retrievers_dict[collection_name])
        for retriever in cur_retrievers:
            nodes += retriever.retrieve(query, **kwargs)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query)
        return nodes
    
    def api_retrieve(self, query: str, collection_name: str):
        response = requests.post(self.qdrant_api_url, 
                                 json={"query": query, 
                                       "collection_name": collection_name})
        nodes = pickle.loads(response.content)
        return nodes

    def api_retrieve_wrapper(self, args):
        query_str, collection_name = args
        return self.api_retrieve(query=query_str, collection_name=collection_name)

    def custom_retrieve(self, query, need_categories, need_months):
        nodes = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for category in need_categories:
                for month in need_months:
                    collection_name = "{}_{}_{}".format(qdrant_collection_prefix, category, month).replace(" ", "_").lower()
                    future = executor.submit(
                        self.api_retrieve_wrapper, 
                        (query.query_str, collection_name)
                    )
                    futures.append((future, category, month))
            for future, category, month in futures:
                result = future.result()
                nodes.extend(result)
        return nodes
