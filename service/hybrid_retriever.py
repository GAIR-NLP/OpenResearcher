from llama_index.core.retrievers import BaseRetriever
from service.field_selector import field_selector
from service.date_selector import date_selector
from config import category_list, qdrant_month_list, elastic_seach_month_list
import concurrent.futures

def retrieve_bm25(elastic_search_retriever, query, need_categories, need_months):
    return elastic_search_retriever.custom_retrieve(query, need_categories, need_months)

def retrieve_vector(qdrant_retriever, query, need_categories, need_months):
    return qdrant_retriever.custom_retrieve(query, need_categories, need_months)

class HybridRetriever(BaseRetriever):
    def __init__(self, qdrant_retriever, 
                 elastic_search_retriever,
                 node_postprocessors):
        super().__init__()
        self.qdrant_retriever = qdrant_retriever
        self.elastic_search_retriever = elastic_search_retriever
        self.node_postprocessors = node_postprocessors

    def _retrieve(self, query):
        need_categories = field_selector(query)
        if len(need_categories) == 0:
            need_categories = category_list
        qdrant_need_months, es_need_month = date_selector(query, range_type="all")
        if qdrant_need_months is None:
            qdrant_need_months = qdrant_month_list
        if es_need_month is None:
            es_need_month = elastic_seach_month_list
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_bm25 = executor.submit(retrieve_bm25, self.elastic_search_retriever, query, need_categories, es_need_month)
            future_vector = executor.submit(retrieve_vector, self.qdrant_retriever, query, need_categories, qdrant_need_months)
            
            bm25_nodes = future_bm25.result()
            vector_nodes = future_vector.result()
        nodes = bm25_nodes + vector_nodes
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query)
        return nodes
    
    def custom_retrieve_vector(self, query: str):
        need_categories = field_selector(query)
        if len(need_categories) == 0:
            need_categories = category_list
        qdrant_need_months = date_selector(query, range_type="qdrant")
        if qdrant_need_months is None:
            qdrant_need_months = qdrant_month_list
        nodes = retrieve_vector(self.qdrant_retriever, query, need_categories, qdrant_need_months)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query)
        return nodes
    
    def custom_retrieve_bm25(self, query: str):
        need_categories = field_selector(query)
        if len(need_categories) == 0:
            need_categories = category_list
        es_need_month = date_selector(query, range_type="elastic search")
        if es_need_month is None:
            es_need_month = elastic_seach_month_list
        nodes = retrieve_bm25(self.elastic_search_retriever, query, need_categories, es_need_month)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query)
        return nodes
