from llama_index.core.retrievers import BaseRetriever
from service.field_selector import field_selector
from service.date_selector import date_selector
from config import category_list, qdrant_month_list, elastic_search_month_list
import concurrent.futures

def retrieve_bm25(elastic_search_retriever, query, need_categories, need_months):
    """Retrieve nodes using BM25 algorithm."""
    return elastic_search_retriever.custom_retrieve(query, need_categories, need_months)

def retrieve_vector(qdrant_retriever, query, need_categories, need_months):
    """Retrieve nodes using vector search."""
    return qdrant_retriever.custom_retrieve(query, need_categories, need_months)

class HybridRetriever(BaseRetriever):
    def __init__(self, qdrant_retriever, elastic_search_retriever, node_postprocessors):
        super().__init__()
        self.qdrant_retriever = qdrant_retriever
        self.elastic_search_retriever = elastic_search_retriever
        self.node_postprocessors = node_postprocessors

    def _retrieve(self, query):
        """Retrieve nodes using both BM25 and vector search concurrently."""
        need_categories = self._get_categories(query)
        qdrant_need_months, es_need_month = self._get_date_ranges(query)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_bm25 = executor.submit(retrieve_bm25, self.elastic_search_retriever, query, need_categories, es_need_month)
            future_vector = executor.submit(retrieve_vector, self.qdrant_retriever, query, need_categories, qdrant_need_months)
            
            bm25_nodes = future_bm25.result()
            vector_nodes = future_vector.result()

        nodes = bm25_nodes + vector_nodes
        return self._postprocess_nodes(nodes, query)

    def custom_retrieve_vector(self, query: str):
        """Retrieve nodes using only vector search."""
        need_categories = self._get_categories(query)
        qdrant_need_months = date_selector(query, range_type="qdrant") or qdrant_month_list
        nodes = retrieve_vector(self.qdrant_retriever, query, need_categories, qdrant_need_months)
        return self._postprocess_nodes(nodes, query)

    def custom_retrieve_bm25(self, query: str):
        """Retrieve nodes using only BM25 search."""
        need_categories = self._get_categories(query)
        es_need_month = date_selector(query, range_type="elastic search") or elastic_search_month_list
        nodes = retrieve_bm25(self.elastic_search_retriever, query, need_categories, es_need_month)
        return self._postprocess_nodes(nodes, query)

    def _get_categories(self, query):
        """Get categories for the query."""
        need_categories = field_selector(query)
        return need_categories if need_categories else category_list

    def _get_date_ranges(self, query):
        """Get date ranges for Qdrant and Elasticsearch."""
        qdrant_need_months, es_need_month = date_selector(query, range_type="all")
        return (
            qdrant_need_months or qdrant_month_list,
            es_need_month or elastic_search_month_list
        )

    def _postprocess_nodes(self, nodes, query):
        """Apply postprocessors to the retrieved nodes."""
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query)
        return nodes