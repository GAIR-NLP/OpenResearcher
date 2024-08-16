from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import pickle
import requests
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import BaseNodePostprocessor

from config import category_list, qdrant_month_list, qdrant_collection_prefix
from service.field_selector import field_selector
from service.date_selector import date_selector

@dataclass
class RetrievalResult:
    nodes: List[NodeWithScore]
    category: str
    month: str

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        qdrant_api_url: str,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ):
        """Initialize the QdrantRetriever.

        Args:
            qdrant_api_url (str): The URL for the Qdrant API endpoint.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): List of node postprocessors.
        """
        super().__init__()
        self.qdrant_api_url = qdrant_api_url
        self.node_postprocessors = node_postprocessors or []

    @lru_cache(maxsize=128)
    def _get_collection_name(self, category: str, month: str) -> str:
        """Generate a collection name based on category and month.

        Args:
            category (str): The category of the collection.
            month (str): The month of the collection.

        Returns:
            str: The formatted collection name.
        """
        return f"{qdrant_collection_prefix}_{category}_{month}".replace(" ", "_").lower()

    def _api_retrieve(self, query: str, collection_name: str) -> List[NodeWithScore]:
        """Retrieve nodes from the Qdrant API.

        Args:
            query (str): The query string.
            collection_name (str): The name of the collection to query.

        Returns:
            List[NodeWithScore]: List of retrieved nodes with scores.
        """
        response = requests.post(
            self.qdrant_api_url,
            json={"query": query, "collection_name": collection_name}
        )
        return pickle.loads(response.content)

    def _retrieve_parallel(self, query: str, categories: List[str], months: List[str]) -> List[RetrievalResult]:
        """Perform parallel retrieval across multiple categories and months.

        Args:
            query (str): The query string.
            categories (List[str]): List of categories to search.
            months (List[str]): List of months to search.

        Returns:
            List[RetrievalResult]: List of retrieval results.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._api_retrieve, query, self._get_collection_name(category, month))
                for category in categories
                for month in months
            ]
            results = [
                RetrievalResult(future.result(), category, month) 
                for future, (category, month) in zip(futures, 
                                                     [(cat, mon) for cat in categories for mon in months])
            ]
        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes based on the query bundle.

        Args:
            query_bundle (QueryBundle): The query bundle containing the query string.

        Returns:
            List[NodeWithScore]: List of retrieved nodes with scores.
        """
        query_str = query_bundle.query_str
        categories = field_selector(query_str) or category_list
        months = date_selector(query_str, range_type="qdrant") or qdrant_month_list

        results = self._retrieve_parallel(query_str, categories, months)
        nodes = [node for result in results for node in result.nodes]

        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

        return nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Public method to retrieve nodes based on the query bundle.

        Args:
            query_bundle (QueryBundle): The query bundle containing the query string.

        Returns:
            List[NodeWithScore]: List of retrieved nodes with scores.
        """
        return self._retrieve(query_bundle)

# Example usage:
if __name__ == "__main__":
    # Initialize the retriever
    retriever = QdrantRetriever(
        qdrant_api_url="https://your-qdrant-api-endpoint.com/retrieve",
        node_postprocessors=[
            # Add any postprocessors here
        ]
    )

    # Create a query bundle
    query = QueryBundle(query_str="Example query string")

    # Retrieve nodes
    retrieved_nodes = retriever.retrieve(query)

    # Process the retrieved nodes
    for node in retrieved_nodes:
        print(f"Node: {node.node.text}, Score: {node.score}")