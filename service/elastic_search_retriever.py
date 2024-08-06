from llama_index.core.retrievers import BaseRetriever
from config import category_list, elastic_seach_month_list
from service.field_selector import field_selector
from service.date_selector import date_selector
from llama_index.core.schema import NodeWithScore
from llama_index.core import Document
from config import category_dict
import requests
import pickle
from llama_index.core.schema import QueryBundle

class ElasticSearchRetriever(BaseRetriever):
    def __init__(
        self,
        elastic_search_api_url,
        # embed_model,
        # similarity_top_k,
        # node_postprocessors
    ):
        super().__init__()
        self.elastic_search_api_url = elastic_search_api_url
        # self.node_postprocessors = node_postprocessors
        # self.embed_model = embed_model
        # self.similarity_top_k = similarity_top_k
        # elastic_search_bm25 = ElasticsearchStore(
        #     es_url=elastic_search_url,
        #     index_name=elastic_search_index_name,
        #     retrieval_strategy=AsyncBM25Strategy(),
        # )
        # index = VectorStoreIndex.from_vector_store(
        #         elastic_search_bm25,
        #         embed_model=self.embed_model,
        # )
        # self.retriever = index.as_retriever(similarity_top_k=self.similarity_top_k)

    def _retrieve(self, query, **kwargs):
        nodes = self.retriever.retrieve(query, **kwargs)
        query_str = query.query_str
        need_categories = field_selector(query_str)
        if len(need_categories) == 0:
            need_categories = category_list
        filtered_nodes = []
        need_months = date_selector(query_str, range_type="elastic search")
        if need_months is None:
            need_months = elastic_seach_month_list
        for node_with_score in nodes:
            cur_node = node_with_score.node
            cur_text = cur_node.text
            paper_month = cur_node.metadata['paper_time']
            if paper_month not in need_months:
                continue
            match_one = False
            paper_categories = cur_node.metadata['categories']
            if paper_categories is not None:
                for category in paper_categories.split(" "):
                    if category_dict[category] in need_categories:
                        match_one = True
                        break
            else:
                match_one = True
            if not match_one:
                continue
            paper_id = cur_node.metadata['id']
            cur_doc = Document(
                text=cur_text,
                metadata=cur_node.metadata,
                excluded_llm_metadata_keys=['authors', 'journal-ref', 'categories', 'paper_time'],
                excluded_embed_metadata_keys=['id', 'authors']
            )
            cur_doc.__setattr__(name="doc_id", value=paper_id.replace(".", "")+"0000")
            filtered_nodes.append(NodeWithScore(node=cur_doc, score=node_with_score.score))
        for postprocessor in self.node_postprocessors:
            filtered_nodes = postprocessor.postprocess_nodes(filtered_nodes, query_bundle=query)
        print("es nodes len: {}".format(len(filtered_nodes)))
        return filtered_nodes

    def api_retrieve(self, query: str):
        response = requests.post(self.elastic_search_api_url, 
                                 json={"query": query})
        nodes = pickle.loads(response.content)
        return nodes
    
    def custom_retrieve(self, query, need_categories, need_months):
        if isinstance(query, QueryBundle):
            nodes = self.api_retrieve(query.query_str)
        else:
            nodes = self.api_retrieve(query)
        filtered_nodes = []
        for node_with_score in nodes:
            cur_node = node_with_score.node
            cur_text = cur_node.text
            paper_month = cur_node.metadata['paper_time']
            if paper_month not in need_months:
                continue
            match_one = False
            paper_categories = cur_node.metadata['categories']
            if paper_categories is not None:
                for category in paper_categories.split(" "):
                    if category_dict[category] in need_categories:
                        match_one = True
                        break
            else:
                match_one = True
            if not match_one:
                continue
            paper_id = cur_node.metadata['id']
            cur_doc = Document(
                text=cur_text,
                metadata=cur_node.metadata,
                excluded_llm_metadata_keys=['authors', 'journal-ref', 'categories', 'paper_time'],
                excluded_embed_metadata_keys=['id', 'authors']
            )
            cur_doc.__setattr__(name="doc_id", value=paper_id.replace(".", "")+"0000")
            filtered_nodes.append(NodeWithScore(node=cur_doc, score=node_with_score.score))
        # for postprocessor in self.node_postprocessors:
        #     filtered_nodes = postprocessor.postprocess_nodes(filtered_nodes, query_bundle=query)
        print("es nodes len: {}".format(len(filtered_nodes)))
        return filtered_nodes