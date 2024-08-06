from llama_index.vector_stores.elasticsearch import AsyncBM25Strategy
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from tqdm import tqdm
from llama_index.core.schema import Document
from config import elastic_search_index_name, elastic_search_url, embed_model_path, elastic_seach_month_list
import argparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser.text.sentence import SentenceSplitter
import json
from datetime import datetime

def get_meta_data_dict(meta_data_path):
    with open(meta_data_path, 'r') as f:
        lines = f.readlines()
    meta_data_dict = {}
    for line in tqdm(lines):
        meta_data = json.loads(line)
        idx = meta_data['id']
        meta_data_dict[idx] = meta_data
    return meta_data_dict

def format_metadata(metadata):
    meta_format = ""
    if metadata['id'] is not None:
        meta_format += f"id: {metadata['id']}\n"
    if metadata['title'] is not None:
        meta_format += f"title: {metadata['title']}\n"
    if metadata['abstract'] is not None:
        meta_format += f"abstract: {metadata['abstract']}\n"
    if metadata['authors'] is not None:
        meta_format += f"authors: {metadata['authors']}\n"
    # if metadata['journal-ref'] is not None:
    #     meta_format += f"journal-ref: {metadata['journal-ref']}\n"
    # if metadata['categories'] is not None:
    #     meta_format += f"categories: {metadata['categories']}\n"
    return meta_format.strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_data_path", default="data/arxiv-metadata-oai-snapshot.jsonl")
    parser.add_argument("--chunk_size", default=512, type=int)
    parser.add_argument("--embed_batch_size", default=32, type=int)
    parser.add_argument("--insert_batch_size", default=300000000, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    meta_data_path = args.meta_data_path
    meta_data_dict = get_meta_data_dict(meta_data_path)
    transformations = [
        SentenceSplitter(chunk_size=args.chunk_size)
    ]
    elastic_search_bm25 = ElasticsearchStore(
        es_url=elastic_search_url,
        index_name=elastic_search_index_name,
        retrieval_strategy=AsyncBM25Strategy(),
    )
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path, 
                                       device='cuda', 
                                       embed_batch_size=args.embed_batch_size, 
                                       trust_remote_code=True)
    documents = []
    for idx, meta_data in tqdm(meta_data_dict.items()):
        paper_id = meta_data['id']
        rfc_time = meta_data['versions'][0]['created']
        dt = datetime.strptime(rfc_time, "%a, %d %b %Y %H:%M:%S GMT")
        paper_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        paper_month = paper_time[2:7].replace("-", "")
        if paper_month not in elastic_seach_month_list:
            continue
        cur_doc = Document(
            text=format_metadata(meta_data),
            metadata={
                'id': meta_data['id'],
                'title': meta_data['title'],
                'paper_time': paper_month,
                'journal-ref': meta_data['journal-ref'],
                'categories':meta_data['categories'],
            },
            excluded_embed_metadata_keys=['id', 'title', 'paper_time', 'journal-ref', 'categories']
        )
        documents.append(cur_doc)
    
    storage_context = StorageContext.from_defaults(vector_store=elastic_search_bm25)
    index = VectorStoreIndex.from_documents(documents,
                                            storage_context=storage_context,
                                            show_progress=True,
                                            insert_batch_size=args.insert_batch_size,
                                            transformations=transformations,
                                            embed_model=embed_model)

