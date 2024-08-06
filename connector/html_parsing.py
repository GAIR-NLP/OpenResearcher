import os
from tqdm import tqdm
import json
import os
import html2text
from datetime import datetime
from nltk.tokenize import blankline_tokenize
import argparse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from utils.my_sentence_splitter import MySentenceSplitter
from utils.qdrant_helper import sparse_doc_vectors, sparse_query_vectors, reciprocal_rank_fusion
from config import qdrant_host, qdrant_port, qdrant_collection_prefix, embed_model_path, category_list, qdrant_month_list, category_dict

def get_meta_data_dict(meta_data_path):
    with open(meta_data_path, 'r') as f:
        lines = f.readlines()
    meta_data_dict = {}
    for line in tqdm(lines):
        meta_data = json.loads(line)
        idx = meta_data['id']
        meta_data_dict[idx] = meta_data
    return meta_data_dict


def split_id_func(i, doc):
    # doc.doc_id = 2301000010001
    # 2301.00001 0001 001    
    # paper id is 2301.00001 
    # the 0001 paragraph 
    # 001 segmentation, which is segmented by sentence splitter
    prefix = doc.doc_id
    formatted_str = str(i).zfill(3)
    return int(prefix + formatted_str)

def get_unique_doc_id(paper_id="2301.00001",para_id=0):
    prefix = paper_id.replace(".", "")
    formatted_str = str(para_id).zfill(4)
    return prefix + formatted_str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--end_index", default=-1, type=int)
    parser.add_argument("--meta_data_path", default="data/arxiv-metadata-oai-snapshot.jsonl")
    parser.add_argument("--target_dir", default='data/2401', type=str)
    parser.add_argument("--chunk_size", default=512, type=int)
    parser.add_argument("--embed_batch_size", default=30, type=int)
    parser.add_argument("--insert_batch_size", default=60, type=int)
    parser.add_argument("--para_threshold", default=200, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    start_index = args.start_index
    end_index = args.end_index
    target_dir = args.target_dir
    meta_data_path = args.meta_data_path
    chunk_size = args.chunk_size
    embed_batch_size = args.embed_batch_size
    insert_batch_size = args.insert_batch_size
    para_threshold = args.para_threshold
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path, device='cuda', embed_batch_size=embed_batch_size, trust_remote_code=True)
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    aclient = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
    transformations = [
        MySentenceSplitter(chunk_size=chunk_size, id_func=split_id_func)
    ]
    vector_stores = {}
    storage_contexts = {}
    collection_documents = {}
    for category in category_list:
        for month in qdrant_month_list:
            collection_name = "{}_{}_{}".format(qdrant_collection_prefix, category, month).replace(" ", "_").lower()
            print(collection_name)
            vector_store = QdrantVectorStore(
                collection_name,
                client=client,
                aclient=aclient,
                enable_hybrid=True,
                batch_size=insert_batch_size,
                sparse_doc_fn=sparse_doc_vectors,
                sparse_query_fn=sparse_query_vectors,
                hybrid_fusion_fn=reciprocal_rank_fusion,
            )
            vector_stores[collection_name] = vector_store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            storage_contexts[collection_name] = storage_context
            collection_documents[collection_name] = []
    meta_data_dict = get_meta_data_dict(meta_data_path)
    
    # loop all files in the dir
    for root, dirs, files in os.walk(target_dir):
        for dir_name in dirs[start_index: end_index]:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in meta_data_dict:
                continue
            doc_path = os.path.join(dir_path, 'doc.html')

            # check doc.html 
            if not os.path.isfile(doc_path):
                continue
            with open(doc_path, encoding='utf-8') as f:
                content = f.read()
            content = html2text.html2text(content)
            content = content.split("###### Abstract", 1)[-1]
            content = content.split("## References", 1)[0]
            content = content.split("## REFERENCES", 1)[0]
            content = content.split("## REFERENCE", 1)[0]
            content = content.split("## Reference", 1)[0]
            content = content.split("## Acknowledgment", 1)[0]
            
            paragraphs = blankline_tokenize(content)
            new_paragraphs = []
            cur_para = ""
            for paragraph in paragraphs:
                paragraph = paragraph.replace("\n", " ").strip()
                if len(paragraph) < para_threshold:
                    if len(paragraph) + len(cur_para.strip()) < chunk_size:
                        cur_para += paragraph + "\n"
                    else:
                        new_paragraphs.append(cur_para.strip())
                        cur_para = paragraph
                else:
                    if len(cur_para.strip()) > 0:
                        if len(cur_para.strip()) >= para_threshold / 2:
                            new_paragraphs.append(cur_para.strip())
                        cur_para = ""
                    new_paragraphs.append(paragraph)

            para_cnt = 0
            for paragraph in new_paragraphs:
                meta_data = meta_data_dict[dir_name]
                filtered_meta_data = {'id': dir_name}
                if meta_data['authors'] is not None:
                    filtered_meta_data["authors"] = meta_data['authors']
                if meta_data['title'] is not None:
                    filtered_meta_data["title"] = meta_data['title']
                if meta_data['journal-ref'] is not None:
                    filtered_meta_data["journal-ref"] = meta_data['journal-ref']
                if meta_data['categories'] is not None:
                    filtered_meta_data["categories"] = meta_data['categories'].split(" ")
                if dir_name not in meta_data_dict:
                    paper_time = "20{}-{}-01T12:00:00Z".format(dir_name[0:2], dir_name[2:4])
                else:
                    metadata = meta_data_dict[dir_name]
                    rfc_time = metadata['versions'][0]['created']
                    dt = datetime.strptime(rfc_time, "%a, %d %b %Y %H:%M:%S GMT")
                    paper_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                paper_month = paper_time[2:7].replace("-", "")
                paper_time = int(paper_time[0:10].replace("-", ""))
                filtered_meta_data["paper_time"] = paper_time
                cur_doc = Document(
                    text=paragraph,
                    metadata=filtered_meta_data,
                    excluded_llm_metadata_keys=['authors', 'journal-ref', 'categories'],
                    excluded_embed_metadata_keys=['id', 'authors']
                )
                unique_doc_id = get_unique_doc_id(paper_id=dir_name, para_id=para_cnt)
                cur_doc.__setattr__(name="doc_id", value=unique_doc_id)
                belongs = []
                
                for category in meta_data['categories'].split(" "):
                    if category_dict[category] not in belongs:
                        belongs.append(category_dict[category])
                for belong in belongs:
                    collection_name = "{}_{}_{}".format(qdrant_collection_prefix, belong, paper_month).replace(" ", "_").lower()
                    collection_documents[collection_name].append(cur_doc)
                para_cnt += 1
    for category in category_list:
        for month in qdrant_month_list:
            collection_name = "{}_{}_{}".format(qdrant_collection_prefix, category, month).replace(" ", "_").lower()
            print('{} Document number is {}'.format(collection_name, len(collection_documents[collection_name])))
            index = VectorStoreIndex.from_documents(
                collection_documents[collection_name],
                storage_context=storage_contexts[collection_name],
                embed_model=embed_model,
                show_progress=True,
                insert_batch_size=insert_batch_size,
                transformations=transformations
            )
    print("done")