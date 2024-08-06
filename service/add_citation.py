import nltk
from llama_index.retrievers.bm25 import BM25Retriever

def split_sentences(text):
    return nltk.sent_tokenize(text)

def add_citation_with_retrieved_node(retrieved_nodes, final_response):
    if retrieved_nodes is None or len(retrieved_nodes) <= 0:
        return final_response
    bm25_retriever = BM25Retriever.from_defaults(nodes=retrieved_nodes, similarity_top_k=2)
    sentences = [sentence for sentence in split_sentences(final_response) if len(sentence) > 20]
    start = 0
    cite_cnt = 1
    threshold = 13.5
    cited_paper_id_to_cnt = {}
    cited_paper_list = []
    for sentence in sentences:
        left = final_response.find(sentence, start)
        right = left + len(sentence)
        relevant_nodes = bm25_retriever.retrieve(sentence)
        if len(relevant_nodes) == 0 or len(sentence.strip()) < 20:
            start = right
            continue
        if len(relevant_nodes) == 1 or relevant_nodes[0].node.metadata['id'] == relevant_nodes[1].node.metadata['id']:
            paper1 = relevant_nodes[0]
            paper1_id = paper1.node.metadata['id']
            paper1_title = paper1.node.metadata['title']
            if paper1.score > threshold:
                if paper1_id not in cited_paper_id_to_cnt:
                    cited_paper_id_to_cnt[paper1_id] = cite_cnt
                    cited_paper_list.append((paper1_id, paper1_title))
                    cite_cnt += 1
                paper1_cite_cnt = cited_paper_id_to_cnt[paper1_id]
                cite_str = f"[[{paper1_cite_cnt}]](https://arxiv.org/abs/{paper1_id})"
                final_response = final_response[:right - 1] + cite_str + final_response[right - 1:]
                start = right + len(cite_str)
            continue
        paper1 = relevant_nodes[0]
        paper2 = relevant_nodes[1]
        paper1_id = paper1.node.metadata['id']
        paper1_title = paper1.node.metadata['title']
        paper2_id = paper2.node.metadata['id']
        paper2_title = paper2.node.metadata['title']
        if paper1.score > threshold and paper2.score > threshold:
            if paper1_id not in cited_paper_id_to_cnt:
                cited_paper_id_to_cnt[paper1_id] = cite_cnt
                cited_paper_list.append((paper1_id, paper1_title))
                cite_cnt += 1
            if paper2_id not in cited_paper_id_to_cnt:
                cited_paper_id_to_cnt[paper2_id] = cite_cnt
                cited_paper_list.append((paper2_id, paper2_title))
                cite_cnt += 1
            paper1_cite_cnt = cited_paper_id_to_cnt[paper1_id]
            paper2_cite_cnt = cited_paper_id_to_cnt[paper2_id]
            if paper1_cite_cnt > paper2_cite_cnt:
                paper1_cite_cnt, paper2_cite_cnt = paper2_cite_cnt, paper1_cite_cnt
                paper1_id, paper2_id = paper2_id, paper1_id
            cite_str = f"[[{paper1_cite_cnt}]](https://arxiv.org/abs/{paper1_id})[[{paper2_cite_cnt}]](https://arxiv.org/abs/{paper2_id})"
            final_response = final_response[:right - 1] + cite_str + final_response[right - 1:]
            start = right + len(cite_str)
        elif paper1.score > threshold:
            if paper1_id not in cited_paper_id_to_cnt:
                cited_paper_id_to_cnt[paper1_id] = cite_cnt
                cited_paper_list.append((paper1_id, paper1_title))
                cite_cnt += 1
            paper1_cite_cnt = cited_paper_id_to_cnt[paper1_id]
            cite_str = f"[[{paper1_cite_cnt}]](https://arxiv.org/abs/{paper1_id})"
            final_response = final_response[:right - 1] + cite_str + final_response[right - 1:]
            start = right + len(cite_str)
        elif paper2.score > threshold:
            if paper2_id not in cited_paper_id_to_cnt:
                cited_paper_id_to_cnt[paper2_id] = cite_cnt
                cited_paper_list.append((paper2_id, paper2_title))
                cite_cnt += 1
            paper2_cite_cnt = cited_paper_id_to_cnt[paper2_id]
            cite_str = f"[[{paper2_cite_cnt}]](https://arxiv.org/abs/{paper2_id})"
            final_response = final_response[:right - 1] + cite_str + final_response[right - 1:]
            start = right + len(cite_str)
    cited_list_str = ""
    for cite_idx, (cited_paper_id, cited_paper_title) in enumerate(cited_paper_list, start=1):
        cited_list_str += f"""[[{cite_idx}] {cited_paper_title}](https://arxiv.org/abs/{cited_paper_id})\n\n"""
    if len(cited_list_str) > 0:
        final_response += "\n\n**REFERENCES**\n\n" + cited_list_str
    return final_response