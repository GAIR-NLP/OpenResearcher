import nltk
from llama_index.retrievers.bm25 import BM25Retriever

def split_sentences(text):
    """Split the input text into sentences."""
    return nltk.sent_tokenize(text)

def create_bm25_retriever(retrieved_nodes):
    """Create and return a BM25Retriever instance."""
    return BM25Retriever.from_defaults(nodes=retrieved_nodes, similarity_top_k=2)

def process_citation(paper, cited_paper_id_to_cnt, cited_paper_list, cite_cnt):
    """Process a single citation and update citation counts."""
    paper_id = paper.node.metadata['id']
    paper_title = paper.node.metadata['title']
    if paper_id not in cited_paper_id_to_cnt:
        cited_paper_id_to_cnt[paper_id] = cite_cnt
        cited_paper_list.append((paper_id, paper_title))
        cite_cnt += 1
    return cited_paper_id_to_cnt[paper_id], cite_cnt

def create_cite_string(paper_id, cite_cnt):
    """Create a citation string for a paper."""
    return f"[[{cite_cnt}]](https://arxiv.org/abs/{paper_id})"

def add_citation_to_response(final_response, right, cite_str):
    """Add a citation string to the response."""
    return final_response[:right - 1] + cite_str + final_response[right - 1:]

def create_references_list(cited_paper_list):
    """Create a formatted string of references."""
    cited_list_str = ""
    for cite_idx, (cited_paper_id, cited_paper_title) in enumerate(cited_paper_list, start=1):
        cited_list_str += f"""[[{cite_idx}] {cited_paper_title}](https://arxiv.org/abs/{cited_paper_id})\n\n"""
    return cited_list_str

def add_citation_with_retrieved_node(retrieved_nodes, final_response):
    """Main function to add citations to the response."""
    if not retrieved_nodes:
        return final_response

    bm25_retriever = create_bm25_retriever(retrieved_nodes)
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
            paper = relevant_nodes[0]
            if paper.score > threshold:
                cite_cnt, cite_cnt = process_citation(paper, cited_paper_id_to_cnt, cited_paper_list, cite_cnt)
                cite_str = create_cite_string(paper.node.metadata['id'], cite_cnt - 1)
                final_response = add_citation_to_response(final_response, right, cite_str)
                start = right + len(cite_str)
        else:
            paper1, paper2 = relevant_nodes[0], relevant_nodes[1]
            if paper1.score > threshold and paper2.score > threshold:
                cite_cnt1, cite_cnt = process_citation(paper1, cited_paper_id_to_cnt, cited_paper_list, cite_cnt)
                cite_cnt2, cite_cnt = process_citation(paper2, cited_paper_id_to_cnt, cited_paper_list, cite_cnt)
                cite_str = create_cite_string(paper1.node.metadata['id'], cite_cnt1) + create_cite_string(paper2.node.metadata['id'], cite_cnt2)
                final_response = add_citation_to_response(final_response, right, cite_str)
                start = right + len(cite_str)
            elif paper1.score > threshold:
                cite_cnt1, cite_cnt = process_citation(paper1, cited_paper_id_to_cnt, cited_paper_list, cite_cnt)
                cite_str = create_cite_string(paper1.node.metadata['id'], cite_cnt1)
                final_response = add_citation_to_response(final_response, right, cite_str)
                start = right + len(cite_str)
            elif paper2.score > threshold:
                cite_cnt2, cite_cnt = process_citation(paper2, cited_paper_id_to_cnt, cited_paper_list, cite_cnt)
                cite_str = create_cite_string(paper2.node.metadata['id'], cite_cnt2)
                final_response = add_citation_to_response(final_response, right, cite_str)
                start = right + len(cite_str)

    cited_list_str = create_references_list(cited_paper_list)
    if cited_list_str:
        final_response += "\n\n**REFERENCES**\n\n" + cited_list_str

    return final_response