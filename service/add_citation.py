import nltk
from llama_index.retrievers.bm25 import BM25Retriever
from typing import List, Tuple, Optional

def split_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)

def add_citation_with_retrieved_node(retrieved_nodes: Optional[List], final_response: str) -> str:
    if not retrieved_nodes:
        return final_response

    bm25_retriever = BM25Retriever.from_defaults(nodes=retrieved_nodes, similarity_top_k=2)
    sentences = [sentence for sentence in split_sentences(final_response) if len(sentence) > 20]
    
    THRESHOLD = 13.5
    cited_papers = {}
    cited_paper_list = []
    
    for sentence in sentences:
        relevant_nodes = bm25_retriever.retrieve(sentence)
        if not relevant_nodes or len(sentence.strip()) < 20:
            continue

        citations = process_relevant_nodes(relevant_nodes, THRESHOLD, cited_papers, cited_paper_list)
        if citations:
            final_response = insert_citation(final_response, sentence, citations)

    if cited_paper_list:
        final_response += generate_references(cited_paper_list)

    return final_response

def process_relevant_nodes(relevant_nodes: List, threshold: float, cited_papers: dict, cited_paper_list: List) -> str:
    if len(relevant_nodes) == 1 or relevant_nodes[0].node.metadata['id'] == relevant_nodes[1].node.metadata['id']:
        return process_single_paper(relevant_nodes[0], threshold, cited_papers, cited_paper_list)
    else:
        return process_two_papers(relevant_nodes[0], relevant_nodes[1], threshold, cited_papers, cited_paper_list)

def process_single_paper(paper, threshold: float, cited_papers: dict, cited_paper_list: List) -> str:
    if paper.score <= threshold:
        return ""
    
    paper_id = paper.node.metadata['id']
    paper_title = paper.node.metadata['title']
    
    if paper_id not in cited_papers:
        cited_papers[paper_id] = len(cited_papers) + 1
        cited_paper_list.append((paper_id, paper_title))
    
    return f"[[{cited_papers[paper_id]}]](https://arxiv.org/abs/{paper_id})"

def process_two_papers(paper1, paper2, threshold: float, cited_papers: dict, cited_paper_list: List) -> str:
    citations = []
    for paper in (paper1, paper2):
        if paper.score > threshold:
            citation = process_single_paper(paper, threshold, cited_papers, cited_paper_list)
            if citation:
                citations.append(citation)
    
    return "".join(citations)

def insert_citation(text: str, sentence: str, citation: str) -> str:
    start = text.find(sentence)
    if start == -1:
        return text
    end = start + len(sentence)
    return f"{text[:end]}{citation}{text[end:]}"

def generate_references(cited_paper_list: List[Tuple[str, str]]) -> str:
    references = "\n\n**REFERENCES**\n\n"
    for idx, (paper_id, paper_title) in enumerate(cited_paper_list, start=1):
        references += f"[[{idx}] {paper_title}](https://arxiv.org/abs/{paper_id})\n\n"
    return references