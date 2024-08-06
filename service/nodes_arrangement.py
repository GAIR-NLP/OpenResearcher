from functools import cmp_to_key

def format_metadata(metadata):
    meta_format = ""
    if 'id' in metadata:
        meta_format += f"id: {metadata['id']}\n"
    if 'title' in metadata:
        meta_format += f"title: {metadata['title']}\n"
    if 'authors' in metadata:
        meta_format += f"authors: {metadata['authors']}\n"
    if 'journal-ref' in metadata:
        meta_format += f"journal-ref: {metadata['journal-ref']}\n"
    if 'categories' in metadata:
        meta_format += f"categories: {metadata['categories']}\n"
    if 'paper_time' in metadata:
        meta_format += f"paper_time: {metadata['paper_time']}\n"
    return meta_format


def compare_node(a, b):
    if a.score > b.score:
        return -1
    return 1

def nodes_arrangement(nodes):
    paper_dict = {}
    nodes = sorted(nodes, key=cmp_to_key(compare_node))
    for node in nodes:
        paper_id = node.metadata['id']
        if paper_id not in paper_dict:
            paper_dict[paper_id] = []
        paper_dict[paper_id].append(node)
    contents = []
    vis_dict = {}
    for node in nodes:
        paper_id = node.metadata['id']
        if paper_id in vis_dict:
            continue
        vis_dict[paper_id] = True
    # for paper_id, cur_papers in paper_dict.items():
        cur_papers = paper_dict[paper_id]
        cur_papers.sort(key=lambda node: node.node_id)
        metadata = cur_papers[0].metadata
        cur_content = format_metadata(metadata)
        for node in cur_papers:
            cur_content += node.text + '\n'
        contents.append(cur_content.strip())
    return contents