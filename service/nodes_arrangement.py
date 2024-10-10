from functools import cmp_to_key
from collections import defaultdict

def format_metadata(metadata):
    return "\n".join(
        f"{key}: {metadata[key]}" for key in [
            'id', 'title', 'authors', 'journal-ref', 'categories', 'paper_time'
        ] if key in metadata
    )

def compare_node(a, b):
    return -1 if a.score > b.score else 1

def nodes_arrangement(nodes):
    paper_dict = defaultdict(list)
    nodes = sorted(nodes, key=cmp_to_key(compare_node))

    for node in nodes:
        paper_dict[node.metadata['id']].append(node)

    contents = []
    vis_dict = {node.metadata['id']: False for node in nodes}

    actions = {
        True: lambda paper_id, cur_papers: None,
        False: lambda paper_id, cur_papers: handle_new_paper(paper_id, cur_papers, vis_dict, contents)
    }

    for paper_id, cur_papers in paper_dict.items():
        actions[vis_dict[paper_id]](paper_id, cur_papers)

    return contents

def handle_new_paper(paper_id, cur_papers, vis_dict, contents):
    vis_dict[paper_id] = True
    cur_papers.sort(key=lambda n: n.node_id)
    metadata = cur_papers[0].metadata
    cur_content = format_metadata(metadata)
    cur_content += "\n" + "\n".join(node.text for node in cur_papers)
    contents.append(cur_content.strip())
