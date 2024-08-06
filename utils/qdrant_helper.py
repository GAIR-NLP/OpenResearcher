from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from llama_index.core.vector_stores import VectorStoreQueryResult
from config import sparse_doc_embed_model_path, sparse_query_embed_model_path

doc_tokenizer = AutoTokenizer.from_pretrained(
    sparse_doc_embed_model_path, device_map = 'cuda'
)
doc_model = AutoModelForMaskedLM.from_pretrained(
    sparse_doc_embed_model_path, device_map = 'cuda'
)

query_tokenizer = AutoTokenizer.from_pretrained(
    sparse_query_embed_model_path, device_map = 'cuda'
)
query_model = AutoModelForMaskedLM.from_pretrained(
    sparse_query_embed_model_path, device_map = 'cuda'
)


def sparse_doc_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = doc_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    output = doc_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


def sparse_query_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = query_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    output = query_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


def relative_score_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,  # passed in from the query engine
    top_k: int = 20,  # passed in from the query engine i.e. similarity_top_k
) -> VectorStoreQueryResult:
    """
    Fuse dense and sparse results using relative score fusion.
    """
    # print("process fuse")
    # sanity check
    assert dense_result.nodes is not None
    assert dense_result.similarities is not None
    assert sparse_result.nodes is not None
    assert sparse_result.similarities is not None

    # deconstruct results
    sparse_result_tuples = list(
        zip(sparse_result.similarities, sparse_result.nodes)
    )
    sparse_result_tuples.sort(key=lambda x: x[0], reverse=True)

    dense_result_tuples = list(
        zip(dense_result.similarities, dense_result.nodes)
    )
    dense_result_tuples.sort(key=lambda x: x[0], reverse=True)

    # track nodes in both results
    all_nodes_dict = {x.node_id: x for x in dense_result.nodes}
    for node in sparse_result.nodes:
        if node.node_id not in all_nodes_dict:
            all_nodes_dict[node.node_id] = node

    # normalize sparse similarities from 0 to 1
    sparse_similarities = [x[0] for x in sparse_result_tuples]
    max_sparse_sim = max(sparse_similarities)
    min_sparse_sim = min(sparse_similarities)
    sparse_similarities = [
        (x - min_sparse_sim) / (max_sparse_sim - min_sparse_sim)
        for x in sparse_similarities
    ]
    sparse_per_node = {
        sparse_result_tuples[i][1].node_id: x
        for i, x in enumerate(sparse_similarities)
    }

    # normalize dense similarities from 0 to 1
    dense_similarities = [x[0] for x in dense_result_tuples]
    max_dense_sim = max(dense_similarities)
    min_dense_sim = min(dense_similarities)
    dense_similarities = [
        (x - min_dense_sim) / (max_dense_sim - min_dense_sim)
        for x in dense_similarities
    ]
    dense_per_node = {
        dense_result_tuples[i][1].node_id: x
        for i, x in enumerate(dense_similarities)
    }

    # fuse the scores
    fused_similarities = []
    for node_id in all_nodes_dict:
        sparse_sim = sparse_per_node.get(node_id, 0)
        dense_sim = dense_per_node.get(node_id, 0)
        fused_sim = alpha * (sparse_sim + dense_sim)
        fused_similarities.append((fused_sim, all_nodes_dict[node_id]))

    fused_similarities.sort(key=lambda x: x[0], reverse=True)
    fused_similarities = fused_similarities[:top_k]

    # create final response object
    return VectorStoreQueryResult(
        nodes=[x[1] for x in fused_similarities],
        similarities=[x[0] for x in fused_similarities],
        ids=[x[1].node_id for x in fused_similarities],
    )


def reciprocal_rank_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 20,
) -> VectorStoreQueryResult:
    def reciprocal_rank(similarities: List[float]) -> List[float]:
        ranks = [0] * len(similarities)
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = 1 / (rank + 60)
        return ranks

    # Calculate reciprocal ranks for dense and sparse results
    dense_ranks = reciprocal_rank(dense_result.similarities)
    sparse_ranks = reciprocal_rank(sparse_result.similarities)

    # Fusion of reciprocal ranks
    fused_ranks = [(1 - alpha) * dense_r + alpha * sparse_r for dense_r, sparse_r in zip(dense_ranks, sparse_ranks)]

    # Get the indices of top_k fused ranks
    top_indices = sorted(range(len(fused_ranks)), key=lambda i: fused_ranks[i], reverse=True)[:top_k]

    # Construct the fused result based on top_k indices
    fused_ids = [dense_result.ids[i] for i in top_indices]
    fused_nodes = [dense_result.nodes[i] for i in top_indices]
    fused_similarities = [dense_result.similarities[i] for i in top_indices]

    return VectorStoreQueryResult(ids=fused_ids, nodes=fused_nodes, similarities=fused_similarities)
