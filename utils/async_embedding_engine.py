import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
import random
from config import sparse_doc_embed_model_path, sparse_query_embed_model_path
class ModelInstance:
    def __init__(self, doc_model_name: str, query_model_name: str, device: str):
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_name)
        self.doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_name).to(device)
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.query_model = AutoModelForMaskedLM.from_pretrained(query_model_name).to(device)
        self.device = device
        self.doc_model.eval()
        self.query_model.eval()
        self.lock = asyncio.Lock()

    async def compute_sparse_vectors(self, texts: List[str], is_doc: bool) -> Tuple[List[List[int]], List[List[float]]]:
        async with self.lock:
            tokenizer = self.doc_tokenizer if is_doc else self.query_tokenizer
            model = self.doc_model if is_doc else self.query_model
            
            tokens = tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = model(**tokens)
            
            logits, attention_mask = output.logits, tokens.attention_mask
            relu_log = torch.log(1 + torch.relu(logits))
            weighted_log = relu_log * attention_mask.unsqueeze(-1)
            tvecs, _ = torch.max(weighted_log, dim=1)

            indices = []
            vecs = []
            for batch in tvecs:
                batch_indices = batch.nonzero(as_tuple=True)[0].tolist()
                indices.append(batch_indices)
                vecs.append(batch[batch_indices].tolist())

            return indices, vecs

class AsyncEmbeddingEngine:
    def __init__(self, doc_model_name: str, query_model_name: str, num_instances: int = 3):
        self.instances = []
        available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        if not available_devices:
            available_devices = ['cpu']
        
        for i in range(num_instances):
            device = available_devices[i % len(available_devices)]
            self.instances.append(ModelInstance(doc_model_name, query_model_name, device))

    async def get_sparse_vectors(self, texts: List[str], is_doc: bool) -> Tuple[List[List[int]], List[List[float]]]:
        instance = random.choice(self.instances)
        return await instance.compute_sparse_vectors(texts, is_doc)

class TextRequest(BaseModel):
    texts: List[str]

app = FastAPI()
embedding_engine = AsyncEmbeddingEngine(sparse_doc_embed_model_path, 
                                        sparse_query_embed_model_path, 
                                        num_instances=10)

@app.post("/embed_doc")
async def embed_doc_texts(request: TextRequest):
    indices, vecs = await embedding_engine.get_sparse_vectors(request.texts, is_doc=True)
    return {"indices": indices, "vectors": vecs}

@app.post("/embed_query")
async def embed_query_texts(request: TextRequest):
    indices, vecs = await embedding_engine.get_sparse_vectors(request.texts, is_doc=False)
    return {"indices": indices, "vectors": vecs}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12345)