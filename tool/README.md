## Tools for Agent Selection and Operation

### chat_pdf_tool:
- Currently reads from `/path/to/your/html/base/dir/{2301}/{2301.00001}/doc.html`, then parses into text
- Queries chat_model with a simple prompt for responses
- Wrapped function: `chat_with_pdf`
  - Input: paper ID and a question
  - Output: chat_model's answer

### qdrant_tool
RetrieverWithRerank class:
- Initialization: Takes one or more retrievers and a node_postprocessors (rerank)
- Qdrant uses custom retriever: MyQdrantRetriever

### bm25_tool
- BM25 uses ElasticsearchStore

### metadata_retrieve
- Adds filter retrieval to the Qdrant vector database
- Available filters: id, title, author
- Requires the model to generate input that meets the conditions