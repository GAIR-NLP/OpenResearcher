## Connector: Operations between components and databases

### Qdrant Vector Storage: html_parsing.py
- Qdrant deployed on localhost:6333
- Visualization page: http://localhost:6333/dashboard#/collections/
- `get_meta_data_dict()` retrieves all arXiv metadata, where key is the paper ID and value is the paper's metadata
- Paper metadata format can be referenced in pdf_parsing.py, line 87
- `get_unique_doc_id` retrieves the paragraph ID of a paper:
  - Example: paper_id="2301.00001", para_id=0 represents the first paragraph of that paper
  - para_id is 4 digits, auto-filled with leading zeros (max 9999 paragraphs per paper)
- `split_id_func` retrieves the ID of paragraph chunks:
  - Example: If a paragraph is 7680 characters long, it's split into 10 chunks of 512 characters each
  - chunk_id is 3 digits, auto-filled with leading zeros (max 999 chunks per paragraph)
- Unique global ID for each chunk: 2301-00001-0001-002 (paper 2301.00001, 2nd paragraph, 3rd chunk)
- Metadata stored for each chunk: authors, title, journal-ref, categories, paper_time

### BM25 Storage: meta_elastic.py
- Elasticsearch deployed on port 9200
- Visualization: Kibana on port 5601
- In Elasticsearch, the node text = concatenated string of each paper's metadata
- Each chunk is 512 characters long