# OpenResearcher: Unleashing AI for Accelerated Scientific Research
This is the official repository for OpenResearcher.

**Note: This repository is actively maintained and regularly updated to provide the latest features and improvements.**

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Performance](#-performance)
- [Get started](#-get-started)
  - [Setup](#-setup)
  - [Supported models](#-supported-models)
    - [Using Api](#-using-api)
    - [Using Opensource LLMs](#-using-opensource-llms)
  - [Process Data to embeddings](#-process-data-to-embeddings)
  - [Usage](#-usage)
- [Citation](#-citation)


## 📝 Introduction

<p align="center"> <img src="images/logo.jpg" style="width: 33%;" id="title-icon">       </p>

<p align="center"> Welcome to OpenResearcher, an advanced Scientific Research Assistant designed to provide a helpful answer to a research query.

<p align="center"> With access to the arXiv corpus, OpenResearcher can provide the latest scientific insights.

<p align="center"> Explore the frontiers of science with OpenResearcher—where answers await.

## 🏆 Performance
We release the benchmarking results on various RAG-related systems as a leaderboard.

| Models                                             | Correctness |      |      | Richness |      |      | Relevance |      |      |
| -------------------------------------------------- | :---------- | ---- | ---- | -------- | ---- | ---- | --------- | ---- | ---- |
| (Compared to [Perplexity](https://perplexity.ai/)) | Win         | Tie  | Lose | Win      | Tie  | Lose | Win       | Tie  | Lose |
| [iAsk.Ai](https://iask.ai/)                        | 2           | 16   | 12   | 12       | 6    | 12   | 2         | 8    | 20   |
| [You.com](https://you.com/)                        | 3           | 21   | 6    | 9        | 5    | 16   | 4         | 13   | 13   |
| [Phind](https://www.phind.com/)                    | 2           | 26   | 2    | 15       | 7    | 8    | 5         | 13   | 12   |
| Naive RAG                                          | 1           | 22   | 7    | 14       | 8    | 8    | 5         | 16   | 9    |
| OpenResearcher                                     | **10**      | 13   | 7    | **25**   | 4    | 1    | **15**    | 13   | 2    |


We used human experts to evaluate the responses from various RAG systems. If one answer was significantly better than another, it was judged as a win for the former and a lose for the latter. If the two answers were similar, it was considered a tie.



| Models                                             | Richness |      |      | Relevance |      |      |
| -------------------------------------------------- | -------- | ---- | ---- | --------- | ---- | ---- |
| (Compared to [Perplexity](https://perplexity.ai/)) | Win      | Tie  | Lose | Win       | Tie  | Lose |
| [iAsk.Ai](https://iask.ai/)                        | 42       | 0    | 67   | 38        | 0    | 71   |
| [You.com](https://you.com/)                        | 15       | 0    | 94   | 16        | 0    | 93   |
| [Phind](https://www.phind.com/)                    | 52       | 1    | 56   | 54        | 0    | 55   |
| Naive RAG                                          | 41       | 1    | 67   | 57        | 0    | 52   |
| OpenResearcher                                     | **62**   | 2    | 45   | **74**    | 0    | 35   |

GPT-4 Preference Results compared with Perplexity AI outcome. 



## 🚀 Get Started

### 🛠️ Setup <a name="setup"></a>

##### Install necessary packages:

To begin using OpenResearcher, you need to install the required dependencies. You can do this by running the following command:
```bash
git clone https://github.com/GAIR-NLP/OpenResearcher.git 
conda create -n openresearcher python=3.10 
conda activate openresearcher
cd OpenResearcher
pip install -r requirements.txt
```



##### Install Qdrant vector search engine:

First, download the latest Qdrant image from Dockerhub:

```sh
docker pull qdrant/qdrant
```

Then, run the service:

```sh
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

For more Qdrant installation details, you can follow this [link](https://qdrant.tech/documentation/quickstart/).



### 🤖 Supported models

OpenResearcher currently supports API models from [OpenAI](https://openai.com/), [Deepseek](https://www.deepseek.com/), and [Aliyun](https://www.aliyun.com/), as well as most [huggingface](https://huggingface.co/) models supported by vllm.

### Using API:

Modify the API and base URL values in the config.py file located in the root directory to use large language model service platforms that support the OpenAI interface

For example, if you use [Deepseek](https://www.deepseek.com/) as an API provider, and then modify the following value in `config.py`::

```python
...
openai_api_base_url = "https://api.deepseek.com/v1"
openai_api_key = "api key here"
...
```



### Using Opensource LLMs:

Please use [vllm](https://github.com/vllm-project/vllm) to set up the API server for open-source LLMs. For example, use the following command to deploy a Llama 3 70B hosted on HuggingFace:

```sh
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-70B-Instruct \
  --tensor-parallel-size 8 \
  --dtype auto \
  --api-key sk-dummy \
  --gpu-memory-utilization 0.9 \
  --port 5000
```

Then we can initialize the chat-llm with `config.py`:

```python
...
openai_api_base_url = "http://localhost:5000/v1"
openai_api_key = "sk-dummy"
...
```



### Enable Web search:

We currently support [Bing Search](https://www.microsoft.com/en-us/bing/apis) in OpenResearcher. Modify the following value in `config.py`:

```python
...
bing_search_key = "api key here"
bing_search_end_point = "https://api.bing.microsoft.com/"
...
```



### 📊 Process Data to embeddings

#### Indexing and Saving in Qdrant

**1. Download arXiv data (html file) and metadata into the /data** 

​	arXiv data refers to https://info.arxiv.org/help/bulk_data/index.html

​	Metadata refers to https://www.kaggle.com/datasets/Cornell-University/arxiv 

The directory of `data`is formatted as follows:

```
   - data/
     - 2401/  # pub date   
       - 2401.00001/  # paper id    
         - doc.html   # paper content 
       - 2401.00002/
         - doc.html
     - 2402/
    ...
     -arxiv-metadata-oai-snapshot.jsonl   # metadata        
```

**2. Parse the html data**

```sh
CUDA_VISIBLE_DEVICES=0 python -um connector.html_parsing --target_dir /path/to/target/directory --start_index 0 --end_index -1 \
--meta_data_path /path/to/metadata/file
```

**Parameter explanation:**

​	**target_dir:** process the 'target_dir' papers

​	**start_index,end_index:** papers in directory from 'start_index' to 'end_index' will be processed

​	**meta_data_path:** metadata saved path



### 📘 Usage

### Run the RAG application

First, run the Qdrant retriever server:

```sh
python -um utils.async_qdrant_retriever
```

Then run the Elastic Search retriever server:

```sh
python -um utils.async_elasticsearch_retriever
```

Then you can run the OpenResearcher system by following the command:

```sh
 CUDA_VISIBLE_DEVICES=0 streamlit run ui_app.py
```



## 📚 Citation
If this work is helpful, please kindly cite as:

```
@article{zheng2024openresearcher,
  title={OpenResearcher: Unleashing AI for Accelerated Scientific Research},
  author={Zheng, Yuxiang and Sun, Shichao and Qiu, Lin and Ru, Dongyu and Jiayang, Cheng and Li, Xuefeng and Lin, Jifan and Wang, Binjie and Luo, Yun and Pan, Renjie and others},
  journal={arXiv preprint arXiv:2408.06941},
  year={2024}
}
```
