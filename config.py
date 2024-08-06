base_paper_path = "/path/to/your/html/base/dir/" # store all paper html base path
embed_model_path = "Alibaba-NLP/gte-large-en-v1.5" # dense embed_model_path
rerank_model_path = "BAAI/bge-reranker-v2-m3" # reranker path
elastic_search_url = "http://localhost:9200" # elastic search service url
elastic_search_index_name = "elastic_search_bm25" # elastic search index name
qdrant_host = "localhost" # qdrant search service url
qdrant_port = 6333 # qdrant search service port
qdrant_collection_prefix = "openresearcher" # qdrant collection's name
sparse_doc_embed_model_path = "naver/efficient-splade-VI-BT-large-doc" # sparse doc_embed_model path
sparse_query_embed_model_path = "naver/efficient-splade-VI-BT-large-query"# sparse query_embed_model path

openai_api_base_url = "api base url here"
openai_api_key = "api key here"

agent_model_base_url = "api base url here"
agent_model = "deepseek-chat"

bing_search_key = "api key here"
bing_search_end_point = "api end point here"

date_selector_model = "deepseek-chat"
extractor_model = "deepseek-chat"
field_selector_model = "deepseek-chat"
query_decomposition_model = "deepseek-chat"
query_router_model = "deepseek-chat"
query_understanding_model = "deepseek-chat"
self_critic_model = "deepseek-chat"
sub_query_response_model = "deepseek-chat"
summerize_model = "deepseek-chat"

llm_chat_model = "deepseek-chat"
chat_pdf_tool_model = "deepseek-chat"
code_tool_model = "deepseek-coder"
math_tool_model = "deepseek-coder"

category_list = ["Computer Science",
                 "Economics",
                 "Electrical Engineering and Systems Science",
                 "Mathematics",
                 "Physics",
                 "Quantitative Biology",
                 "Quantitative Finance",
                 "Statistics"
                 ]

elastic_seach_month_list = [
    "2201", "2202", "2203", "2204", "2205", "2206", 
    "2207", "2208", "2209", "2210", "2211", "2212",
    "2301", "2302", "2303", "2304", "2305", "2306", 
    "2307", "2308", "2309", "2310", "2311", "2312",
    "2401", "2402", "2403", "2404", 
    "2405", "2406", "2407"
]

qdrant_month_list = [
    # "2201", "2202", "2203", "2204", "2205", "2206", 
    # "2207", "2208", "2209", "2210", "2211", 
    "2212",
    "2301", "2302", "2303", "2304", "2305", "2306", 
    "2307", "2308", "2309", "2310", "2311", "2312",
    "2401", "2402",
    "2403", "2404", "2405", "2406"
]


category_dict = {
    "cs.AI": "Computer Science",
    "cs.AR": "Computer Science",
    "cs.CC": "Computer Science",
    "cs.CE": "Computer Science",
    "cs.CG": "Computer Science",
    "cs.CL": "Computer Science",
    "cs.CR": "Computer Science",
    "cs.CV": "Computer Science",
    "cs.CY": "Computer Science",
    "cs.DB": "Computer Science",
    "cs.DC": "Computer Science",
    "cs.DL": "Computer Science",
    "cs.DM": "Computer Science",
    "cs.DS": "Computer Science",
    "cs.ET": "Computer Science",
    "cs.FL": "Computer Science",
    "cs.GL": "Computer Science",
    "cs.GR": "Computer Science",
    "cs.GT": "Computer Science",
    "cs.HC": "Computer Science",
    "cs.IR": "Computer Science",
    "cs.IT": "Computer Science",
    "cs.LG": "Computer Science",
    "cs.LO": "Computer Science",
    "cs.MA": "Computer Science",
    "cs.MM": "Computer Science",
    "cs.MS": "Computer Science",
    "cs.NA": "Computer Science",
    "cs.NE": "Computer Science",
    "cs.NI": "Computer Science",
    "cs.OH": "Computer Science",
    "cs.OS": "Computer Science",
    "cs.PF": "Computer Science",
    "cs.PL": "Computer Science",
    "cs.RO": "Computer Science",
    "cs.SC": "Computer Science",
    "cs.SD": "Computer Science",
    "cs.SE": "Computer Science",
    "cs.SI": "Computer Science",
    "cs.SY": "Computer Science",
    "econ.EM": "Economics",
    "econ.GN": "Economics",
    "econ.TH": "Economics",
    "eess.AS": "Electrical Engineering and Systems Science",
    "eess.IV": "Electrical Engineering and Systems Science",
    "eess.SP": "Electrical Engineering and Systems Science",
    "eess.SY": "Electrical Engineering and Systems Science",
    "math.AC": "Mathematics",
    "math.AG": "Mathematics",
    "math.AP": "Mathematics",
    "math.AT": "Mathematics",
    "math.CA": "Mathematics",
    "math.CO": "Mathematics",
    "math.CT": "Mathematics",
    "math.CV": "Mathematics",
    "math.DG": "Mathematics",
    "math.DS": "Mathematics",
    "math.FA": "Mathematics",
    "math.GM": "Mathematics",
    "math.GN": "Mathematics",
    "math.GR": "Mathematics",
    "math.GT": "Mathematics",
    "math.HO": "Mathematics",
    "math.IT": "Mathematics",
    "math.KT": "Mathematics",
    "math.LO": "Mathematics",
    "math.MG": "Mathematics",
    "math.MP": "Mathematics",
    "math.NA": "Mathematics",
    "math.NT": "Mathematics",
    "math.OA": "Mathematics",
    "math.OC": "Mathematics",
    "math.PR": "Mathematics",
    "math.QA": "Mathematics",
    "math.RA": "Mathematics",
    "math.RT": "Mathematics",
    "math.SG": "Mathematics",
    "math.SP": "Mathematics",
    "math.ST": "Mathematics",
    "astro-ph.CO": "Physics",
    "astro-ph.EP": "Physics",
    "astro-ph.GA": "Physics",
    "astro-ph.HE": "Physics",
    "astro-ph.IM": "Physics",
    "astro-ph.SR": "Physics",
    "cond-mat.dis-nn": "Physics",
    "cond-mat.mes-hall": "Physics",
    "cond-mat.mtrl-sci": "Physics",
    "cond-mat.other": "Physics",
    "cond-mat.quant-gas": "Physics",
    "cond-mat.soft": "Physics",
    "cond-mat.stat-mech": "Physics",
    "cond-mat.str-el": "Physics",
    "cond-mat.supr-con": "Physics",
    "gr-qc": "Physics",
    "hep-ex": "Physics",
    "hep-lat": "Physics",
    "hep-ph": "Physics",
    "hep-th": "Physics",
    "math-ph": "Physics",
    "nlin.AO": "Physics",
    "nlin.CD": "Physics",
    "nlin.CG": "Physics",
    "nlin.PS": "Physics",
    "nlin.SI": "Physics",
    "nucl-ex": "Physics",
    "nucl-th": "Physics",
    "physics.acc-ph": "Physics",
    "physics.ao-ph": "Physics",
    "physics.app-ph": "Physics",
    "physics.atm-clus": "Physics",
    "physics.atom-ph": "Physics",
    "physics.bio-ph": "Physics",
    "physics.chem-ph": "Physics",
    "physics.class-ph": "Physics",
    "physics.comp-ph": "Physics",
    "physics.data-an": "Physics",
    "physics.ed-ph": "Physics",
    "physics.flu-dyn": "Physics",
    "physics.gen-ph": "Physics",
    "physics.geo-ph": "Physics", 
    "physics.hist-ph": "Physics", 
    "physics.ins-det": "Physics", 
    "physics.med-ph": "Physics", 
    "physics.optics": "Physics", 
    "physics.plasm-ph": "Physics", 
    "physics.pop-ph": "Physics", 
    "physics.soc-ph": "Physics", 
    "physics.space-ph": "Physics",
    "quant-ph": "Physics",
    "q-bio.BM": "Quantitative Biology",
    "q-bio.CB": "Quantitative Biology",
    "q-bio.GN": "Quantitative Biology",
    "q-bio.MN": "Quantitative Biology",
    "q-bio.NC": "Quantitative Biology",
    "q-bio.OT": "Quantitative Biology",
    "q-bio.PE": "Quantitative Biology",
    "q-bio.QM": "Quantitative Biology",
    "q-bio.SC": "Quantitative Biology",
    "q-bio.TO": "Quantitative Biology",
    "q-fin.CP": "Quantitative Finance",
    "q-fin.EC": "Quantitative Finance",
    "q-fin.GN": "Quantitative Finance",
    "q-fin.MF": "Quantitative Finance",
    "q-fin.PM": "Quantitative Finance",
    "q-fin.PR": "Quantitative Finance",
    "q-fin.RM": "Quantitative Finance",
    "q-fin.ST": "Quantitative Finance",
    "q-fin.TR": "Quantitative Finance",
    "stat.AP": "Statistics",
    "stat.CO": "Statistics",
    "stat.ME": "Statistics",
    "stat.ML": "Statistics",
    "stat.OT": "Statistics",
    "stat.TH": "Statistics"
}