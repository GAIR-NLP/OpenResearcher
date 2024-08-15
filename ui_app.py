import streamlit as st
from streamlit_chat import message
from PIL import Image
from service.query_understanding import Multi_Turn_Query_Understanding
from service.query_decomposition import query_decomposition
from init import hybrid_retriever
from service.nodes_arrangement import nodes_arrangement
from service.sub_query_response import sub_query_response
from service.extractor import extractor_paper, extractor_internet
from service.summerize import summerize
from service.self_critic import self_critic, self_refine
from service.query_internet import query_internet
import concurrent.futures
from service.query_router import query_router
from tool.bm25_tool import bm25_tool
from tool.qdrant_tool import qdrant_tool
from tool.chat_pdf_tool import chat_pdf_tool
from tool.math_tool import math_tool
from tool.code_tool import code_tool
from tool.internet_tool import internet_tool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openllm import OpenLLM
from llm.chat_llm import chat
from service.add_citation import add_citation_with_retrieved_node
from streamlit.runtime.scriptrunner.script_run_context import (
    add_script_run_ctx,
    get_script_run_ctx,
)
from llama_index.core.schema import NodeWithScore
from threading import current_thread
from example_history.load_example import multimodal, wizard_lm, what_is_ppo
import copy
from config import agent_model, openai_api_base_url, openai_api_key, llm_chat_model, agent_model_base_url

st.set_page_config(page_title="OpenResearcher", page_icon=Image.open("images/page_icon.jpg"), layout="wide")

# Setting page title and header
st.markdown(
    "<h1 style='text-align: center;'>OpenResearcher</h1>",
    unsafe_allow_html=True,
)
st.divider()
st.markdown(
    "<center><i>Welcome to OpenResearcher, an advanced Scientific Research Assistant designed to provide a helpful answer to a research query. <br> With access to the arXiv corpus, OpenResearcher can provide you with the latest scientific insights. <br> Explore the frontiers of science with OpenResearcher—where answers await.</i></center>",
    unsafe_allow_html=True,
)
st.divider()

# Initialise session state variables
if "cnt" not in st.session_state:
    st.session_state.cnt = 0
    st.session_state.query = ""
    st.session_state.done = False
    st.session_state.llm = Multi_Turn_Query_Understanding()
    st.session_state.mode = -1
    st.session_state.skip_rerun = False

# Sidebar
counter_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown(
        "<h3 style='text-align: center;'>Ask anything you want to know!</h3>",
        unsafe_allow_html=True,
    )

    st.sidebar.image("images/logo.jpg", use_column_width=True)
    
    st.write('')
    st.write('')
    st.write('')

    st.markdown(
        "<p><b>Example: </b></p>",
        unsafe_allow_html=True,
    )
    history_ppo_button = st.sidebar.button("What is PPO?", 
                                           key="hostory::what is ppo",
                                           use_container_width=True)
    if history_ppo_button:
        st.session_state.mode = -1
        st.session_state.cnt = 0
        st.session_state.query = ""
        st.session_state.done = False
        st.session_state.messages = copy.deepcopy(what_is_ppo)
        st.session_state.llm = Multi_Turn_Query_Understanding()
        st.session_state.skip_rerun = False
    
    history_multimodal_button = st.sidebar.button("In multimodal pretraining, the...", 
                                                  key="hostory::multimodal",
                                                  use_container_width=True)
    if history_multimodal_button:
        st.session_state.mode = -1
        st.session_state.cnt = 0
        st.session_state.query = ""
        st.session_state.done = False
        st.session_state.messages = copy.deepcopy(multimodal)
        st.session_state.llm = Multi_Turn_Query_Understanding()
        st.session_state.skip_rerun = False
    
    history_wizardlm_button = st.sidebar.button("Search the paper and tell about...", 
                                                key="hostory::wizardlm",
                                                use_container_width=True)
    if history_wizardlm_button:
        st.session_state.mode = -1
        st.session_state.cnt = 0
        st.session_state.query = ""
        st.session_state.done = False
        st.session_state.messages = copy.deepcopy(wizard_lm)
        st.session_state.llm = Multi_Turn_Query_Understanding()
        st.session_state.skip_rerun = False

    st.write('')
    st.write('')
        
    clear_button = st.sidebar.button("Clear Chat History", 
                                     key="clear",
                                     type="primary",
                                     use_container_width=True)
    if clear_button:
        st.session_state.mode = -1
        st.session_state.cnt = 0
        st.session_state.query = ""
        st.session_state.done = False
        st.session_state.messages = []
        st.session_state.llm = Multi_Turn_Query_Understanding()
        st.session_state.skip_rerun = False
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "expanders" in message:
            for i in range(0, 2):
                expander_name, expander_content = message['expanders'][i]
            # for expander_name, expander_content in message['expanders']:
                with st.expander(expander_name, expanded=True):
                    st.write(expander_content)
            with st.expander("Sub Answers:", expanded=True):
                columns_info = message['expanders'][2]
                columns_size = len(columns_info)
                columns = st.columns(columns_size)
                for i in range(columns_size):
                    column_info = columns_info[i]
                    with columns[i]:
                        tabs_info = column_info[0]
                        tabs = st.tabs([tabs_name for tabs_name, tabs_content in tabs_info])
                        for j in range(len(tabs_info)):
                            with tabs[j]:
                                st.write(tabs_info[j][1])
                        st.write(column_info[1])
        elif "critic_expander" in message:
            name, critic = message['critic_expander']
            with st.expander(name, expanded=True):
                st.write(critic)
        st.markdown(message["content"])
    

llm = OpenLLM(model=agent_model, 
             api_base=agent_model_base_url, 
             api_key=openai_api_key)

agent = ReActAgent.from_tools(
    [qdrant_tool, 
     bm25_tool, 
     chat_pdf_tool,
     math_tool, 
     code_tool,
     internet_tool
     ],
    llm=llm,
    verbose=True,
    max_iterations=15,
)

import nltk
def split_sentences(text):
    return nltk.sent_tokenize(text)

def process_content(query_str, content, row, row_ctx):
    add_script_run_ctx(current_thread(), row_ctx)
    with row:
        st.write("RETRIEVED INFO:\n\n")
        cleaned_content = st.write_stream(extractor_paper(query_str=query_str, content=content))
    return cleaned_content

def dummy_write_stream(generator):
    response = ""
    for gen in generator:
        response += gen
    return response

def process_internet_content(query_str):
    internet_content = query_internet(query_str)
    cleaned_content = dummy_write_stream(extractor_internet(query_str=query_str, content=internet_content))
    return cleaned_content

def process_sub_query(sub_query, column, ctx, result):
    add_script_run_ctx(current_thread(), ctx)
    query_str = sub_query
    arr = nodes_arrangement(result)
    context_list = []
    tabs_info = []
    with column:
        tabs_name = [content.split("\n")[0].split(":")[-1].strip() for content in arr]
        tabs = st.tabs(tabs_name)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            row_ctx = get_script_run_ctx()
            future_to_content = [executor.submit(process_content, query_str, arr[i], tabs[i], row_ctx) for i in range(len(arr))]
            web_search = executor.submit(process_internet_content, query_str)
            for i, future in enumerate(future_to_content):
                cleaned_content = future.result(timeout=120)
                context_list.append(cleaned_content)
                tabs_info.append((tabs_name[i], "RETRIEVED INFO:\n\n" + cleaned_content))
            web_search_result = web_search.result()
            context_list.append(web_search_result)
        st.write("SUB ANSWER:\n\n")
        sub_response = st.write_stream(sub_query_response(query_str, context_list))
    column_info = (tabs_info, "SUB ANSWER:\n\n" + sub_response)
    return sub_query, sub_response, result, column_info, web_search_result

def dedup_node(retrieved_nodes):
    if len(retrieved_nodes) > 0 and isinstance(retrieved_nodes[0], NodeWithScore):
        dedup_nodes = []
        node_id_dict = {}
        for node in retrieved_nodes:
            node_id = node.node.node_id
            if node_id not in node_id_dict:
                node_id_dict[node_id] = 1
                dedup_nodes.append(node.node)
        return dedup_nodes
    return retrieved_nodes

def retrieve_for_sub_query(query):
    return hybrid_retriever.retrieve(query)

def get_final_response():
    with st.chat_message("assistant"):
        expanders = []
        with st.expander("Rewrited Question:", expanded=True):
            rewrite = st.write_stream(st.session_state.llm.query_rewrite_according_messages(st.session_state.messages))
            expanders.append(("Rewrited Question:", rewrite))
        with st.spinner('Thinking...'):
            with st.expander("Sub Queries:", expanded=True):
                sub_queries = query_decomposition(rewrite)
                st.write(sub_queries)
                expanders.append(("Sub Queries:", sub_queries))
        sub_res_list = []
        retrieved_nodes = []
        with st.expander("Sub Answers:", expanded=True):
            columns = st.columns(len(sub_queries))
        columns_info = []
        with st.spinner('Thinking...'):
            retrieve_results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_retrieve = [executor.submit(retrieve_for_sub_query, sub_queries[i]) for i in range(len(sub_queries))]
                for i, future in enumerate(future_to_retrieve):
                    retrieve_results.append(future.result())
        web_search_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ctx = get_script_run_ctx()
            future_to_query = [executor.submit(process_sub_query, sub_queries[i], columns[i], ctx, retrieve_results[i]) for i in range(len(sub_queries))]
            for i, future in enumerate(future_to_query):
                sub_query, sub_response, sub_retrieved_nodes, column_info, web_search_result = future.result(timeout=120)
                retrieved_nodes += sub_retrieved_nodes
                sub_res_list.append((sub_query, sub_response))
                columns_info.append(column_info)
                web_search_results.append(web_search_result)
        expanders.append(columns_info)
        final_response = st.write_stream(summerize(rewrite, sub_res_list))
        deduped_nodes = dedup_node(retrieved_nodes)
        final_response_cite = add_citation_with_retrieved_node(deduped_nodes, final_response)
    st.session_state.messages.append({"role": "assistant", 
                                      "content": final_response_cite, 
                                      "expanders": expanders})
    return rewrite, retrieved_nodes, final_response, web_search_results

st.session_state.query = st.chat_input("What do you want to know? I will give your an answer.")

if st.session_state.mode == -1 and st.session_state.query:
    st.session_state.mode = query_router(st.session_state.query, st.session_state.messages)

if st.session_state.mode == 0:
    st.session_state.messages.append({"role": "user", "content": st.session_state.query})
    with st.chat_message("user"):
        st.markdown(st.session_state.query)
    with st.chat_message("assistant"):
        response = st.write_stream(chat(messages=st.session_state.messages,
                                                         model=llm_chat_model))
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.mode = -1


if st.session_state.mode == 1 and st.session_state.cnt < 2 and not st.session_state.done:
    st.session_state.messages.append({"role": "user", "content": st.session_state.query})
    with st.chat_message("user"):
        st.markdown(st.session_state.query)
    if st.session_state.cnt == 0:
        with st.chat_message("assistant"):
            response = st.write_stream(st.session_state.llm.query_understanding_chat(st.session_state.messages))
            st.session_state.cnt += 1
            if not response:
                st.session_state.done = True
                st.rerun()
        if len(response) > 0:
            st.session_state.messages.append({"role": "assistant", "content": response})
    elif st.session_state.cnt == 1:
        st.session_state.done = True

if st.session_state.mode == 1 and st.session_state.done or st.session_state.cnt >= 2:
    if not st.session_state.skip_rerun:
        rewrite, retrieved_nodes, final_response, web_search_results = get_final_response()
        st.session_state.skip_rerun = True
        st.session_state.rerun_info = [rewrite, retrieved_nodes, final_response, web_search_results]
        st.rerun()
    else:
        rewrite = st.session_state.rerun_info[0]
        retrieved_nodes = st.session_state.rerun_info[1]
        final_response = st.session_state.rerun_info[2]
        web_search_results = st.session_state.rerun_info[3]
        with st.spinner('Self Reflecting...'):
            critic = self_critic(rewrite, final_response).strip()
        if len(critic) > 0:
            with st.chat_message("assistant"):
                with st.expander("Self Critic:", expanded=True):
                    st.write(critic)
                context_critic = "\n\n".join(nodes_arrangement(retrieved_nodes))
                context_critic += "\n\n" + "\n\n".join(web_search_results)
                refined_response = st.write_stream(self_refine(rewrite, context_critic, final_response, critic))
                deduped_nodes = dedup_node(retrieved_nodes)
                refined_response_cite = add_citation_with_retrieved_node(deduped_nodes, refined_response)
                st.session_state.messages.append({"role": "assistant", 
                                                "content": refined_response_cite,
                                                "critic_expander": ("Self Critic:", critic)})
        st.session_state.cnt = 0
        st.session_state.done = False
        st.session_state.mode = -1
        st.session_state.skip_rerun = False
        st.session_state.rerun_info = None
        st.rerun()

if st.session_state.mode == 2:
    st.session_state.messages.append({"role": "user", "content": st.session_state.query})
    with st.chat_message("user"):
        st.markdown(st.session_state.query)
    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            agent_response = agent.chat(st.session_state.query).response
        def streaming(content):
            chunks = content.split(" ")
            for chunk in chunks:
                yield chunk + " "
        response = st.write_stream(streaming(agent_response))
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.mode = -1
