import pandas as pd
from pandasai import SmartDataframe
import os
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

"""
set up the chat response from the dataframe
"""

from pandasai.llm import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"] 
llm_openai = OpenAI(api_token=openai.api_key, model='gpt-3.5-turbo', temperature=0)   
# gpt-4-0613 ; gpt-3.5-turbo-0125 ; gpt-4-1106-preview'  gpt-4-0613

# import google.generativeai as genai
from pandasai.llm import GoogleVertexAI
llm = GoogleVertexAI(api_token="VERTEX_KEY",
                     project_id="ga-pi3",
                     location="us-central1",
                     model="gemini-pro")

# note that in real application, the data shall be authorized by the questioners, and filtered by the person's userID. 
df = pd.read_csv("lti.csv")
df['index'] = df["Metrics (in USD'mil)"]
df = df.set_index('index')

pandas_ai = SmartDataframe(df, config={"llm": llm})

def chat_response_df(message):
    "useful for when you want to answer queries about my own KPI, budget, target and/or actual performance, Gross Profit (GP) Margin, Net Income Margin, SSG of Revenue Contribution."
    result = pandas_ai.chat(message)
    
    if type(result)==str:
        return result
    elif type(result)==pd.DataFrame:
        result = result.drop(index='Description', axis=0, errors='ignore')
        idx = result["Metrics (in USD'mil)"]!="Description"
        result = result.loc[idx]
        return result.to_markdown()

"""
set up the chat response from the documents
"""
from utils import build_automerging_index, get_automerging_query_engine, get_sentence_window_query_engine, build_sentence_window_index

from llama_index import SimpleDirectoryReader
documents = SimpleDirectoryReader(
    input_files=["./data/ar2024.pdf","./data/ar2023.pdf","./data/ar2022.pdf"]
).load_data()

from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index.llms import OpenAI

auto_merging_index_0 = build_automerging_index(
    documents,
    llm=llm_openai,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_0",
    chunk_sizes=[2048,512],
)

auto_merging_engine_0 = get_automerging_query_engine(
    auto_merging_index_0,
    similarity_top_k=12,
    rerank_top_n=6,
)
message =""

def chat_response_doc(message):
    """ useful for when you want to answer queries about the company Lenovo's annual reports and statements."""
    question = f"you are useful assisant to read and understand documents. your task is answer query using the contents from the given documents. Now the query is: {message}"
    return auto_merging_engine_0.query(question)


"""
set up the chat agent
"""

import json
from typing import Sequence, List
from openai.types.chat import ChatCompletionMessageToolCall

# for llama_index version >=0.10
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

# for version <0.10
# from llama_index.tools import QueryEngineTool, ToolMetadata
# from llama_index.tools import BaseTool, FunctionTool
# from llama_index.llms import ChatMessage

tool_doc = FunctionTool.from_defaults(fn=chat_response_doc)
tool_df = FunctionTool.from_defaults(fn=chat_response_df)

query_engine_tools = [
    QueryEngineTool(
        query_engine=tool_df,
        metadata=ToolMetadata(
            name=f"individual_KPI",
            description=(
                "useful for when you want to answer queries about my own KPI, budget, target and/or actual performance, Gross Profit (GP) Margin, Net Income Margin, SSG of Revenue Contribution."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=tool_doc,
        metadata=ToolMetadata(
            name=f"company_report",
            description=(
                "useful for when you want to answer queries about the company Lenovo's annual reports and statements."
            ),
        ),
    ),
]

from llama_index.agent.openai import OpenAIAgent
agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)

"""
set up the gradio UI
"""
def echo(message, history):
    result = agent.chat(message)

    if type(result)==str:
        return result
    elif type(result)==pd.DataFrame:
        result = result.drop(index='Description', axis=0, errors='ignore')
        idx = result["Metrics (in USD'mil)"]!="Description"
        result = result.loc[idx]
        return result.to_markdown()

demo = gr.ChatInterface(fn=echo, examples=["what are the metrics or KPIs for the year 1 budget and actual for FY23/24 plan", "what are the KPIs for LTI plan", "what is the actual value of Gross Profit (GP) Margin for Fy22/23'"], title="LTI Chatbot")
demo.launch()
