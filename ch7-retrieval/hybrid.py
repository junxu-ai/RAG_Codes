import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from importlib.metadata import version
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

prompt0 = """
# system
As a language model with expertise in reading and summarizing annual or sustainability reports. your task is to provide an answer of a query based on the text from one or more documents. Your response should meet the following criteria: 
1. "Answer": If the evidence is found, respond "Yes.". otherwise, respond "No.".
2. "Explanation": Provide an explanation for your answer. 
3. "Evidence":  extract the relevenat sentences as evidence from given documents. 
4.  Your response should be formated as a json object with "Answer", "Explanation" and "Evidence" as keys.
5. think step by step
"""

questions =" Physical climate risks are either acute or chronic. Acute risks include extreme weather, droughts, heatwaves, storms, floods, extreme precipitation and wildfires. Chronic risks include rising temperatures, rising sea levels, the expansion of tropical pests and diseases into temperate zones, and an accelerating loss of biodiversity. Does the company identify climate-related 'physical risk' as an emerging or existing risk to its business (direct business operations only or wider value chain or supply chain as well)"

# the user may change the file with any public annual report with ESG secitions
documents = SimpleDirectoryReader(input_files=[r"docs/Corporation Limited_AR_2023-12-31_English (1).pdf"]).load_data()

splitter = SentenceSplitter(chunk_size=256)
index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter], show_progress=True
)

## semantic retrieval
from llama_index.retrievers.bm25 import BM25Retriever
vector_retriever = index.as_retriever(similarity_top_k=5)

## 
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=5
)

## integrate two retrievers into one
from llama_index.core.retrievers import QueryFusionRetriever
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    retriever_weights=[0.6, 0.4],
    similarity_top_k=10,
    num_queries=1,  # set this to 1 to disable query generation
    mode="dist_based_score",
    use_async=True,
    verbose=True,
)

nodes_with_scores = retriever.retrieve(
    questions
)

## show the relevant contents
contents=""
for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text[:500]}...\n-----")
    contents = contents +node.text+ "\n\n"

## use the llama_index's native query engine
# from llama_index.core.query_engine import RetrieverQueryEngine
# query_engine = RetrieverQueryEngine.from_args(retriever)
# response = query_engine.query(questions)
# from llama_index.core.response.notebook_utils import display_response
# display_response(response)
# print(response.response)

## or use the native openai API directly
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": prompt0},
    {"role": "user", "content": "# query \n"+questions+"\n\n # text \n"+contents}
  ]
)
print(completion.choices[0].message)


