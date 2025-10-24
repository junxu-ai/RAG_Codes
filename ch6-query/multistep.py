from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

# load documents
documents = SimpleDirectoryReader(".\codes\data\ml").load_data()
index = VectorStoreIndex.from_documents(documents)

from llama_index.llms.openai import OpenAI

gpt4 = OpenAI(temperature=0, model="gpt-4o-mini")

# set Logging to DEBUG for more detailed outputs
from llama_index.core.query_engine import MultiStepQueryEngine

query_engine = index.as_query_engine(llm=gpt4)

step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)
index_summary = "Used to answer questions about the machine learning methodology."

query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary
)
response_gpt4 = query_engine.query(
    "How to build a end-2-end machine learning project efficiently?",
)
