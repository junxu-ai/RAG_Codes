from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load documents
documents = SimpleDirectoryReader(".\data\ml").load_data()

# Try with a different OpenAI embedding model
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Create index with explicit embedding model
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Initialize LLM
gpt4 = OpenAI(temperature=0, model="gpt-4o-mini")

from llama_index.core.query_engine import MultiStepQueryEngine

# Set up base query engine
query_engine = index.as_query_engine(llm=gpt4)

# Set up step decomposition transform
step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)
index_summary = "Used to answer questions about the machine learning methodology."

# Create multi-step query engine
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary
)

# Execute query
response_gpt4 = query_engine.query(
    "How to build a end-2-end machine learning project efficiently?",
)

# Display response
print(str(response_gpt4))