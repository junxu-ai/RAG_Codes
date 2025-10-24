# llamaindex==0.12.16

# Import necessary classes from LlamaIndex.
from llama_index.core import  VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine

# Load documents from the specified directory.
# Adjust the path "\codes\ch6-query" as needed.
documents = SimpleDirectoryReader(".\codes\data\ml").load_data()

# Build a VectorStoreIndex from the loaded documents.
# Using the 'from_documents' factory method is recommended for consistency.
index = VectorStoreIndex.from_documents(documents)

# Define the query string.
query_str = "How to build a machine learning model efficiently?"

# Initialize the HyDE query transform with the option to include the original query.
hyde = HyDEQueryTransform(include_original=True)

# Create a base query engine from the index.
base_query_engine = index.as_query_engine()

# Wrap the base query engine with a TransformQueryEngine using the HyDE query transform.
query_engine = TransformQueryEngine(base_query_engine, query_transform=hyde)

# Execute the query using the transformed query engine.
response = query_engine.query(query_str)

# Print the final response.
print(response)
