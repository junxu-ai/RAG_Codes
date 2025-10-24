# to be verified

# A simple demo. more information can be found at https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/

# Import a sample Document helper
from llama_index.core import Document

# Import the OpenAI embedding model (ensure you have your OpenAI API key configured)
from llama_index.embeddings.openai import OpenAIEmbedding

# Import a node parser that splits documents into sentence chunks
from llama_index.core.node_parser import SentenceSplitter

# Import a metadata extractor to extract titles from documents
from llama_index.core.extractors import TitleExtractor

# Import the ingestion pipeline which sequentially applies transformations to documents
from llama_index.core.ingestion import IngestionPipeline

# Import the Qdrant vector store wrapper and initialize a Qdrant client for in-memory usage
# pip install -qU langchain-qdrant # langchain version
# pip install llama-index-vector-stores-qdrant llama-index-readers-file llama-index-embeddings-fastembed llama-index-llms-openai
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# Initialize Qdrant client in memory (for persistent usage, provide a persistent location)
client = qdrant_client.QdrantClient(location=":memory:")

# Create a Qdrant vector store specifying the client and the collection name
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

# Create an ingestion pipeline with a sequence of transformations:
# 1. SentenceSplitter: breaks text into chunks of 25 tokens (no overlap)
# 2. TitleExtractor: extracts a title from the first few nodes as metadata
# 3. OpenAIEmbedding: calculates vector embeddings for each node
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=10),
        TitleExtractor(),
        OpenAIEmbedding(),
    ],
    vector_store=vector_store,  # Attach the vector store to automatically upsert nodes
)

# Run the pipeline to process a sample document; Document.example() provides a demo document.
pipeline.run(documents=[Document.example()])

# Now create a VectorStoreIndex from the vector store.
# This index abstracts query operations over the stored vector embeddings.
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store)

# The 'index' object can now be used to query your vector store.
