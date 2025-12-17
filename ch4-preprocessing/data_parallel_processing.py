from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import SentenceSplitter
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.query.query_transform.base import HyDEQueryTransform

# Configure parallel ingestion pipeline
def create_parallel_pipeline():
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=64,
                paragraph_separator="\n\n"
            ),
            OpenAIEmbedding(
                model="text-embedding-3-small",
                embed_batch_size=256  # Optimized batch size
            )
        ]
    )

# Process documents with parallel workers
documents = SimpleDirectoryReader(
    "../paul_graham_essay/data",
    num_files_limit=1000  # Handle large datasets
).load_data()

pipeline = create_parallel_pipeline()
nodes = pipeline.run(
    documents=documents,
    num_workers=8,  # Optimal for modern CPUs
    show_progress=True
)

# Build optimized vector index
index = VectorStoreIndex(
    nodes,
    batch_size=2048,  # Parallel indexing
    use_async=True
)

# Create query engines
query_engine = index.as_query_engine()

# Optional: Create HyDE query engine for improved semantic search
hyde_transform = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde_transform)

# Example usage:
# response = query_engine.query("What did the author do growing up?")
# hyde_response = hyde_query_engine.query("What did the author do growing up?")