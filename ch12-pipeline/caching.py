# to be verified

from llama_index import Document, IngestionPipeline
from llama_index.transformations import SentenceSplitter, TitleExtractor

# Step 1: Set up the Ingestion Pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ]
)

# Step 2: Run the pipeline with sample documents (caching will occur automatically)
documents = [Document(text="This is an example document. It will be split into chunks.")]
nodes = pipeline.run(documents=documents)

# Step 3: Persist (save) the cache to disk for reuse in future runs
pipeline.persist("./pipeline_storage")

# Step 4: Load the pipeline from the stored cache
new_pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        TitleExtractor(),
    ]
)
new_pipeline.load("../data)

# Step 5: Run the pipeline again (this will be instant due to caching)
new_nodes = new_pipeline.run(documents=documents)
print("Successfully loaded from cache:", new_nodes)

# Step 6: Clearing the cache if needed
new_pipeline.cache.clear()
print("Cache cleared successfully.")
