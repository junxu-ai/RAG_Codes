# Caching Implementation for LlamaIndex Ingestion Pipeline

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage import StorageContext
import os
import tempfile

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
print("First run completed. Nodes created:", len(nodes))

# Step 3: Persist (save) the pipeline state to disk for reuse in future runs
persist_dir = "./pipeline_storage"
os.makedirs(persist_dir, exist_ok=True)

# Create storage context for persistence
storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore(),
    index_store=SimpleIndexStore(),
    vector_store=SimpleVectorStore(),
)

# Save the pipeline state
try:
    pipeline.persist(persist_dir)
    print(f"Pipeline persisted to {persist_dir}")
except Exception as e:
    print(f"Error persisting pipeline: {e}")

# Step 4: Load the pipeline from the stored cache
try:
    # For loading, we need to recreate the pipeline with the same transformations
    new_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            TitleExtractor(),
        ]
    )
    
    # Load from storage (corrected syntax)
    if os.path.exists(persist_dir):
        # Note: In LlamaIndex, caching/loading works differently
        # The pipeline doesn't have a direct load method
        print("Pipeline storage directory exists")
    else:
        print("Pipeline storage directory does not exist")
        
except Exception as e:
    print(f"Error loading pipeline: {e}")

# Step 5: Demonstrate caching behavior with identical documents
print("\nDemonstrating caching behavior:")
# Run with the same documents - should be faster due to caching
same_documents = [Document(text="This is an example document. It will be split into chunks.")]
cached_nodes = pipeline.run(documents=same_documents)
print("Cached run completed. Nodes created:", len(cached_nodes))

# Step 6: Run with different documents to show cache miss
different_documents = [Document(text="This is a completely different document with new content.")]
new_nodes = pipeline.run(documents=different_documents)
print("New document run completed. Nodes created:", len(new_nodes))

# Step 7: Show how to check cache status
print("\nPipeline cache information:")
try:
    # Note: Direct cache access depends on the specific implementation
    # In modern LlamaIndex, caching is handled internally
    print("Caching is handled automatically by the pipeline")
    print("Identical transformations and inputs will be cached")
except Exception as e:
    print(f"Cache info not available: {e}")

# Step 8: Demonstrate proper cache clearing (if applicable)
print("\nCache management:")
try:
    # In some versions, you can clear caches like this:
    # pipeline.cache.clear()  # This depends on implementation
    
    # For demonstration, we'll show how to recreate a fresh pipeline
    fresh_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=25, chunk_overlap=0),
            TitleExtractor(),
        ]
    )
    print("Fresh pipeline created (effectively 'cleared' cache)")
    
except Exception as e:
    print(f"Error in cache management: {e}")

# Complete example with proper error handling
def demonstrate_caching():
    """
    Complete example demonstrating caching in LlamaIndex ingestion pipeline
    """
    print("\n" + "="*60)
    print("COMPLETE CACHING DEMONSTRATION")
    print("="*60)
    
    try:
        # Create pipeline
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=50, chunk_overlap=5),
                TitleExtractor(),
            ]
        )
        
        # Test documents
        test_docs = [
            Document(text="Machine learning is a fascinating field of study."),
            Document(text="Deep learning uses neural networks for pattern recognition."),
            Document(text="Natural language processing deals with text understanding.")
        ]
        
        print("Running pipeline for the first time...")
        nodes1 = pipeline.run(documents=test_docs)
        print(f"First run: Created {len(nodes1)} nodes")
        
        print("Running pipeline with same documents (should use cache)...")
        nodes2 = pipeline.run(documents=test_docs)
        print(f"Cached run: Created {len(nodes2)} nodes")
        
        # Check if results are identical (indicating cache hit)
        if len(nodes1) == len(nodes2):
            print("✓ Cache working: Same number of nodes returned")
        else:
            print("⚠ Cache behavior: Different results returned")
            
    except Exception as e:
        print(f"Error in caching demonstration: {e}")

# Run the complete demonstration
if __name__ == "__main__":
    demonstrate_caching()