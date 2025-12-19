# pip install llama-index-packs-dense-x-retrieval
# it needs a lower version of llammaindex, e.g., 1.10.50

try:
    from llama_index.packs import DenseXRetrievalPack
except ImportError as e:
    print("DenseXRetrievalPack not available. Using a lower version of llamaindex or alternative implementation.")
    

from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
import asyncio

# Load or create sample documents
# Option 1: Load from directory
try:
    documents = SimpleDirectoryReader("./data/ml").load_data()
except:
    # Option 2: Create sample documents if no data directory exists
    documents = [
        Document(text="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data."),
        Document(text="Deep learning uses neural networks with multiple layers to model complex patterns in data."),
        Document(text="Natural language processing enables computers to understand and generate human language."),
        Document(text="Computer vision allows machines to interpret and understand visual information from the world."),
        Document(text="Reinforcement learning trains agents to make sequences of decisions through trial and error.")
    ]

# Create DenseXRetrievalPack
dense_pack = DenseXRetrievalPack(
    documents,
    proposition_llm=OpenAI(model="gpt-4o-mini", max_tokens=7500),
    query_llm=OpenAI(model="gpt-4o", max_tokens=2560),  # Fixed: removed extra space
    text_splitter=SentenceSplitter(chunk_size=1024)
)

# Get the query engine from the pack
dense_query_engine = dense_pack.query_engine

# Create baseline vector store index for comparison
base_index = VectorStoreIndex.from_documents(documents)
base_query_engine = base_index.as_query_engine()

# Example queries to test both engines
test_queries = [
    "What is machine learning and how does it work?",
    "Explain the difference between deep learning and traditional ML approaches",
    "How does reinforcement learning differ from supervised learning?"
]

# Function to compare query results
def compare_engines(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print('='*60)
    
    # Test DenseX retrieval
    print("\n--- DenseX Retrieval Results ---")
    try:
        dense_response = dense_query_engine.query(query)
        print(f"Response: {dense_response}")
    except Exception as e:
        print(f"Error with DenseX: {e}")
    
    # Test Baseline retrieval
    print("\n--- Baseline Retrieval Results ---")
    try:
        base_response = base_query_engine.query(query)
        print(f"Response: {base_response}")
    except Exception as e:
        print(f"Error with baseline: {e}")

# Run comparisons
if __name__ == "__main__":
    for query in test_queries:
        compare_engines(query)
    
    # Example of custom query
    print(f"\n{'='*60}")
    print("CUSTOM QUERY EXAMPLE")
    print('='*60)
    
    custom_query = "Compare and contrast the main approaches in AI: machine learning, deep learning, and reinforcement learning"
    
    print("\n--- DenseX Retrieval ---")
    dense_response = dense_query_engine.query(custom_query)
    print(f"Response: {dense_response}")
    
    print("\n--- Baseline Retrieval ---")
    base_response = base_query_engine.query(custom_query)
    print(f"Response: {base_response}")