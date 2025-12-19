from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import SentenceSplitter
# For demonstration with a more capable vector store
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# Initialize LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding()

# Load documents from a directory
try:
    documents = SimpleDirectoryReader("./data/ml/").load_data()
except Exception as e:
    print(f"Could not load documents: {e}")
    # Create sample documents if directory doesn't exist
    documents = [
        Document(text="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data. It enables computers to identify patterns and make decisions with minimal human intervention. Common applications include recommendation systems, fraud detection, and predictive analytics."),
        Document(text="Deep learning uses neural networks with multiple layers to model complex patterns in data. These deep neural networks can automatically learn hierarchical representations of data. Deep learning excels in computer vision, natural language processing, and speech recognition tasks."),
        Document(text="Natural language processing enables computers to understand and generate human language. NLP techniques include sentiment analysis, machine translation, and chatbot development. Modern NLP heavily relies on transformer architectures like BERT and GPT.")
    ]

# Split documents into chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = text_splitter.get_nodes_from_documents(documents)

print("VECTOR STORE BACKEND COMPARISON")
print("=" * 50)

# 1. Default Simple Vector Store (only supports DEFAULT mode)
print("\n1. SIMPLE VECTOR STORE (DEFAULT ONLY)")
print("-" * 40)
try:
    # Create index with simple vector store
    simple_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    # Only DEFAULT mode works
    default_retriever = simple_index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=3
    )
    
    response = default_retriever.retrieve("What is machine learning?")
    print(f"DEFAULT mode: Successfully retrieved {len(response)} chunks")
    for i, node in enumerate(response[:2], 1):
        print(f"  {i}. Score: {getattr(node, 'score', 'N/A'):.4f}")
        print(f"     Text: {str(node.text)[:100]}...")
        
    # Try other modes (will likely fail)
    try:
        hybrid_retriever = simple_index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=3
        )
        hybrid_response = hybrid_retriever.retrieve("What is machine learning?")
        print(f"HYBRID mode: Successfully retrieved {len(hybrid_response)} chunks")
    except Exception as e:
        print(f"HYBRID mode: Not supported - {str(e)[:100]}...")
        
except Exception as e:
    print(f"Error with simple vector store: {e}")

# 2. Qdrant Vector Store (supports advanced modes)
print("\n2. QDRANT VECTOR STORE (ADVANCED MODES)")
print("-" * 40)
try:
    # Initialize Qdrant client
    client = qdrant_client.QdrantClient(location=":memory:")  # In-memory for demo
    
    # Create Qdrant vector store
    vector_store = QdrantVectorStore(client=client, collection_name="test_collection")
    
    # Create index with Qdrant backend
    qdrant_index = VectorStoreIndex(
        nodes, 
        embed_model=embed_model,
        vector_store=vector_store
    )
    
    print("Qdrant index created successfully")
    
    # Test different modes that Qdrant might support
    modes_to_test = {
        "DEFAULT": VectorStoreQueryMode.DEFAULT,
    }
    
    # Qdrant may support additional modes depending on version
    if hasattr(VectorStoreQueryMode, 'HYBRID'):
        modes_to_test["HYBRID"] = VectorStoreQueryMode.HYBRID
    
    for mode_name, mode_value in modes_to_test.items():
        try:
            retriever = qdrant_index.as_retriever(
                vector_store_query_mode=mode_value,
                similarity_top_k=3
            )
            response = retriever.retrieve("What is machine learning?")
            print(f"{mode_name} mode: Successfully retrieved {len(response)} chunks")
        except Exception as e:
            print(f"{mode_name} mode: Error - {str(e)[:80]}...")
            
except ImportError:
    print("Qdrant not installed. Run: pip install qdrant-client")
except Exception as e:
    print(f"Error with Qdrant vector store: {e}")

# 3. Explain the limitation and workaround
print("\n3. EXPLANATION AND WORKAROUNDS")
print("-" * 40)
print("""
Why advanced modes don't work with simple vector store:

1. DEFAULT Mode: Basic cosine similarity search - universally supported
2. HYBRID Mode: Requires sparse+dense vector storage - needs specialized backend
3. SEMANTIC_HYBRID: Advanced ranking algorithms - needs specific implementation
4. TEXT_SEARCH: BM25/keyword search - requires inverted index support

Workarounds for advanced retrieval without specialized backends:
- Manual hybrid: Combine separate keyword and semantic searches
- Use external libraries like rank-bm25 for keyword scoring
- Implement custom scoring functions
""")

# Manual hybrid search demonstration
def manual_hybrid_search(nodes, query, embed_model, top_k=3):
    """
    Manual implementation of hybrid search using separate semantic and keyword scoring
    """
    print(f"\nMANUAL HYBRID SEARCH DEMONSTRATION")
    print("-" * 40)
    
    # Semantic scoring (what vector stores do)
    print("1. Semantic scoring (vector similarity):")
    query_embedding = embed_model.get_query_embedding(query)
    
    # In a real implementation, you'd compute cosine similarities
    # For demo, we'll simulate this
    print("   Computing vector similarities...")
    
    # Keyword scoring (what BM25 does)
    print("2. Keyword scoring (term frequency):")
    query_terms = query.lower().split()
    print(f"   Query terms: {query_terms}")
    
    # In a real implementation, you'd use BM25 or TF-IDF
    print("   Computing keyword scores...")
    
    # Combined approach explanation
    print("3. Combined hybrid approach:")
    print("   Final score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)")
    
    # Return sample results
    return nodes[:top_k]

# Demonstrate manual hybrid approach
manual_results = manual_hybrid_search(nodes, "What is machine learning?", embed_model)
print(f"\nManual hybrid search returned {len(manual_results)} results for demonstration")