from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os

# Initialize the LLM and embedding model
llm = OpenAI(model="gpt-4-mini")
embed_model = OpenAIEmbedding()

# Load documents from a directory
try:
    documents = SimpleDirectoryReader("./data/ml").load_data()
except Exception as e:
    print(f"Could not load documents: {e}")
    # Create sample documents if directory doesn't exist
    from llama_index.core import Document
    documents = [
        Document(text="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data. It enables computers to identify patterns and make decisions with minimal human intervention. Common applications include recommendation systems, fraud detection, and predictive analytics."),
        Document(text="Deep learning uses neural networks with multiple layers to model complex patterns in data. These deep neural networks can automatically learn hierarchical representations of data. Deep learning excels in computer vision, natural language processing, and speech recognition tasks."),
        Document(text="Natural language processing enables computers to understand and generate human language. NLP techniques include sentiment analysis, machine translation, and chatbot development. Modern NLP heavily relies on transformer architectures like BERT and GPT.")
    ]

# Split documents into smaller chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = text_splitter.get_nodes_from_documents(documents)

# Create a VectorStoreIndex for the chunks
vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

# Create sample specialized query engines (simulating table/SQL query engines)
class MockTableQueryEngine:
    """Mock table query engine for demonstration"""
    def query(self, query_str):
        return "This is a mock response from a specialized table query engine. In practice, this would query structured data sources."

# Create IndexNodes for specialized retrievers/query engines
# In practice, these would point to actual specialized indices or query engines
index_nodes = []
for i, node in enumerate(nodes[:3]):  # Create sample index nodes for first 3 nodes
    index_node = IndexNode(
        text=f"Specialized content for node {i}",
        index_id=f"specialized_{i}",
        metadata={"type": "specialized_content", "node_ref": node.node_id}
    )
    index_nodes.append(index_node)

# Create specialized indices for demonstration
specialized_indices = {}
specialized_query_engines = {}

# Create a few specialized indices (simulating different data sources)
for i in range(min(2, len(documents))):
    # Create a small specialized index for demonstration
    specialized_nodes = text_splitter.get_nodes_from_documents([documents[i]])
    specialized_index = VectorStoreIndex(specialized_nodes, embed_model=embed_model)
    specialized_indices[f"specialized_{i}"] = specialized_index
    specialized_query_engines[f"specialized_{i}"] = specialized_index.as_query_engine()

# Add a mock table query engine
specialized_query_engines["table_engine"] = MockTableQueryEngine()

# Initialize the RecursiveRetriever with the vector index and specialized retrievers
retriever_dict = {
    "vector_index": vector_index.as_retriever(similarity_top_k=5),
}

# Add specialized retrievers
for idx, specialized_engine in specialized_query_engines.items():
    if hasattr(specialized_engine, 'as_retriever'):
        retriever_dict[idx] = specialized_engine.as_retriever(similarity_top_k=3)

query_engine_dict = specialized_query_engines

# Create RecursiveRetriever
recursive_retriever = RecursiveRetriever(
    retriever_dict=retriever_dict,
    query_engine_dict=query_engine_dict,
    root_id="vector_index"  # Start with vector index as root
)

# Create a RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# Example usage
if __name__ == "__main__":
    # Execute queries
    test_queries = [
        "What is machine learning?",
        "Explain deep learning applications",
        "How does NLP work with transformers?"
    ]
    
    for query_text in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query_text}")
        print('='*60)
        
        try:
            response = query_engine.query(query_text)
            print(f"Response: {str(response)}")
        except Exception as e:
            print(f"Error executing query: {e}")
            
    # Demonstrate how recursive retrieval works with specialized engines
    print(f"\n{'='*60}")
    print("DEMONSTRATION OF RECURSIVE RETRIEVAL")
    print('='*60)
    
    complex_query = "Show me data from both general ML concepts and specialized tables"
    print(f"Complex query: {complex_query}")
    
    try:
        response = query_engine.query(complex_query)
        print(f"Recursive retrieval response: {str(response)}")
    except Exception as e:
        print(f"Error in recursive retrieval: {e}")