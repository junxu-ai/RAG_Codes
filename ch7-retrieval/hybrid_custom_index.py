from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import SentenceSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

# Function to mix keyword search and semantic search
def mixed_retrieval(index, nodes, query, keyword_weight=0.5, semantic_weight=0.5):
    """
    Function to mix keyword search and semantic search
    """
    # Perform keyword search using TF-IDF
    keyword_results = keyword_search(nodes, query)
    
    # Perform semantic search using the index
    semantic_results = semantic_search(index, query)
    
    # Combine results
    combined_results = {}
    
    # Add keyword search results
    for doc_id, score in keyword_results:
        combined_score = score * keyword_weight
        if doc_id in combined_results:
            combined_results[doc_id] += combined_score
        else:
            combined_results[doc_id] = combined_score
    
    # Add semantic search results
    for doc_id, score in semantic_results:
        combined_score = score * semantic_weight
        if doc_id in combined_results:
            combined_results[doc_id] += combined_score
        else:
            combined_results[doc_id] = combined_score
    
    # Sort results by combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results

def keyword_search(nodes, query):
    """
    Perform TF-IDF based keyword search
    """
    # Extract texts from nodes
    texts = [node.text for node in nodes]
    doc_ids = [getattr(node, 'node_id', f"doc_{i}") for i, node in enumerate(nodes)]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Transform query
    query_vec = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Return results as (doc_id, score) tuples
    results = [(doc_ids[i], similarities[i]) for i in range(len(similarities)) if similarities[i] > 0]
    return sorted(results, key=lambda x: x[1], reverse=True)

def semantic_search(index, query, top_k=10):
    """
    Perform semantic search using the vector index
    """
    # Use the index's retriever for semantic search
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    
    # Extract doc_ids and scores
    semantic_results = []
    for node in results:
        doc_id = getattr(node, 'node_id', 'unknown')
        score = getattr(node, 'score', 0.0)
        semantic_results.append((doc_id, score))
    
    return semantic_results

def get_top_nodes(nodes, sorted_results, top_k=5):
    """
    Get top nodes based on sorted results
    """
    # Create a mapping from node_id to node
    node_dict = {}
    for node in nodes:
        node_id = getattr(node, 'node_id', None)
        if node_id:
            node_dict[node_id] = node
        # Also try to use index or other identifiers
        else:
            node_dict[str(id(node))] = node
    
    # Get top nodes
    top_nodes = []
    for doc_id, score in sorted_results[:top_k]:
        if doc_id in node_dict:
            top_nodes.append((node_dict[doc_id], score))
        else:
            # Try to find node by matching with available identifiers
            for node in nodes:
                if str(id(node)) == doc_id or getattr(node, 'id_', '') == doc_id:
                    top_nodes.append((node, score))
                    break
    
    return top_nodes

# Main execution
if __name__ == "__main__":
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

    # Create VectorStoreIndex
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain NLP applications"
    ]

    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        # Perform mixed retrieval
        try:
            sorted_results = mixed_retrieval(index, nodes, query, keyword_weight=0.3, semantic_weight=0.7)
            
            print(f"Retrieved {len(sorted_results)} results:")
            
            # Get top nodes with actual content
            top_nodes = get_top_nodes(nodes, sorted_results, top_k=3)
            
            for i, (node, score) in enumerate(top_nodes, 1):
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Text: {str(node.text)[:150]}...")
                print()
                
        except Exception as e:
            print(f"Error in mixed retrieval: {e}")

    # Compare with individual methods
    print("\nCOMPARISON WITH INDIVIDUAL METHODS")
    print("=" * 50)
    
    query = "Compare machine learning and deep learning"
    print(f"Query: {query}")
    
    # Keyword search only
    print("\n1. KEYWORD SEARCH ONLY:")
    keyword_results = keyword_search(nodes, query)
    top_keyword = get_top_nodes(nodes, keyword_results[:3], top_k=3)
    for i, (node, score) in enumerate(top_keyword, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Text: {str(node.text)[:100]}...")
    
    # Semantic search only
    print("\n2. SEMANTIC SEARCH ONLY:")
    semantic_results = semantic_search(index, query)
    top_semantic = get_top_nodes(nodes, semantic_results[:3], top_k=3)
    for i, (node, score) in enumerate(top_semantic, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Text: {str(node.text)[:100]}...")
    
    # Mixed retrieval
    print("\n3. MIXED RETRIEVAL (30% keyword, 70% semantic):")
    mixed_results = mixed_retrieval(index, nodes, query, keyword_weight=0.3, semantic_weight=0.7)
    top_mixed = get_top_nodes(nodes, mixed_results[:3], top_k=3)
    for i, (node, score) in enumerate(top_mixed, 1):
        print(f"  {i}. Score: {score:.4f}")
        print(f"     Text: {str(node.text)[:100]}...")