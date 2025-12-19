## Hierarchical Retrieval Implementation with high-version of LlamaIndex

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document
)
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Dict
import uuid

# Initialize the LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo", max_tokens=750)
embed_model = OpenAIEmbedding(embed_batch_size=128)

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

# Add document IDs for tracking
for i, doc in enumerate(documents):
    if doc.metadata is None:
        doc.metadata = {}
    doc.metadata["doc_id"] = f"doc_{i}"

# Split documents into chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = text_splitter.get_nodes_from_documents(documents)

# Create a Chunk Index (VectorStoreIndex) for the document chunks
chunk_index = VectorStoreIndex(nodes, embed_model=embed_model)

# Generate summaries for each document using LLM
def generate_summaries(documents, llm):
    summaries = []
    for i, doc in enumerate(documents):
        try:
            summary_prompt = f"Provide a concise summary of the following document:\n\n{doc.text[:2000]}"
            summary = llm.complete(summary_prompt).text
            summary_doc = Document(
                text=summary,
                metadata={"doc_id": doc.metadata["doc_id"], "doc_type": "summary"}
            )
            summaries.append(summary_doc)
        except Exception as e:
            print(f"Error generating summary for document {i}: {e}")
            # Fallback to first 500 characters as summary
            summary_doc = Document(
                text=doc.text[:500] + "...",
                metadata={"doc_id": doc.metadata["doc_id"], "doc_type": "summary"}
            )
            summaries.append(summary_doc)
    return summaries

# Create summaries for documents
document_summaries = generate_summaries(documents, llm)

# Create a Summary Index (VectorStoreIndex) for the document summaries
summary_index = VectorStoreIndex.from_documents(document_summaries, embed_model=embed_model)

# Create retrievers for both indices
summary_retriever = summary_index.as_retriever(similarity_top_k=3)
chunk_retriever = chunk_index.as_retriever(similarity_top_k=10)

# Define a function to perform hierarchical retrieval
def hierarchical_retrieval(query: str, top_k: int = 5):
    """
    Perform hierarchical retrieval:
    1. First retrieve relevant document summaries
    2. Then retrieve detailed chunks from those documents
    """
    print(f"Performing hierarchical retrieval for query: {query}")
    
    # First Step: Retrieve relevant summaries
    print("Step 1: Retrieving relevant document summaries...")
    summary_nodes = summary_retriever.retrieve(query)
    print(f"Found {len(summary_nodes)} relevant summaries")
    
    # Extract document identifiers from the retrieved summaries
    doc_ids = list(set([node.metadata["doc_id"] for node in summary_nodes]))  # Use set to deduplicate
    print(f"Identified document IDs: {doc_ids}")
    
    # Second Step: Retrieve relevant chunks from the identified documents
    print("Step 2: Retrieving detailed chunks from identified documents...")
    
    # Collect all relevant chunks
    relevant_chunks = []
    
    # Get chunks for each identified document
    for doc_id in doc_ids:
        # Retrieve all chunks for this document and then filter by query relevance
        all_chunks_for_doc = chunk_index.as_retriever(
            similarity_top_k=10  # Get more chunks initially
        ).retrieve(query)
        
        # Filter to only include chunks from this specific document
        doc_chunks = [chunk for chunk in all_chunks_for_doc if chunk.metadata.get("doc_id") == doc_id]
        relevant_chunks.extend(doc_chunks[:5])  # Take top 5 chunks per document
    
    # Sort by similarity score and return top_k
    relevant_chunks.sort(key=lambda x: x.score or 0, reverse=True)
    
    return relevant_chunks[:top_k]

# Alternative implementation with proper filtering (if supported by your LlamaIndex version)
def hierarchical_retrieval_v2(query: str, top_k: int = 5):
    """
    Alternative hierarchical retrieval implementation
    """
    print(f"Performing hierarchical retrieval for query: {query}")
    
    # First Step: Retrieve relevant summaries
    print("Step 1: Retrieving relevant document summaries...")
    summary_nodes = summary_retriever.retrieve(query)
    print(f"Found {len(summary_nodes)} relevant summaries")
    
    # Extract document identifiers
    doc_ids = list(set([node.metadata["doc_id"] for node in summary_nodes]))
    print(f"Identified document IDs: {doc_ids}")
    
    # Second Step: Simple approach - retrieve top chunks without strict filtering
    print("Step 2: Retrieving relevant chunks...")
    all_chunks = chunk_retriever.retrieve(query)
    
    # Filter chunks to only those from identified documents
    relevant_chunks = [chunk for chunk in all_chunks if chunk.metadata.get("doc_id") in doc_ids]
    
    # If we don't have enough relevant chunks, add some from the top results
    if len(relevant_chunks) < top_k:
        relevant_chunks.extend(all_chunks[:top_k - len(relevant_chunks)])
    
    # Sort by similarity score and return top_k
    relevant_chunks.sort(key=lambda x: x.score or 0, reverse=True)
    
    return relevant_chunks[:top_k]

# Example usage
if __name__ == "__main__":
    query = "Explain how machine learning and deep learning differ in their approaches"
    print(f"Query: {query}\n")
    
    try:
        # Try the first version
        relevant_chunks = hierarchical_retrieval_v2(query, top_k=5)
        
        print("\nRetrieved Chunks:")
        print("=" * 60)
        for i, chunk in enumerate(relevant_chunks, 1):
            print(f"Chunk {i} (Score: {chunk.score:.4f}):")
            print(f"Document ID: {chunk.metadata.get('doc_id', 'N/A')}")
            print(f"Text: {str(chunk.text)[:300]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error during hierarchical retrieval: {e}")
        import traceback
        traceback.print_exc()
        
    # Compare with standard retrieval
    print("\n\nComparison with Standard Retrieval:")
    print("=" * 60)
    try:
        standard_chunks = chunk_retriever.retrieve(query)
        print(f"Standard retrieval found {len(standard_chunks)} chunks")
        for i, chunk in enumerate(standard_chunks[:3], 1):
            print(f"Chunk {i} (Score: {chunk.score:.4f}): {str(chunk.text)[:200]}...")
    except Exception as e:
        print(f"Error during standard retrieval: {e}")