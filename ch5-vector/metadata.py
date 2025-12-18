# pip install langchain langchain-openai llama-index faiss-cpu
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from llama_index.core import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQueryMode
import os

# Make sure to set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Step 1: Define Documents with Metadata
documents = [
    {"content": "Document for general users.", "metadata": {"entitlement": "general"}},
    {"content": "Confidential document for managers.", "metadata": {"entitlement": "manager"}},
    {"content": "Highly sensitive document for admins.", "metadata": {"entitlement": "admin"}}
]

# Step 2: Embed and Index Documents
# Create LlamaIndex documents with metadata
llama_documents = [
    Document(
        text=doc["content"],
        metadata=doc["metadata"]  # Note: using metadata instead of extra_info
    )
    for doc in documents
]

# Build a LlamaIndex index
# Using VectorStoreIndex instead of deprecated SimpleVectorIndex
index = VectorStoreIndex.from_documents(llama_documents)

# Alternatively, build a FAISS index using LangChain
embeddings = OpenAIEmbeddings()
texts = [doc["content"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Step 3: Define Metadata Filtering Function
def retrieve_for_user(user_entitlement, query, use_faiss=False):
    """
    Retrieve documents the user is entitled to access.
    Args:
        user_entitlement (str): The user's entitlement level.
        query (str): The user's query.
        use_faiss (bool): Whether to use FAISS or LlamaIndex.

    Returns:
        List of documents.
    """
    if use_faiss:
        # For FAISS with metadata filtering
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
                "filter": lambda x: x.get("entitlement") == user_entitlement
            }
        )
        results = retriever.invoke(query)
        return results
    else:
        # For LlamaIndex with metadata filtering
        # Note: Actual metadata filtering would need to be done differently
        # This is a simplified approach
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        # For demonstration, we'll return all nodes (filtering would require more complex setup)
        return response

# Step 4: Query with Metadata Filtering
query = "What documents can I access?"
user_entitlement = "manager"  # Simulate a user with 'manager' access

# Using FAISS approach (more straightforward for metadata filtering)
filtered_docs = retrieve_for_user(user_entitlement, query, use_faiss=True)

# Step 5: Display Results
print("Documents user is entitled to access:")
for doc in filtered_docs:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")