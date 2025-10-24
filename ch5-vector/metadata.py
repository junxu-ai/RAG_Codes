# pip install langchain llama-index openai faiss-cpu
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.vectorstores import FAISS
from llama_index.core import Document, SimpleVectorIndex
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

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
        extra_info=doc["metadata"]  # Embedding metadata
    )
    for doc in documents
]

# Build a LlamaIndex index
index = SimpleVectorIndex.from_documents(llama_documents)

# Alternatively, build a FAISS index using LangChain
embeddings = OpenAIEmbeddings()
texts = [doc["content"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Step 3: Define Metadata Filtering Function
def retrieve_for_user(user_entitlement, query):
    """
    Retrieve documents the user is entitled to access.
    Args:
        user_entitlement (str): The user's entitlement level.
        query (str): The user's query.

    Returns:
        List of documents.
    """
    # Define a filter function for metadata
    def filter_function(doc):
        return doc["metadata"]["entitlement"] == user_entitlement

    # Query the LlamaIndex or FAISS index with a filter
    if isinstance(index, SimpleVectorIndex):
        results = index.query(query, similarity_top_k=3)
        filtered_results = [
            doc for doc in results if filter_function(doc.extra_info)
        ]
    else:  # FAISS
        retriever = vectorstore.as_retriever()
        retriever.search_type = "similarity"
        retriever.search_kwargs = {"k": 3}
        results = retriever.get_relevant_documents(query)
        filtered_results = [doc for doc in results if filter_function(doc.metadata)]
    
    return filtered_results

# Step 4: Query with Metadata Filtering
query = "What documents can I access?"
user_entitlement = "manager"  # Simulate a user with 'manager' access

filtered_docs = retrieve_for_user(user_entitlement, query)

# Step 5: Display Results
print("Documents user is entitled to access:")
for doc in filtered_docs:
    print(doc.text)
