## Hierarchical Retrieval Implementation with low-version of LlamaIndex

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
    RecursiveRetriever,
    RetrieverQueryEngine,
    StorageContext,
    Document
)
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import IndexNode, MetadataFilters, MetadataFilter, FilterOperator

# Initialize the LLM and service context
llm_predictor = LLMPredictor(llm=OpenAI(model="gpt-3.5-turbo", max_tokens=750))
embed_model = OpenAIEmbedding(embed_batch_size=128)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

# Load documents from a directory
documents = SimpleDirectoryReader("path_to_your_documents").load_data()

# Split documents into sentences for chunking
text_splitter = SentenceSplitter(chunk_size=1024)
chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc.text))

# Create a Chunk Index (VectorStoreIndex) for the document chunks
chunk_index = VectorStoreIndex.from_documents(chunks, service_context=service_context)

# Generate summaries for each document
summaries = []
for doc in documents:
    summary = llm_predictor.predict(f"Summarize the following document:\n\n{doc.text}")
    summaries.append(summary)

# Create a Summary Index (VectorStoreIndex) for the document summaries
summary_index = VectorStoreIndex.from_documents(summaries, service_context=service_context)

# Create a retriever for the Summary Index
summary_retriever = summary_index.as_retriever(similarity_top_k=5)

# Create a retriever for the Chunk Index
chunk_retriever = chunk_index.as_retriever(similarity_top_k=5)

# Define a function to perform hierarchical retrieval
def hierarchical_retrieval(query):
    # First Step: Retrieve relevant summaries
    summary_nodes = summary_retriever.retrieve(query)
    
    # Extract document identifiers from the retrieved summaries
    doc_ids = [node.metadata["doc_id"] for node in summary_nodes]
    
    # Second Step: Retrieve relevant chunks from the identified documents
    filters = MetadataFilters(
        filters=[MetadataFilter(key="doc_id", operator=FilterOperator.IN, value=doc_ids)]
    )
    chunk_retriever_with_filters = chunk_index.as_retriever(filters=filters)
    chunk_nodes = chunk_retriever_with_filters.retrieve(query)
    
    return chunk_nodes

# Example usage
query = "Your query here"
relevant_chunks = hierarchical_retrieval(query)

# Process the retrieved chunks as needed
for chunk in relevant_chunks:
    print(chunk.text)
