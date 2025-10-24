# llamaindex==0.12.16

from llama_index.core.indices import GPTListIndex, SimpleKeywordTableIndex
from llama_index.core import Document, VectorStoreIndex
from langchain.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.keyword_table import KeywordTableGPTRetriever 

# Step 1: Initialize the LLM for rewriting and retrieval
llm_rewriter = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)  # Use GPT-4 for query rewriting
llm_retriever = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Use GPT-3.5 for retrieval and reading

# Step 2: Define Documents (Simulated Data for Testing)
documents = [
    Document(text="LangChain is a framework for building applications with LLMs, including prompt management and chains."),
    Document(text="The NBA champion of 2020 is the Los Angeles Lakers."),
    Document(text="LlamaIndex provides tools for data indexing and retrieval optimized for LLM interactions.")
]

# Step 3: Create Index for Retrieval
index = SimpleKeywordTableIndex.from_documents(documents)

# Step 4: Define the Rewrite-Retrieve-Read Pipeline
class RewriteRetrieveReadPipeline:
    def __init__(self, rewriter, retriever, index):
        self.rewriter = rewriter
        self.retriever = retriever  # Use KeywordTableRetriever
        self.index = index
        self.query_engine = RetrieverQueryEngine(  # Ensure RetrieverQueryEngine is imported
            retriever=self.retriever,
        )

    def rewrite_query(self, query):
        # Use LLM to rewrite the query
        rewriter_prompt = f"Rewrite the following query to make it concise and retrieval-focused:\n\n{query}\n\nRewritten Query:"
        rewritten_query = self.rewriter.predict(rewriter_prompt)
        return rewritten_query.strip()

    def retrieve_documents(self, rewritten_query):
        # Retrieve relevant documents based on rewritten query
        return self.retriever.retrieve(rewritten_query)

    def read_documents(self, retrieved_docs, query):
        # Use LLM to synthesize answers based on retrieved documents
        doc_texts = " ".join([doc.text for doc in retrieved_docs])
        reader_prompt = f"Based on the following information:\n\n{doc_texts}\n\nAnswer the query:\n{query}\n\nResponse:"
        response = llm_retriever.predict(reader_prompt)
        return response.strip()

    def execute_pipeline(self, query):
        # Execute the full Rewrite-Retrieve-Read pipeline
        rewritten_query = self.rewrite_query(query)
        print(f"Rewritten Query: {rewritten_query}")
        retrieved_docs = self.retrieve_documents(rewritten_query)
        final_response = self.read_documents(retrieved_docs, query)
        return final_response

# Step 5: Initialize and Execute the Pipeline
retriever = KeywordTableGPTRetriever(index=index)  # Initialize the correct retriever
pipeline = RewriteRetrieveReadPipeline(llm_rewriter, retriever, index)

user_query = "The NBA champion of 2020 is the Los Angeles Lakers! Tell me what is Langchain framework?"
response = pipeline.execute_pipeline(user_query)

print("\nFinal Response:")
print(response)
