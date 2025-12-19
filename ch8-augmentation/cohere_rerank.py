# to be verified 
# https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank/
import os  
from llama_index.postprocessor.cohere_rerank import CohereRerank  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response
    
documents = SimpleDirectoryReader("./data/ml").load_data()

# build index
index = VectorStoreIndex.from_documents(documents=documents)

# Get API key with error handling
try:
    api_key = os.environ["COHERE_API_KEY"]
except KeyError:
    raise ValueError("Please set the COHERE_API_KEY environment variable")

cohere_rerank = CohereRerank(api_key=api_key, top_n=2) # return top 2 nodes from reranker  
    
query_engine = index.as_query_engine(  
    similarity_top_k=10, # we can set a high top_k here to ensure maximum relevant retrieval  
    node_postprocessors=[cohere_rerank], # pass the reranker to node_postprocessors  
)  
    
response = query_engine.query(  
    "What is MLOps's role in a machine learning project?",  
)
print(response)



"""
below is a class version of the above code
"""

"""
import os
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.pprint_utils import pprint_response

class RerankQueryEngine:
    def __init__(self, data_dir="./data/", top_n=2, similarity_top_k=10):
        """
        Initializes the RerankQueryEngine.

        Args:
            data_dir (str): The directory containing the documents.
            top_n (int): The number of nodes to return from the reranker.
            similarity_top_k (int): The number of nodes to retrieve before reranking.
        """
        self.data_dir = data_dir
        self.top_n = top_n
        self.similarity_top_k = similarity_top_k
        
        # Load documents and build index
        self.documents = SimpleDirectoryReader(self.data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents=self.documents)
        
        # Get API key
        try:
            api_key = os.environ["COHERE_API_KEY"]
        except KeyError:
            raise ValueError("Please set the COHERE_API_KEY environment variable")
            
        # Initialize reranker
        self.cohere_rerank = CohereRerank(api_key=api_key, top_n=self.top_n)
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            node_postprocessors=[self.cohere_rerank],
        )

    def query(self, query_str):
        """
        Queries the engine with a given string.

        Args:
            query_str (str): The query to execute.

        Returns:
            The response from the query engine.
        """
        return self.query_engine.query(query_str)
    
    def query_and_print(self, query_str):
        """
        Queries the engine and prints formatted response.

        Args:
            query_str (str): The query to execute.
        """
        response = self.query(query_str)
        pprint_response(response)
        return response

# Usage example:
# engine = RerankQueryEngine()
# response = engine.query("What does AI can do for financial institutions?")

"""