import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import prompts as prompts  ## from the local prompts.py file
from concurrent.futures import ThreadPoolExecutor

## Jina Reranker Example. 
# you can either use the llamaindex built-in function or the native api call. 
# !pip install llama-index-postprocessor-jinaai-rerank
# !pip install llama-index-embeddings-jinaai
# from llama_index.postprocessor.jina_rerank import JinaRerank
# from llama_index.embeddings.jinaai import JinaEmbedding
# see example from https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/JinaRerank/
# the code below use native api call to Jina Reranker
class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

## Cohere Reranker Example
# you can either use the llamaindex built-in function or the native api call. below is a code using llamaindex

# from llama_index.postprocessor.cohere_rerank import CohereRerank
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# class RerankQueryEngine:
#     def __init__(self, api_key, data_dir="./data/", top_n=2, similarity_top_k=10):
#         """
#         Initializes the RerankQueryEngine.

#         Args:
#             api_key (str): The Cohere API key.
#             data_dir (str): The directory containing the documents.
#             top_n (int): The number of nodes to return from the reranker.
#             similarity_top_k (int): The number of nodes to retrieve before reranking.
#         """
#         self.api_key = api_key
#         self.data_dir = data_dir
#         self.top_n = top_n
#         self.similarity_top_k = similarity_top_k
        
#         self.documents = SimpleDirectoryReader(self.data_dir).load_data()
#         self.index = VectorStoreIndex.from_documents(documents=self.documents)
#         self.cohere_rerank = CohereRerank(api_key=self.api_key, top_n=self.top_n)
        
#         self.query_engine = self.index.as_query_engine(
#             similarity_top_k=self.similarity_top_k,
#             node_postprocessors=[self.cohere_rerank],
#         )

#     def query(self, query_str):
#         """
#         Queries the engine with a given string.

#         Args:
#             query_str (str): The query to execute.

#         Returns:
#             The response from the query engine.
#         """
#         return self.query_engine.query(query_str)

class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
      
    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_single_block
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
        
        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
    """ Rerank multiple text blocks using LLM. Here's a succinct breakdown:
            It formats the retrieved text blocks into a single string with clear block boundaries.
            It constructs a user prompt that includes the query, the formatted text blocks, and instructions for the LLM to provide rankings.
            It sends the prompt to the LLM using the completions.parse method and retrieves the response.
            It extracts the parsed response and returns it as a dictionary.
            The method is designed to handle multiple text blocks and returns a ranking of these blocks based on their relevance to the query.
    arguments:
    - query: The user query to rank the text blocks against.
    - retrieved_documents: A list of text blocks retrieved from a vector store or other source. 

    """
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ],
            response_format=self.schema_for_multiple_blocks
        )

        response = completion.choices[0].message.parsed
        response_dict = response.model_dump()
      
        return response_dict

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 10, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        Here's a succinct breakdown:

            The method takes in a query, a list of documents, and two optional parameters: documents_batch_size (default 4) and llm_weight (default 0.7).
            It creates batches of documents based on the documents_batch_size.
            If the batch size is 1, it processes each document individually using the get_rank_for_single_block method. Otherwise, it processes batches of documents using the get_rank_for_multiple_blocks method.
            For each document, it calculates a combined score by weighting the LLM relevance score and the vector similarity score (inverted since lower is better).
            It processes the documents in parallel using a ThreadPoolExecutor.
            Finally, it sorts the results by the combined score in descending order and returns the reranked list of documents.   

            arguments:
            - query: The user query to rerank the documents against.
            - documents: A list of documents to be reranked, where each document is a dictionary
              containing at least 'text' and 'distance' keys.
            - documents_batch_size: The size of each batch of documents to process (default is 10).
            - llm_weight: The weight given to the LLM relevance score in the combined score calculation (default is 0.7).     
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
