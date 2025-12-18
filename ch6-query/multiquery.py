# llamaindex==0.12.16

# Import the PromptTemplate class (this path works with the latest version of LlamaIndex)
from llama_index.core import PromptTemplate

# Define the prompt string. This prompt instructs the LLM to generate a specified number
# of search queries (one per line) that are related to the provided input query.
query_gen_str = """\
You are a helpful assistant that generates multiple search queries based on a single input query.
Generate {num_queries} search queries, one on each line, related to the following input query:
Query: {query}
Queries:
"""
from llama_index.llms.openai import OpenAI

# Instantiate the LLM with your desired model (make sure you have your OpenAI API key set in your environment)
llm = OpenAI(model="gpt-3.5-turbo")

# Now call the generate_queries function with the defined llm instance.

# Create a PromptTemplate instance from the string.
query_gen_prompt = PromptTemplate(query_gen_str)

def generate_queries(query: str, llm, num_queries: int = 5):
    """
    Generates multiple search queries from a single input query using an LLM.

    Parameters:
        query (str): The input query for which to generate search queries.
        llm: An instance of an LLM that supports the 'predict' method.
        num_queries (int): The number of queries to generate (default is 5).

    Returns:
        list[str]: A list of generated search queries.
    """
    # Get the LLM's response by passing the prompt template with the required parameters.
    response = llm.predict(
        query_gen_prompt,
        num_queries=num_queries,
        query=query
    )
    
    # Split the response by newlines and remove any empty or whitespace-only lines.
    queries = [q.strip() for q in response.split("\n") if q.strip()]
    
    # Join the queries into a single string for display purposes.
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    
    return queries

# Example usage:
# (Make sure that 'llm' is defined. For example, you can create an LLM instance as follows:
#  from llama_index.llms.openai import OpenAI
#  llm = OpenAI(model="gpt-3.5-turbo"))
if __name__ == '__main__':
    # Replace this with your actual LLM instance
    # For demonstration, assume llm is already defined.
    queries = generate_queries("How to build a machine learning model efficiently?", llm)
