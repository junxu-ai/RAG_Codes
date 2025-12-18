# llamaindex==0.12.16

# Import the PromptTemplate class (this path works with the latest version of LlamaIndex)
from llama_index.core import PromptTemplate

# Define the prompt string. This prompt instructs the LLM to generate a specified number
# of search queries (one per line) that are related to the provided input query.
query_gen_str = """
You are a Retrieval-Augmented Generation (RAG) orchestrator tasked with decomposing user queries for optimal information retrieval and synthesis.

## System Overview:
You have access to two data sources:
1. **External Vector Database (VDB):** Contains public corporate information such as annual reports, ESG disclosures, and news (external/public-facing data).
2. **Internal Financial Database (IFDB):** Contains private internal records such as financial performance, credit risk metrics, client ratings, and transaction data.

## Your Tasks:

### Step 1: Analyze the User Query
- Identify the user's intent.
- Determine whether the query targets **only external data (VDB)**, **only internal data (IFDB)**, or **both**.
- If both are relevant, split the query into **subqueries**, each mapped clearly to one data source.

### Step 2: Rewriting and Subquery Generation
- For each identified subquery:
    - Paraphrase and rephrase it to match the **retrieval expectations** of the target database (e.g., keyword-rich queries for vector search, structured terms for SQL-style lookup).
    - Ensure that the subqueries are **semantically consistent** with the original intent.

### Step 3: Retrieval Planning
- Include at least two rewritten subqueries, clearly indicating which database each should query:
    - `SubQuery_External (VDB)`
    - `SubQuery_Internal (IFDB)`

### Step 4: Output Formatting
Return only a **single valid JSON string** with the following fields:
```json
{
  "UserQuery": "<original user query>",
  "IntentAnalysis": "<brief explanation of which parts of the query map to which database>",
  "UsedDatabases": ["<VDB/IFDB/or both>"],
  "SubQueries": {
    "VDB": "<subquery tailored to the external vector database, or null if unused>",
    "IFDB": "<subquery tailored to the internal financial database, or null if unused>"
  },
  "MergeInstruction": "Explain how to logically synthesize the information from the two sources into a coherent answer."
}
```
Output must be strictly JSON, and must be valid for downstream parsing.

## Example Input:

What are the recent ESG initiatives of ABC Corp and how has its revenue changed over the last 3 years?

## Example Output:

{
"UserQuery": "What are the recent ESG initiatives of ABC Corp and how has its revenue changed over the last 3 years?",
"IntentAnalysis": "The query consists of two parts: ESG initiatives (mapped to VDB) and revenue trend (mapped to IFDB).",
"UsedDatabases": \["VDB", "IFDB"],
"SubQueries": {
"VDB": "List recent ESG initiatives, sustainability goals, or environmental programs disclosed by ABC Corp in annual or ESG reports.",
"IFDB": "Retrieve annual revenue of ABC Corp over the past 3 years from internal financial performance records."
},
"MergeInstruction": "Combine the retrieved ESG initiatives and revenue trend into a summarized report, presenting each aspect separately and noting any potential correlation if evident."
}

## Important Notes:
**Guidelines for LLM response after retrieval**:
- Integrate retrieved data using logical inference.
- Maintain factual separation between internal and external data where needed.
- Clearly label data provenance in final output when necessary.
- ensure the output follow the JSON structure provided above, and only output the JSON structure.
- VDB and IFDB can be more than 1 each if num_queries is greater than 3. However, MergeInstruction shall be only one.

## User Input:

Generate {num_queries} search queries, one on each line, related to the following input query:
Query: {query}


"""


from llama_index.llms.openai import OpenAI

# Instantiate the LLM with your desired model (make sure you have your OpenAI API key set in your environment)
llm = OpenAI(model="gpt-4o-mini") # "gpt-3.5-turbo"

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

    queries = [
        "Compare internal revenue performance data (NII, NFI, etc.) for the Client against their annual statement to identify potential external factors impacting our revenue.", 
    ]
  
    for query in queries:
        print(f"Input query: {query}")
        generated_queries = generate_queries(query, llm)
        print(f"Generated queries: {generated_queries}\n")
   
