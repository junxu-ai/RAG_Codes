import pandas as pd
from llama_index.packs.tables import ChainOfTablePack
from llama_index.llms.openai import OpenAI


# Create a sample table using pandas DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Score": [85, 90, 78],
    "Age": [25, 30, 22]
}
df = pd.DataFrame(data)

# Initialize the LLM (for example, using OpenAI's GPT model)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

"""
you may download the ChainOfTablePack from the LlamaIndex CLI using the following command:

llamaindex-cli download-llamapack ChainOfTablePack --download-dir ./chain_of_table_pack

or use the following code snippet to download the ChainOfTablePack:
from llama_index.core.llama_pack import download_llama_pack
ChainOfTablePack = download_llama_pack("ChainOfTablePack", "./chain_of_table_pack")

"""

# Create the ChainOfTablePack instance with the table and LLM
chain_pack = ChainOfTablePack(table=df, llm=llm, verbose=True)

# Define a query that the pack will process on the table
query = "Which student has the highest score?"
response = chain_pack.run(query)

# Print the response from the ChainOfTablePack
print("Response:", response)
