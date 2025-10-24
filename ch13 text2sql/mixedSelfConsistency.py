import pandas as pd
import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Add project root to path

from llama_index.packs.tables.mix_self_consistency.base import MixSelfConsistencyQueryEngine
from llama_index.llms.openai import OpenAI
import asyncio

'''
from llama_index.core.llama_pack import download_llama_pack
download_llama_pack(  
        "MixSelfConsistencyPack",  
        "./mix_self_consistency_pack",  
    )  
'''

# Initialize an LLM (ensure your OPENAI_API_KEY is set in the environment)
llm = OpenAI(temperature=0.7)      

# Create a sample table using pandas DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Score": [85, 90, 78],
    "Age": [25, 30, 22]
}
df = pd.DataFrame(data)
      
query_engine = MixSelfConsistencyQueryEngine(  
    df=df,  
    llm=llm,  
    text_paths=5, # sampling 5 textual reasoning paths  
    symbolic_paths=5, # sampling 5 symbolic reasoning paths  
    aggregation_mode="self-consistency", # aggregates results across both text and symbolic paths via self-consistency (i.e. majority voting)  
    verbose=True,  
)  


async def main():
    response = await query_engine.aquery("what is the average age of all persons?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())