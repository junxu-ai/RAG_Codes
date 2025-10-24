from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  
from llama_index.core.output_parsers import LangchainOutputParser  
from llama_index.llms.openai import OpenAI  
from langchain.output_parsers import StructuredOutputParser, ResponseSchema  
    
# load documents, build index  
documents = SimpleDirectoryReader(r"./codes/data/").load_data()  
index = VectorStoreIndex.from_documents(documents)  
    
# define output schema  
response_schemas = [  
    ResponseSchema(  
        name="application",  
        description="Describes the bitcoin application used.",  
    ),  
    ResponseSchema(  
        name="methodology",  
        description="Describes the bitcoin methodology used.",  
    ),  
]  
    
# define output parser  
lc_output_parser = StructuredOutputParser.from_response_schemas(  
    response_schemas  
)  
output_parser = LangchainOutputParser(lc_output_parser)  
    
# Attach output parser to LLM  
llm = OpenAI(output_parser=output_parser)  
    
# obtain a structured response  
query_engine = index.as_query_engine(llm=llm)  
response = query_engine.query(  
    "What are the key points described in this paper?",  
)  
print(str(response))