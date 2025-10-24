from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

# load documents
documents = SimpleDirectoryReader(".\codes\data\ml").load_data()
index = VectorStoreIndex.from_documents(documents)

from llama_index.core.postprocessor import LongContextReorder  
      
reorder = LongContextReorder()  
    
reorder_engine = index.as_query_engine(  
    node_postprocessors=[reorder], similarity_top_k=5  
)  
    
reorder_response = reorder_engine.query("how to build a machine learning model efficiently?")

print(reorder_response)