# to be verified due to the large model size

from llama_index.core.query_engine import RetrieverQueryEngine  
from llama_index.core.response_synthesizers import CompactAndRefine  
# pip install llama-index-postprocessor-longllmlingua
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor  
from llama_index.core import QueryBundle  

# Configure the LongLLMLingua postprocessor to compress retrieved context
node_postprocessor = LongLLMLinguaPostprocessor(
    # Instruction prompt specifying how to compress the context and focus on the relevant query
    instruction_str="Compress the context and provide the most relevant response to the user's query.",
    target_token=300,  # Limit the compressed context to approximately 300 tokens
    rank_method="longllmlingua",  # Use LongLLMLingua's optimized ranking and compression mechanism
    device_map="cpu" ,  # Specify the device for processing (CPU or GPU)
    additional_compress_kwargs={
        "condition_compare": True,  # Enable comparison between contextual segments for relevance
        "condition_in_question": "after",  # Compress the context after retrieval for efficiency
        "context_budget": "+100",  # Budget adjustment to allow for additional context expansion
        "reorder_context": "sort",  # Reorder documents to prioritize the most relevant context
    },
)

# Retrieve the nodes based on the user's query
retrieved_nodes = retriever.retrieve(query_str)  
synthesizer = CompactAndRefine()  

# Step 1: Postprocess the retrieved nodes to compress and refine the context using LongLLMLingua
compressed_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=query_str)
)

# Display the compressed content for review/debugging
print("\n\n".join([n.get_content() for n in compressed_nodes]))

# Step 2: Synthesize the final response based on the compressed context
response = synthesizer.synthesize(query_str, compressed_nodes)


