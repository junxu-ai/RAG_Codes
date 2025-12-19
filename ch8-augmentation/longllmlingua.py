# to be verified due to the large model size

from llama_index.core.query_engine import RetrieverQueryEngine  
from llama_index.core.response_synthesizers import CompactAndRefine  
# pip install llama-index-postprocessor-longllmlingua
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor  
from llama_index.core import QueryBundle  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/ml").load_data()
index = VectorStoreIndex.from_documents(documents)
# Create retriever from index
retriever = index.as_retriever(similarity_top_k=5)

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

# Define the query
query_str = "What are the main applications of AI in finance?"

# Retrieve the nodes based on the user's query
retrieved_nodes = retriever.retrieve(query_str)  
# synthesizer = CompactAndRefine()  

# Step 1: Postprocess the retrieved nodes to compress and refine the context using LongLLMLingua
compressed_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=query_str)
)

# Display the compressed content for review/debugging
print("\n\n".join([n.get_content() for n in compressed_nodes]))

# Step 2: Synthesize the final response based on the compressed context
# response = synthesizer.synthesize(query_str, compressed_nodes)

# Step 3: Postprocess the retrieved nodes to compress and refine the context using LongLLMLingua
print(f"\nAfter compression, {len(compressed_nodes)} nodes remain")

# Display the compressed content for review/debugging
print("\n=== Compressed Content ===")
for i, node in enumerate(compressed_nodes):
    print(f"\n--- Compressed Node {i+1} ---")
    content = node.get_content()
    print(content)

# If you want to compare the token count reduction:
print("\n=== Compression Summary ===")
original_lengths = [len(node.get_content().split()) for node in retrieved_nodes]
compressed_lengths = [len(node.get_content().split()) for node in compressed_nodes]

total_original_words = sum(original_lengths)
total_compressed_words = sum(compressed_lengths)

print(f"Total words before compression: {total_original_words}")
print(f"Total words after compression: {total_compressed_words}")
print(f"Compression ratio: {total_compressed_words/total_original_words:.2%}")


