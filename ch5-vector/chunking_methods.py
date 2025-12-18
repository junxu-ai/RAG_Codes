from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from llama_index.core import Document
from llama_index.core.indices.keyword_table import KeywordTableIndex

# Sample Text for Chunking
text = """
# Introduction
This document explains different chunking methods.

## Fixed-Length Chunking
Fixed-length chunking splits text into equal-sized pieces. It's simple but may break context.

## Semantic/Sliding Window Chunking
This method uses overlapping windows to maintain context across chunks. It's useful for conversation-like text.

## Structure-Based Chunking
Text is split based on natural structure, like paragraphs or headings.

# Conclusion
Choose the method based on your data and use case.
"""

# 1. Fixed-Length Chunking
def fixed_length_chunking(text, chunk_size):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return splitter.split_text(text)

# 2. Semantic/Sliding Window Chunking
def sliding_window_chunking(text, chunk_size, overlap):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# 3. Structure-Based Chunking
def structure_based_chunking(text):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")])
    return splitter.split_text(text)

# 4. Custom/Hybrid Chunking
def custom_chunking(text, chunk_size, overlap, split_by_structure=False):
    if split_by_structure:
        md_chunks = structure_based_chunking(text)
        # Extract content from markdown splits
        return [chunk.page_content for chunk in md_chunks]
    else:
        return sliding_window_chunking(text, chunk_size, overlap)

# Apply Chunking Methods
fixed_chunks = fixed_length_chunking(text, chunk_size=100)
sliding_chunks = sliding_window_chunking(text, chunk_size=100, overlap=20)
structured_chunks = structure_based_chunking(text)
custom_chunks = custom_chunking(text, chunk_size=120, overlap=40, split_by_structure=True)

# Display Results
print("Fixed-Length Chunking:")
for i, chunk in enumerate(fixed_chunks):
    print(f"Chunk {i+1}: {repr(chunk)}\n---")

print("\nSliding Window Chunking:")
for i, chunk in enumerate(sliding_chunks):
    print(f"Chunk {i+1}: {repr(chunk)}\n---")

print("\nStructure-Based Chunking:")
for i, chunk in enumerate(structured_chunks):
    print(f"Chunk {i+1}: Content={repr(chunk.page_content)}, Metadata={chunk.metadata}\n---")

print("\nCustom/Hybrid Chunking:")
for i, chunk in enumerate(custom_chunks):
    print(f"Chunk {i+1}: {repr(chunk)}\n---")

# Use the Chunks in LlamaIndex
llama_documents = [Document(text=chunk) for chunk in fixed_chunks]  # Using fixed chunks for simplicity
index = KeywordTableIndex.from_documents(documents=llama_documents)

# Query the Index
query_engine = index.as_query_engine()
response = query_engine.query("What is sliding window chunking?")
print("\nQuery Result:")
print(str(response))