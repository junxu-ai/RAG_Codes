from importlib.metadata import version
from packaging.version import parse
import os

# Version-dependent imports for backward compatibility.
if parse(version('llama_index')) < parse('0.10'):
    from llama_index import ServiceContext, VectorStoreIndex, StorageContext, Document
    from llama_index.node_parser import SentenceWindowNodeParser
    from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
    from llama_index.indices.postprocessor import SentenceTransformerRerank
    from llama_index import load_index_from_storage
    from llama_index import SimpleDirectoryReader
else:
    from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage, Document
    from llama_index.core.node_parser import SentenceWindowNodeParser
    from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
    from llama_index.core import SimpleDirectoryReader
    from llama_index.llms.openai import OpenAI

def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=5,
    save_dir="sentence_index",
    rebuild=True
):
    """
    Build or load a VectorStoreIndex using a sentence window node parser.
    
    Parameters:
        documents (list[Document]): List of Document objects to be indexed.
        llm: An LLM instance used to create the service context.
        embed_model (str): Identifier for the embedding model.
        sentence_window_size (int): Number of sentences per window.
        save_dir (str): Directory path where the index will be persisted.
        rebuild (bool): If True, rebuild the index even if a saved index exists.
    
    Returns:
        VectorStoreIndex: The built or loaded index.
    """
    # Create the SentenceWindowNodeParser with default settings.
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    
    # If the save directory does not exist or rebuild is requested, build the index and persist it.
    if (not os.path.exists(save_dir)) or rebuild:
        print('Building the vector index from documents...')
        sentence_index = VectorStoreIndex.from_documents(
            documents
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        print('Loading the vector index from storage...')
        storage_ctx = StorageContext.from_defaults(persist_dir=save_dir)
        sentence_index = load_index_from_storage(storage_ctx, service_context=sentence_context)
    
    print("Index details:")
    print(sentence_index)
    return sentence_index

def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
    verbose=True
):
    """
    Build a query engine using the given index with additional postprocessors.
    
    Parameters:
        sentence_index (VectorStoreIndex): The index built with a sentence window node parser.
        similarity_top_k (int): Number of top similar nodes to retrieve.
        rerank_top_n (int): Number of nodes to rerank using the SentenceTransformer-based reranker.
        verbose (bool): If True, display verbose output.
    
    Returns:
        QueryEngine: The query engine with the applied postprocessors.
    """
    # Define a postprocessor to replace metadata based on a target key.
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    # Define a reranker that uses a SentenceTransformer model for reranking.
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-MiniLM-L-2-v2"  ## alternatively, you can use "BAAI/bge-reranker-base"
    )
    print('Building the sentence window query engine...')
    # Build the query engine from the index, applying the defined postprocessors.
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        node_postprocessors=[postproc, rerank],
        verbose=verbose,
    )
    return sentence_window_engine
def load_documents_from_files(directory):
    """
    Load text files from a given directory and convert them into a list of Document objects.
    
    Parameters:
        directory (str): Path to the directory containing text files.
    
    Returns:
        list[Document]: List of Document objects with file content and metadata.
    """
    documents = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        # Check if it's a file and (optionally) ends with .txt (or any desired extension)
        if os.path.isfile(file_path) and file_name.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Create a Document and add file metadata
            documents.append(Document(text=text, metadata={"file_name": file_name}))
    return documents

# Example usage:
if __name__ == '__main__':

    # Directory containing text files.
    files_directory = "./codes/data/ml"
    
    # Load documents from the files in the directory.
    documents = load_documents_from_files(files_directory)
    print(f"Loaded {len(documents)} documents from {files_directory}")

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    if documents and llm:
        index = build_sentence_window_index(documents, llm)
        query_engine = get_sentence_window_query_engine(index)
        # Execute a query and print the result.
        response = query_engine.query("What are the key insights about machine learning methodology?")
        print("\nQuery Response:")
        print(response)
    else:
        print("Please provide valid documents and an LLM instance to build the index.")
