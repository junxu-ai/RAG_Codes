# pip install faiss-cpu llama-index-vector-stores-faiss
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import SentenceSplitter

# FAISS imports
try:
    import faiss
    from llama_index.vector_stores.faiss import FaissVectorStore
except ImportError as e:
    raise SystemExit(
        "FAISS not available. Install with:\n"
        "  pip install faiss-cpu llama-index-vector-stores-faiss\n"
        f"ImportError: {e}"
    )

# Initialize LLM and embedding model
# Specify dimensions to match the embedding model (1536 for text-embedding-3-small)
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1536)
llm = OpenAI(model="gpt-3.5-turbo")

# Load documents from a directory or create samples
try:
    documents = SimpleDirectoryReader("./data/ml/").load_data()
except Exception as e:
    print(f"Could not load documents: {e}")
    documents = [
        Document(
            text=(
                "Machine learning is a subset of artificial intelligence that focuses "
                "on algorithms that learn from data. It enables computers to identify "
                "patterns and make decisions with minimal human intervention. "
                "Common applications include recommendation systems, fraud detection, "
                "and predictive analytics."
            )
        ),
        Document(
            text=(
                "Deep learning uses neural networks with multiple layers to model "
                "complex patterns in data. These deep neural networks can "
                "automatically learn hierarchical representations of data. "
                "Deep learning excels in computer vision, natural language processing, "
                "and speech recognition tasks."
            )
        ),
        Document(
            text=(
                "Natural language processing enables computers to understand and "
                "generate human language. NLP techniques include sentiment analysis, "
                "machine translation, and chatbot development. Modern NLP heavily "
                "relies on transformer architectures like BERT and GPT."
            )
        ),
    ]

# Split documents into chunks
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = text_splitter.get_nodes_from_documents(documents)

# Create FAISS-backed vector store and storage context
embed_dim = getattr(embed_model, "dimensions", 1536) or 1536
faiss_index = faiss.IndexFlatL2(embed_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create VectorStoreIndex using FAISS
index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)

# Define test questions
questions = [
    "What is machine learning?",
    "How does deep learning work?",
    "Explain NLP applications",
]

print("HYBRID RETRIEVAL MODES COMPARISON (FAISS BACKEND)")
print("=" * 60)

# Show available modes
print("Available VectorStoreQueryMode options:")
available_modes = [mode for mode in dir(VectorStoreQueryMode) if not mode.startswith("_")]
for mode in available_modes:
    print(f"  - {mode}")

# Modes to test
modes_to_test = {
    "DEFAULT": VectorStoreQueryMode.DEFAULT,
    "HYBRID": VectorStoreQueryMode.HYBRID,
    "SEMANTIC_HYBRID": VectorStoreQueryMode.SEMANTIC_HYBRID,
#     "TEXT_SEARCH": VectorStoreQueryMode.TEXT_SEARCH, # supported in elasticsearch
}

for mode_name, mode_value in modes_to_test.items():
    print(f"\n{mode_name} SEARCH")
    print("-" * 30)

    try:
        retriever = index.as_retriever(
            vector_store_query_mode=mode_value,
            similarity_top_k=3,
        )

        for question in questions[:1]:  # test with first question
            print(f"Question: {question}")
            response = retriever.retrieve(question)
            print(f"Retrieved {len(response)} chunks:")
            for i, node in enumerate(response, 1):
                score = getattr(node, "score", "N/A")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                print(f"  {i}. Score: {score_str}")
                print(f"     Text: {str(node.text)[:150]}...")

    except NotImplementedError as e:
        print(f"Mode {mode_name} not implemented by FAISS store: {e}")
    except Exception as e:
        print(f"Error with {mode_name}: {e}")

# Detailed comparison function
def compare_all_modes(index, query):
    print(f"\nDETAILED COMPARISON FOR QUERY: '{query}'")
    print("=" * 70)

    results_summary = {}

    for mode_name in ["DEFAULT", "HYBRID", "SEMANTIC_HYBRID", "TEXT_SEARCH"]:
        if hasattr(VectorStoreQueryMode, mode_name):
            mode_value = getattr(VectorStoreQueryMode, mode_name)
            try:
                retriever = index.as_retriever(
                    vector_store_query_mode=mode_value,
                    similarity_top_k=3,
                )
                response = retriever.retrieve(query)
                results_summary[mode_name] = len(response)

                print(f"\n{mode_name} MODE:")
                print(f"  Retrieved {len(response)} chunks")
                for i, node in enumerate(response, 1):
                    score = getattr(node, "score", "N/A")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    print(f"  {i}. Score: {score_str}")
                    print(f"     Text: {str(node.text)[:120]}...")

            except NotImplementedError as e:
                print(f"\n{mode_name} MODE: Not implemented by FAISS store - {e}")
                results_summary[mode_name] = "Not implemented"
            except Exception as e:
                print(f"\n{mode_name} MODE: Error - {e}")
                results_summary[mode_name] = "Error"
        else:
            print(f"\n{mode_name} MODE: Not available")
            results_summary[mode_name] = "Not available"

    print("\nSUMMARY:")
    for mode, count in results_summary.items():
        print(f"  {mode}: {count} results")

# Run detailed comparison
if __name__ == "__main__":
    test_query = "Compare machine learning and deep learning approaches"
    compare_all_modes(index, test_query)
