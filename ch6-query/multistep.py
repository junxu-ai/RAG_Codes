from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

def _ensure_openai_embedding_private_attrs(embed_model: OpenAIEmbedding) -> None:
    """Work around a llama-index + pydantic v2 private-attr init issue."""
    for attr_name in ("_text_engine", "_query_engine"):
        try:
            getattr(embed_model, attr_name)
            continue
        except AttributeError:
            pass

        try:
            setattr(embed_model, attr_name, embed_model.model_name)
        except Exception:
            private_dict = getattr(embed_model, "__pydantic_private__", None)
            if isinstance(private_dict, dict):
                private_dict[attr_name] = embed_model.model_name

# Load documents
documents = SimpleDirectoryReader(r".\data\ml").load_data()

# OpenAI embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
_ensure_openai_embedding_private_attrs(embed_model)

# Create index with explicit embedding model
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Initialize LLM with slightly higher temperature for more varied responses
gpt4 = OpenAI(temperature=0.3, model="gpt-4o-mini")

# Set up base query engine
query_engine = index.as_query_engine(llm=gpt4)

from llama_index.core.indices.query.query_transform.prompts import StepDecomposeQueryTransformPrompt
custom_prompt = StepDecomposeQueryTransformPrompt(
    "Original question: {query_str}\n"
    "Context: {context_str}\n"
    "Previous reasoning: {prev_reasoning}\n"
    "Generate a NEW, not-previously-asked sub-question. "
    "If nothing new is needed, return 'None'.\n"
    "New question: "
)
# Enhanced StepDecomposeQueryTransform with better configuration
step_decompose_transform = StepDecomposeQueryTransform(
    llm=gpt4, 
    verbose=True,    
    step_decompose_query_prompt=custom_prompt,
)

# More descriptive index summary to guide decomposition
index_summary = """
This index contains comprehensive information about machine learning methodologies, 
including data preprocessing, model selection, training procedures, evaluation metrics, 
and deployment strategies. It covers both classical ML (scikit-learn) and deep learning 
(PyTorch) approaches, as well as database integration patterns and UI development practices.
"""

# Create multi-step query engine with enhanced configuration
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary,
    # verbose=True,  # Enable verbose output to see the decomposition process
    # Specify number of steps explicitly
    num_steps=3
)

# Execute query
complex_query = "How can we design and implement an end-to-end, production-oriented machine learning system covering all aspects of the machine learning lifecycle?"

print("Original Query:")
print(complex_query)
print("\n" + "="*80 + "\n")

response_gpt4 = query_engine.query(complex_query)

# Display response
print("Final Response:")
print(str(response_gpt4))

# To debug the decomposition process, you can also access intermediate steps:
if hasattr(query_engine, '_intermediate_steps'):
    print("\nIntermediate Steps:")
    for i, step in enumerate(query_engine._intermediate_steps):
        print(f"Step {i+1}: {step}")