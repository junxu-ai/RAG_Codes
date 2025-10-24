# llamaindex==0.12.16

from llama_index.core import Document, VectorStoreIndex

# Load the file "ml.txt" (ensure this file is in the current working directory)
with open(r".\codes\data\ml\ml.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a Document with metadata; adjust metadata as needed
doc = Document(
    text=text,
    metadata={"title": "Machine Learning Methodology and Related Insights"}
)

# Build an index from the document; you can use any index type supported by LlamaIndex.
# Here, we use VectorStoreIndex as an example.
index = VectorStoreIndex.from_documents([doc])

from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=index.as_query_engine(),
        metadata=ToolMetadata(
            name="ml",
            description="Machine learning methodology",
        ),
    ),
]
# build a sub-question query engine over this tool
# this allows decomposing the question down into sub-questions which then execute against the tool
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
   use_async=True,
)

response = query_engine.query("How to build a machine learning model efficiently? ")
