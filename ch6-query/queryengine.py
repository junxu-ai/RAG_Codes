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
print(response)


## a sample output:
# [ml] Q: What are the best practices for data preprocessing in machine learning?
# [ml] Q: How to select the most suitable machine learning algorithm for a specific task?
# [ml] Q: What techniques can be used for feature selection in machine learning models?
# [ml] A: Feature engineering plays a critical role in extracting relevant characteristics from raw data in machine learning models.
# [ml] A: High-quality data collection, cleaning, handling missing values, and normalizing features are essential steps in data preprocessing for machine learning. Feature engineering is crucial for extracting relevant characteristics from raw data, improving model performance, and reducing the risk of bias and overfitting.
# [ml] A: Choosing the right algorithm for a specific task in machine learning involves considering the nature of the task itself, whether it is classification, regression, or clustering. Depending on the task requirements, practitioners typically select from a range of algorithms such as linear regression, decision trees, support vector machines, neural networks, and ensemble methods. The selection process is crucial to ensure that the chosen algorithm is well-suited to the data and can effectively address the objectives of the task at hand.
