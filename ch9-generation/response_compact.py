from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer

# Users can choose different response modes as needed.
response_synthesizer = get_response_synthesizer(response_mode="compact")

# Define a list of scored nodes.
# For demonstration purposes, we create one NodeWithScore.
nodes = [
    NodeWithScore(node=Node(text="First thing’s first, why are MLOps and AI all anyone is talking about, and is it actually necessary for me to understand? MLOps is summarized as a set of practices that create an assembly line for building and running machine learning models. At its core, it involves automating tasks to foster productive collaboration among data scientists, engineers, and even non-technical stakeholders, ultimately enhancing models and their outputs."), score=1.0),
    NodeWithScore(node=Node(text="Enhancing a machine learning model’s performance can be challenging at times. Despite trying all the strategies and algorithms you’ve learned, you tend to fail at improving the accuracy of your model. You feel helpless and stuck. And this is where 90% of the data scientists give up. The remaining 10% is what differentiates a master data scientist from an average data scientist. This article covers 8 proven ways to re-structure your model approach on how to increase accuracy of machine learning model and improve its accuracy."), score=1.0),
    # You can add additional NodeWithScore instances as needed.
]

# Synthesize the final response based on a query and the list of nodes.
response = response_synthesizer.synthesize("how to build a machine learning model efficiently?", nodes=nodes)

print(response)
