# !pip install langchain
# !pip install -U langchain-community
# !pip install sentence-transformers
# !pip install faiss-gpu
# !pip install pypdf
# !pip install faiss-cpu


# !pip install  langchain-openai
# !pip install  langchain-experimental
# !pip install json-repair
# !pip install neo4j

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# create a free instant from https://neo4j.com/free/
# Establishes a connection to a Neo4j database instance.
# The URL indicates a secure Neo4j connection (neo4j+s://).
# Used to store and query structured knowledge graphs.


graph = Neo4jGraph(
    url= "neo4j+s://d2ebd34e.databases.neo4j.io" ,
    username="neo4j", #default
    password="xFqnLkaXUkmI8jnKZ3qPppFT-CWnhcqONGf6Pqktwbg" #change accordingly
)

text = """
Harry Potter is an orphaned wizard raised by Muggles who discovers, at age eleven, that he is famous in the magical world for surviving Lord Voldemort’s attack that killed his parents, James and Lily Potter. At Hogwarts School of Witchcraft and Wizardry he forms an inseparable trio with Ron Weasley—sixth son in a warm-hearted wizarding family—and Hermione Granger, a brilliant Muggle-born witch. Under headmaster Albus Dumbledore’s mentorship, the trio repeatedly uncovers fragments of Voldemort’s plan to regain power.

Voldemort, born Tom Riddle, is linked to Harry by prophecy and by the piece of his own soul embedded in Harry’s scar. His chief followers, the Death Eaters, include the fanatical Bellatrix Lestrange and the conflicted Severus Snape. Snape, outwardly Voldemort’s ally, is secretly loyal to Dumbledore because of his unrequited love for Lily Potter; his double role shapes the war’s outcome.

Harry’s godfather Sirius Black and former teacher Remus Lupin, both members of the Order of the Phoenix, become surrogate family figures, while gamekeeper Rubeus Hagrid acts as guardian and friend from Harry’s first day. Ron’s sister Ginny evolves from schoolgirl admirer to Harry’s partner, tying Harry permanently to the Weasley clan. Rivalry with Draco Malfoy, heir to a pure-blood supremacist line, mirrors the wider ideological divide in wizarding society.

The story culminates in a hunt for Voldemort’s Horcruxes—objects anchoring his fragmented soul—leading to the Battle of Hogwarts. When the final Horcrux inside himself is destroyed, Harry willingly sacrifices, is briefly “killed,” and returns to defeat Voldemort, freeing the wizarding world. Nineteen years later, the next generation boards the Hogwarts Express, symbolizing hard-won peace and the enduring power of chosen bonds over bloodline dogma.
"""
documents = [Document(page_content=text)]

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))

# Uses LLMGraphTransformer to convert the input text into structured graph documents (nodes and relationships).
llm_transformer_filtered = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer_filtered.convert_to_graph_documents(documents)

# Stores the extracted graph structure into the Neo4j database.
# Each entity and relationship from the text is stored as a node or edge in the graph.

graph.add_graph_documents(
      graph_documents,
      baseEntityLabel=True,
      include_source=True
  )


# Creates a hybrid vector index on top of Neo4j.
# This enables semantic search over both structured graph data and unstructured text content.

embed = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.getenv("OPENAI_API_KEY"))
vector_index = Neo4jVector.from_existing_graph(
    embedding=embed,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url="neo4j+s://d2ebd34e.databases.neo4j.io",
    username="neo4j", #default
    password="xFqnLkaXUkmI8jnKZ3qPppFT-CWnhcqONGf6Pqktwbg" #change accordingly
)
vector_retriever = vector_index.as_retriever()


## start querying
class Entities(BaseModel):
    names: list[str] = Field(..., description="All entities from the text")

## Define the prompt to extract entities
prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract organization and people entities from the text."),
        ("human", "Extract entities from: {question}")
    ])

# Define the LLM chain to extract entities
entity_chain = prompt | llm.with_structured_output(Entities, include_raw=True)
response = entity_chain.invoke({"question": "What's the relation between Harry Potter and Voldemort?"})
entities =  response['raw'].content

import json
entities = json.loads(entities)['names']

# Define the prompt to query the graph


vector_data = [el.page_content for el in vector_retriever.invoke( "Who are Harry Potter and Voldemort?")]
print(vector_data)

context= f"Graph data: {graph_data}\nVector data: {'#Document '.join(vector_data)}"

template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": lambda input: context,  # Generate context from the question
            "question": RunnablePassthrough(),  # Pass the question through without modification
        }
        | prompt  # Apply the prompt template
        | llm  # Use the language model to answer the question based on context
        | StrOutputParser()  # Parse the model's response as a string
    )

response = chain.invoke({"question": "Who are Harry Potter and Voldemort?"}) 
graph_data  = ""
for entity in entities:
    query_response = graph.query(
        """MATCH (p:Person {id: $entity})-[r]->(e)
        RETURN p.id AS source_id, type(r) AS relationship, e.id AS target_id
        LIMIT 50""",
        {"entity": entity}
    )
    graph_data  += "\n".join([f"{el['source_id']} - {el['relationship']} -> {el['target_id']}" for el in query_response])
print(graph_data )


# use both graph and vector data
context=f"Graph data:{graph_data}\nVector data:{'#Document '.join(vector_data)}"
template="""Answer the question based only on the following context:
{context}
Question: {question}
Answer:"""

prompt= ChatPromptTemplate.from_template(template)


chain = (
        {
            "context":lambdainput: context, # Generate context from the question
            "question": RunnablePassthrough(), # Pass the question through without modification
        }
        | prompt  # Apply the prompt template
        | llm  # Use the language model to answer the question based on context
        | StrOutputParser()  # Parse the model's response as a string
    )


response = chain.invoke({"question": "Who are Harry Potter and Voldemort?"}) 