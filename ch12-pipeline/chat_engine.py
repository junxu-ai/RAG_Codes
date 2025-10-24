# to be verified

"""
LlamaIndex Implementation
"""

# Import necessary modules from LlamaIndex
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# Load documents from a local folder (assume 'data' contains your documents)
documents = SimpleDirectoryReader("data").load_data()

# Create a vector index from documents
index = VectorStoreIndex.from_documents(documents)

# Convert the index into a chat engine
chat_engine = index.as_chat_engine()

# Engage in a conversation – ask a question
response = chat_engine.chat("Tell me a joke.")
print(response)


"""
 LangChain Implementation
"""
# Import necessary modules from LangChain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Initialize an LLM (ensure your OPENAI_API_KEY is set in the environment)
llm = OpenAI(temperature=0.7)

# Create a conversation chain that retains context
chat = ConversationChain(llm=llm)

# Start a conversation by sending a message
response = chat.run("Tell me a joke.")
print(response)
