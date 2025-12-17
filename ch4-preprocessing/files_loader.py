# Loading simple files using LangChain
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/wukong_story.txt")
documents = loader.load()

print(documents[0].metadata)     # e.g. {'source': 'data/wukong_story.txt'}
print(documents[0].page_content) # The entire text content of the file


# Loading from Directories

from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load only Markdown files under "data/docs" using TextLoader
loader = DirectoryLoader(
    path="data/docs",
    glob="**/*.md",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Total documents loaded: {len(documents)}")

# Loading PDFs and Other Formats
from langchain.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("data/handbook.pdf")
pdf_documents = pdf_loader.load()
print(f"Loaded {len(pdf_documents)} PDF pages.")

# Data Loading with LlamaIndex
from llama_index import SimpleDirectoryReader

reader = SimpleDirectoryReader("data/ ")
documents = reader.load_data()

print(f"Loaded {len(documents)} documents")
print(documents[0].text[:200])  # Print snippet of the first document's text


# 1.3   Loading Data with Data Processing Tools
from unstructured.partition.auto import partition

## Load and partition a document
elements = partition(filename="data/report.pdf")

## Access different types of elements
for element in elements:
    # Check element type (Title, NarrativeText, Table, etc.)
    print(f"Type: {type(element)}")
    print(f"Text: {element.text}\n")
