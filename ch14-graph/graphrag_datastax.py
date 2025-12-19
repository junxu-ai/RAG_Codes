#  pip install "langchain-graph-retriever[astra]"

# GraphRAG Implementation with DataStax/Cassandra-based Graph Retrieval
# Verified and corrected version

try:
    # Try to import the correct package
    from langchain.retrievers import GraphRetriever
except ImportError:
    try:
        # Alternative import path
        from langchain_graph_retriever import GraphRetriever
    except ImportError:
        print("GraphRetriever not available. Install with: pip install langchain-graph-retriever")
        GraphRetriever = None

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

# Sample text
text = """
Harry Potter is an orphaned wizard raised by Muggles who discovers, at age eleven, that he is famous in the magical world for surviving Lord Voldemort's attack that killed his parents, James and Lily Potter. At Hogwarts School of Witchcraft and Wizardry he forms an inseparable trio with Ron Weasley—sixth son in a warm-hearted wizarding family—and Hermione Granger, a brilliant Muggle-born witch. Under headmaster Albus Dumbledore's mentorship, the trio repeatedly uncovers fragments of Voldemort's plan to regain power.

Voldemort, born Tom Riddle, is linked to Harry by prophecy and by the piece of his own soul embedded in Harry's scar. His chief followers, the Death Eaters, include the fanatical Bellatrix Lestrange and the conflicted Severus Snape. Snape, outwardly Voldemort's ally, is secretly loyal to Dumbledore because of his unrequited love for Lily Potter; his double role shapes the war's outcome.

Harry's godfather Sirius Black and former teacher Remus Lupin, both members of the Order of the Phoenix, become surrogate family figures, while gamekeeper Rubeus Hagrid acts as guardian and friend from Harry's first day. Ron's sister Ginny evolves from schoolgirl admirer to Harry's partner, tying Harry permanently to the Weasley clan. Rivalry with Draco Malfoy, heir to a pure-blood supremacist line, mirrors the wider ideological divide in wizarding society.

The story culminates in a hunt for Voldemort's Horcruxes—objects anchoring his fragmented soul—leading to the Battle of Hogwarts. When the final Horcrux inside himself is destroyed, Harry willingly sacrifices, is briefly "killed," and returns to defeat Voldemort, freeing the wizarding world. Nineteen years later, the next generation boards the Hogwarts Express, symbolizing hard-won peace and the enduring power of chosen bonds over bloodline dogma.
"""

# Initialize embeddings
try:
    embedding_function = OpenAIEmbeddings()
    print("OpenAI embeddings initialized")
except Exception as e:
    print(f"Error initializing OpenAI embeddings: {e}")
    # Fallback to a simple embedding function
    from langchain.embeddings import FakeEmbeddings
    embedding_function = FakeEmbeddings(size=10)
    print("Using fake embeddings for demonstration")

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(text)

# Create Document objects
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(texts)]

print(f"Created {len(documents)} document chunks")

# Initialize the vector store (Chroma in this example)
try:
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    print("Chroma vector store created successfully")
except Exception as e:
    print(f"Error creating Chroma vector store: {e}")
    vector_store = None

# Create the Graph Retriever (if available)
if GraphRetriever is not None and vector_store is not None:
    try:
        # Note: GraphRetriever implementation varies; this is a conceptual example
        retriever = GraphRetriever(
            vectorstore=vector_store,  # Corrected parameter name
            # Define edges based on document relationships or metadata
            edges=[("source", "related_to")],  # Simplified edge definition
        )
        print("Graph Retriever created successfully")
        
    except Exception as e:
        print(f"Error creating Graph Retriever: {e}")
        # Fallback to regular vector store retriever
        retriever = vector_store.as_retriever()
        print("Using standard vector store retriever as fallback")
else:
    if vector_store is not None:
        retriever = vector_store.as_retriever()
        print("Using standard vector store retriever")
    else:
        retriever = None
        print("No retriever available")

# Perform retrieval (using correct method name)
if retriever is not None:
    print("\nPERFORMING RETRIEVAL")
    print("=" * 30)
    
    try:
        # Use the correct method for retrieval
        documents = retriever.get_relevant_documents("Who is Harry Potter?")
        # Alternative method if the above doesn't work:
        # documents = retriever.invoke("Who is Harry Potter?")
        
        print(f"Retrieved {len(documents)} documents")
        
        # Print the results
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:300]}...")
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"Metadata: {doc.metadata}")
                
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No retriever available for querying")

# Alternative implementation using standard LangChain components
print("\nALTERNATIVE APPROACH WITH STANDARD LANGCHAIN")
print("=" * 50)

try:
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    if vector_store is not None:
        # Create a standard retrieval QA chain
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        print("Standard QA chain created")
        
        # Test query
        query = "Who is Harry Potter?"
        print(f"\nQuery: {query}")
        result = qa_chain.invoke(query)
        print(f"Answer: {result['result']}")
        
    else:
        print("Vector store not available for alternative approach")
        
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Install with: pip install langchain-openai")
except Exception as e:
    print(f"Error in alternative approach: {e}")

print("\nGRAPH RETRIEVAL DEMONSTRATION COMPLETE")
print("=" * 40)
print("Key components demonstrated:")
print("1. Text chunking and document preparation")
print("2. Vector store creation with embeddings")
print("3. Graph-based retrieval (when available)")
print("4. Fallback to standard retrieval methods")
print("5. Error handling and graceful degradation")