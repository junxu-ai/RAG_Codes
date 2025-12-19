"""
Chat Engine Implementation with LlamaIndex and LangChain
Unified Interface with Engine Selection
"""

import argparse
import os

def run_llama_index_chat():
    """Run LlamaIndex chat engine"""
    print("LLAMAINDEX CHAT ENGINE")
    print("=" * 30)
    
    try:
        # Import necessary modules from LlamaIndex
        from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        # Load documents from a local folder
        try:
            documents = SimpleDirectoryReader("./data/ml").load_data()
            print(f"Loaded {len(documents)} documents")
        except Exception as e:
            print(f"Could not load documents from './data/ml': {e}")
            # Create sample documents for demonstration
            documents = [
                Document(text="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data. It enables computers to identify patterns and make decisions with minimal human intervention. Common applications include recommendation systems, fraud detection, and predictive analytics."),
                Document(text="Deep learning uses neural networks with multiple layers to model complex patterns in data. These deep neural networks can automatically learn hierarchical representations of data. Deep learning excels in computer vision, natural language processing, and speech recognition tasks."),
                Document(text="Natural language processing enables computers to understand and generate human language. NLP techniques include sentiment analysis, machine translation, and chatbot development. Modern NLP heavily relies on transformer architectures like BERT and GPT.")
            ]
            print("Using sample documents for demonstration")

        # Create a vector index from documents
        embed_model = OpenAIEmbedding()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        print("Vector index created successfully")

        # Convert the index into a chat engine
        chat_engine = index.as_chat_engine()
        print("Chat engine created successfully")

        # Engage in a conversation – ask a question
        response = chat_engine.chat("Tell me about machine learning.")
        print(f"Question: Tell me about machine learning.")
        print(f"Response: {response}")
        
        # Interactive mode
        print("\nEntering interactive mode (type 'quit' to exit):")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            response = chat_engine.chat(user_input)
            print(f"AI: {response}")
            
    except ImportError as e:
        print(f"LlamaIndex not installed or import error: {e}")
        print("Install with: pip install llama-index llama-index-llms-openai llama-index-embeddings-openai")
    except Exception as e:
        print(f"LlamaIndex error: {e}")

def run_langchain_chat():
    """Run LangChain chat engine"""
    print("LANGCHAIN CHAT ENGINE")
    print("=" * 30)
    
    try:
        # Import necessary modules from LangChain
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferMemory

        # Initialize an LLM (ensure your OPENAI_API_KEY is set in the environment)
        try:
            llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo"
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to a mock LLM for demonstration
            from langchain.llms import FakeListLLM
            llm = FakeListLLM(responses=[
                "Machine learning is a fascinating field of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every task.",
                "Why don't scientists trust atoms? Because they make up everything!",
                "Deep learning is a subset of machine learning that uses neural networks with multiple layers to automatically learn hierarchical representations of data."
            ])
            print("Using mock LLM for demonstration")

        # Create a conversation chain that retains context
        memory = ConversationBufferMemory()
        chat = ConversationChain(llm=llm, memory=memory)
        print("Conversation chain created with memory")

        # Start a conversation by sending a message
        response = chat.predict("Tell me about machine learning.")
        print(f"Question: Tell me about machine learning.")
        print(f"Response: {response}")
        
        # Interactive mode
        print("\nEntering interactive mode (type 'quit' to exit):")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            response = chat.predict(user_input)
            print(f"AI: {response}")
            
    except ImportError as e:
        print(f"LangChain not installed or import error: {e}")
        print("Install with: pip install langchain langchain-openai")
    except Exception as e:
        print(f"LangChain error: {e}")

def main(engine_choice):
    """Main function to run the selected chat engine"""
    if engine_choice == "llamaindex":
        run_llama_index_chat()
    elif engine_choice == "langchain":
        run_langchain_chat()
    elif engine_choice == "both":
        print("Running both engines sequentially:\n")
        run_llama_index_chat()
        print("\n" + "="*50 + "\n")
        run_langchain_chat()
    else:
        print(f"Invalid engine choice: {engine_choice}")
        print("Valid options: 'llamaindex', 'langchain', 'both'")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Chat Engine Implementation")
    parser.add_argument(
        "--engine", 
        choices=["llamaindex", "langchain", "both"],
        default="langchain",
        help="Specify which chat engine to use (default: llamaindex)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected engine
    main(args.engine)