# API Gateway Implementation for Multiple LLM Providers
import asyncio
import os
from typing import Optional, Any

# Define a generic callback function for successful responses and errors.
async def llm_callback(response: Any = None, error: Optional[Exception] = None, source: str = ""):
    """
    Generic callback function to handle LLM responses and errors
    """
    if error:
        print(f"[{source} Callback] Error occurred: {error}")
        print(f"[{source} Callback] Error type: {type(error).__name__}")
    else:
        try:
            # Handle different response types
            if source == "Neutrino" and hasattr(response, 'raw'):
                model_name = response.raw.get('model', 'unknown') if response.raw else 'unknown'
                print(f"[{source} Callback] Optimal model: {model_name}")
                if hasattr(response, 'text'):
                    print(f"[{source} Callback] Response: {response.text[:200]}...")
            elif source == "OpenRouter":
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    print(f"[{source} Callback] Response: {response.message.content[:200]}...")
                elif isinstance(response, str):
                    print(f"[{source} Callback] Response: {response[:200]}...")
                else:
                    print(f"[{source} Callback] Response: {str(response)[:200]}...")
            else:
                # Generic response handling
                if hasattr(response, 'text'):
                    print(f"[{source} Callback] Response: {response.text[:200]}...")
                else:
                    print(f"[{source} Callback] Response: {str(response)[:200]}...")
        except Exception as e:
            print(f"[{source} Callback] Error processing response: {e}")

async def demo_neutrino():
    """
    Neutrino Gateway example (simulate API gateway for LLM with callback)
    """
    try:
        # Try to import Neutrino - it may not be available
        from llama_index.llms.neutrino import Neutrino
        from llama_index.core.llms import ChatMessage
    except ImportError as e:
        print("[Neutrino] Module not available or installed")
        await llm_callback(None, error=e, source="Neutrino")
        return
    
    # Get API key from environment variable or use placeholder
    api_key = os.getenv("NEUTRINO_API_KEY", "<your-Neutrino-api-key>")
    
    if api_key == "<your-Neutrino-api-key>":
        print("[Neutrino] API key not configured. Please set NEUTRINO_API_KEY environment variable.")
        return
    
    try:
        # Initialize the Neutrino client with a router parameter.
        llm = Neutrino(
            api_key=api_key,
            router="test"  # "test" router configured in the dashboard.
        )
        
        print("[Neutrino] Sending request...")
        # Assuming complete() supports async invocation.
        response = await llm.acomplete("What is large language model?")
        await llm_callback(response, source="Neutrino")
        
    except Exception as err:
        print(f"[Neutrino] Request failed")
        await llm_callback(None, error=err, source="Neutrino")

async def demo_openrouter():
    """
    OpenRouter Gateway example (simulate API gateway callback functionality)
    """
    try:
        # Try to import OpenRouter - it may not be available
        from llama_index.llms.openrouter import OpenRouter
        from llama_index.core.llms import ChatMessage
    except ImportError as e:
        print("[OpenRouter] Module not available or installed")
        await llm_callback(None, error=e, source="OpenRouter")
        return
    
    # Get API key from environment variable or use placeholder
    api_key = os.getenv("OPENROUTER_API_KEY", "<your-OpenRouter-api-key>")
    
    if api_key == "<your-OpenRouter-api-key>":
        print("[OpenRouter] API key not configured. Please set OPENROUTER_API_KEY environment variable.")
        return
    
    try:
        llm = OpenRouter(
            api_key=api_key,
            max_tokens=256,
            context_window=4096,
            model="gryphe/mythomax-l2-13b",
        )
        
        # Construct a chat message.
        message = ChatMessage(role="user", content="Tell me a joke in one sentence")
        
        print("[OpenRouter] Sending request...")
        # Assuming chat() returns an awaitable response.
        response = await llm.achat([message])
        await llm_callback(response, source="OpenRouter")
        
    except Exception as err:
        print(f"[OpenRouter] Request failed")
        await llm_callback(None, error=err, source="OpenRouter")

async def demo_openai():
    """
    OpenAI Gateway example as fallback/demo
    """
    try:
        from llama_index.llms.openai import OpenAI
        from llama_index.core.llms import ChatMessage
    except ImportError as e:
        print("[OpenAI] Module not available or installed")
        await llm_callback(None, error=e, source="OpenAI")
        return
    
    api_key = os.getenv("OPENAI_API_KEY", "<your-OpenAI-api-key>")
    
    if api_key == "<your-OpenAI-api-key>":
        print("[OpenAI] API key not configured. Please set OPENAI_API_KEY environment variable.")
        return
    
    try:
        llm = OpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            max_tokens=256,
        )
        
        message = ChatMessage(role="user", content="Explain API gateways in one sentence")
        
        print("[OpenAI] Sending request...")
        response = await llm.achat([message])
        await llm_callback(response, source="OpenAI")
        
    except Exception as err:
        print(f"[OpenAI] Request failed")
        await llm_callback(None, error=err, source="OpenAI")

async def main():
    """
    Main function to run all demos
    """
    print("API GATEWAY DEMO WITH CALLBACK FUNCTIONS")
    print("=" * 50)
    
    print("\n1. Starting Neutrino demo with callback function:")
    await demo_neutrino()
    
    print("\n2. Starting OpenRouter demo with callback function:")
    await demo_openrouter()
    
    print("\n3. Starting OpenAI demo with callback function:")
    await demo_openai()

# Alternative sequential execution for better error handling
async def main_sequential():
    """
    Alternative main function that runs demos sequentially with better error isolation
    """
    print("SEQUENTIAL API GATEWAY DEMO")
    print("=" * 40)
    
    demos = [
        ("Neutrino", demo_neutrino),
        ("OpenRouter", demo_openrouter),
        ("OpenAI", demo_openai)
    ]
    
    for name, demo_func in demos:
        print(f"\n--- {name} Demo ---")
        try:
            await demo_func()
        except Exception as e:
            print(f"[{name}] Demo failed with error: {e}")
        await asyncio.sleep(1)  # Small delay between requests

# Environment setup instructions
def print_setup_instructions():
    """
    Print instructions for setting up API keys
    """
    print("\nSETUP INSTRUCTIONS:")
    print("=" * 30)
    print("Set the following environment variables:")
    print("  export NEUTRINO_API_KEY='your-neutrino-api-key'")
    print("  export OPENROUTER_API_KEY='your-openrouter-api-key'")
    print("  export OPENAI_API_KEY='your-openai-api-key'")
    print("\nOr create a .env file with these variables")

if __name__ == "__main__":
    # Print setup instructions
    print_setup_instructions()
    
    # Run the asynchronous main function.
    try:
        print("\n" + "="*60)
        print("RUNNING API GATEWAY DEMOS")
        print("="*60)
        asyncio.run(main_sequential())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nMain execution error: {e}")