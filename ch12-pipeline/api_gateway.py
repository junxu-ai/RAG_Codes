# to be verified

import asyncio

# Define a generic callback function for successful responses and errors.
async def llm_callback(response, error=None, source=""):
    if error:
        print(f"[{source} Callback] Error occurred: {error}")
    else:
        # For demonstration, we assume response.raw is available for Neutrino.
        # For OpenRouter, adjust the processing as needed.
        if source == "Neutrino":
            print(f"[{source} Callback] Optimal model: {response.raw.get('model', 'unknown')}")
        elif source == "OpenRouter":
            print(f"[{source} Callback] Response: {response}")
        else:
            print(f"[{source} Callback] Response: {response}")

async def demo_neutrino():
    # Neutrino Gateway example (simulate API gateway for LLM with callback)
    from llama_index.llms.neutrino import Neutrino
    # ChatMessage might be used when building structured conversations.
    from llama_index.core.llms import ChatMessage
    
    # Initialize the Neutrino client with a router parameter.
    llm = Neutrino(
        api_key="<your-Neutrino-api-key>",
        router="test"  # "test" router configured in the dashboard.
    )
    
    try:
        # Assuming complete() supports async invocation.
        response = await llm.complete("What is large language model?")
        await llm_callback(response, source="Neutrino")
    except Exception as err:
        await llm_callback(None, error=err, source="Neutrino")

async def demo_openrouter():
    # OpenRouter Gateway example (simulate API gateway callback functionality)
    from llama_index.llms.openrouter import OpenRouter
    from llama_index.core.llms import ChatMessage

    llm = OpenRouter(
        api_key="<your-OpenRouter-api-key>",
        max_tokens=256,
        context_window=4096,
        model="gryphe/mythomax-l2-13b",
    )
    
    # Construct a chat message.
    message = ChatMessage(role="user", content="Tell me a joke")
    
    try:
        # Assuming chat() returns an awaitable response.
        response = await llm.chat([message])
        await llm_callback(response, source="OpenRouter")
    except Exception as err:
        await llm_callback(None, error=err, source="OpenRouter")

async def main():
    print("Starting Neutrino demo with callback function:")
    await demo_neutrino()
    print("\nStarting OpenRouter demo with callback function:")
    await demo_openrouter()

# Run the asynchronous main function.
asyncio.run(main())
