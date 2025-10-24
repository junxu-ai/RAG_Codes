"""setup"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# 1a. Instantiate callback manager for event handling
llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug])

# 1b. Define underlying functions
def get_weather(location: str) -> str:
    return "Sunny, 25°C"  # placeholder implementation

def calculate_sum(a: int, b: int) -> int:
    return a + b

# 1c. Build the router (as shown above)
FUNCTION_REGISTRY = {
    "get_weather": get_weather,
    "calculate_sum": calculate_sum,
}
tool_names = list(FUNCTION_REGISTRY.keys())

"""Router Definition"""

async def user_selectable_router(
    function_name: str,
    function_args: dict,
    callback_manager: CallbackManager,
    tool_names: list[str]
) -> str:
    # Ask user which function to run
    payload = {
        "available_tools": tool_names,
        "proposed_function": function_name,
        "proposed_args": function_args,
    }
    await callback_manager.emit(InputRequiredEvent(payload=payload))
    # Wait for user
    human_event = await callback_manager.wait_for(HumanResponseEvent)
    choice = human_event.payload["chosen_function"]
    args = human_event.payload["chosen_args"]
    # Dispatch real function
    result = FUNCTION_REGISTRY[choice](**args)
    return str(result)

"""Register Tool & Instantiate Agent"""
from llama_index.core.tools import FunctionTool, ToolMetadata

# Create metadata for each underlying function so LLM “knows” signatures:
all_functions_metadata = [
    ToolMetadata(
        name="get_weather",
        description="Returns current weather for a location.",
        # Provide JSON Schema for parameters
        fn_schema={"location": {"type": "string", "description": "City name."}},
    ),
    ToolMetadata(
        name="calculate_sum",
        description="Returns the sum of two numbers.",
        fn_schema={
            "a": {"type": "integer", "description": "First integer."},
            "b": {"type": "integer", "description": "Second integer."},
        },
    ),
]

# Wrap the router as a single FunctionTool that advertises all signatures:
router_tool = FunctionTool.from_defaults(
    fn=user_selectable_router,
    tool_metadata=ToolMetadata(
        name="router",  # Agent will see “router”, but the LLM sees get_weather and calculate_sum from metadata
        description="Routes function calls through user confirmation.",
        fn_schema=all_functions_metadata
    )
)

agent = FunctionAgent(
    tools=[router_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You can call get_weather(location) or calculate_sum(a,b)."
                  "Ask the user before invoking the function.",
    callback_manager=callback_manager,
)

from pydantic import BaseModel
class WeatherInput(BaseModel):
    location: str

class SumInput(BaseModel):
    a: int
    b: int

weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="Returns current weather for a location.",
    fn_schema=WeatherInput
)

sum_tool = FunctionTool.from_defaults(
    fn=calculate_sum,
    name="calculate_sum",
    description="Returns the sum of two numbers.",
    fn_schema=SumInput
)
agent = FunctionAgent(
    tools=[weather_tool, sum_tool],  # Not router_tool yet unless adapted
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You can call get_weather(location) or calculate_sum(a,b). Ask the user before invoking the function.",
    callback_manager=callback_manager,
)


"""Run Agent with Event Loop"""
import asyncio, json
from llama_index.core.workflow import (
    InputRequiredEvent,
    HumanResponseEvent,
)
# async def main():
#     # handler = agent.run(user_msg="what is the weather in Paris?")
#     handler = agent.run(user_msg="What is 5 + 7?")

#     async for event in handler.stream_events():
#         if isinstance(event, InputRequiredEvent):
#             # capture keyboard input
#             response = input(event.prefix)
#             # send our response back
#             handler.ctx.send_event(
#                 HumanResponseEvent(
#                     response=response,
#                     user_name=event.user_name,
#                 )
#             )

#     response = await handler
#     print(str(response))

async def main():
    # Start the agent loop (this spawns LLM calls + Workflow)
    run_task = asyncio.create_task(agent.run("What is 5 + 7?"))
    
    # Simultaneously process events:
    while True:
        event = await agent.next_event()  
        if isinstance(event, InputRequiredEvent):
            # Show proposed call & available options
            proposed = event.payload["proposed_function"]
            args = event.payload["proposed_args"]
            options = event.payload["available_tools"]
            print(f"Model wants to call `{proposed}` with {args}.")
            print("Available tools:", options)
            # Prompt user
            choice = input(f"Confirm `{proposed}`? (y/n) ")
            if choice.lower().strip() == "y":
                chosen_fn = proposed
                chosen_args = args
            else:
                chosen_fn = input("Enter function name to call: ")
                chosen_args = json.loads(input(f"Provide JSON args for `{chosen_fn}`: "))
            # Emit response back to agent
            await agent.callback_manager.emit(
                HumanResponseEvent(
                    payload={
                        "chosen_function": chosen_fn,
                        "chosen_args": chosen_args,
                    }
                )
            )
        elif event.event_type == "final_result":
            print("Agent's final answer:\n", event.payload)
            break
        else:
            # Optionally log other events (progress, debug, etc.)
            pass

    await run_task

if __name__ == "__main__":
    asyncio.run(main())
