# llamaindex==0.12.16

import json
from llama_index.core.selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
    PydanticMultiSelector,
    PydanticSingleSelector,
)
from llama_index.core.tools import ToolMetadata

# Define a custom JSON encoder to handle PromptTemplate objects
class PromptTemplateEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            # Attempt to serialize using the object's __dict__ or str representation
            return str(obj)
        except Exception:
            return super().default(obj)

# Define a helper function to pretty-print the prompts dictionary.
def display_prompt_dict(prompts: dict):
    print("Selector Prompts:")
    # Use the custom encoder to handle non-serializable objects
    print(json.dumps(prompts, indent=4, cls=PromptTemplateEncoder))

# Example 1: Single Selection
print("=== SINGLE SELECTOR EXAMPLE ===")
single_selector = LLMSingleSelector.from_defaults()
display_prompt_dict(single_selector.get_prompts())

# Define a list of tool metadata objects representing available tools.
tool_choices = [
    ToolMetadata(
        name="query_routing",
        description="This tool contains a paper about routing method"
    ),
    ToolMetadata(
        name="query_rewriting",
        description="This tool contains the Wikipedia page about rewriting method"
    ),
    ToolMetadata(
        name="query_react",
        description="This tool contains the Wikipedia page about react method"
    ),
]

# Use the single selector to choose the most relevant tool for the query.
try:
    single_result = single_selector.select(
        tool_choices,
        query="Tell me more about rewriting methods"
    )
    
    print("\nSingle Selector - Selected Tool:")
    if single_result.selections:
        for selection in single_result.selections:
            selected_tool = tool_choices[selection.index]
            print(f"  Name: {selected_tool.name}")
            print(f"  Description: {selected_tool.description}")
            print(f"  Reason: {selection.reason}")
    else:
        print("  No selection made")
except Exception as e:
    print(f"Error in single selection: {e}")

# Example 2: Multi Selection
print("\n=== MULTI SELECTOR EXAMPLE ===")
multi_selector = LLMMultiSelector.from_defaults()
display_prompt_dict(multi_selector.get_prompts())

try:
    multi_result = multi_selector.select(
        tool_choices,
        query="Compare routing and rewriting methods"
    )
    
    print("\nMulti Selector - Selected Tools:")
    if multi_result.selections:
        for selection in multi_result.selections:
            selected_tool = tool_choices[selection.index]
            print(f"  Name: {selected_tool.name}")
            print(f"  Description: {selected_tool.description}")
            print(f"  Reason: {selection.reason}")
            print("  ---")
    else:
        print("  No selections made")
except Exception as e:
    print(f"Error in multi selection: {e}")

# Example 3: Pydantic Selectors (if function calling is available)
print("\n=== PYDANTIC SELECTOR EXAMPLE ===")
try:
    pydantic_selector = PydanticSingleSelector.from_defaults()
    pydantic_result = pydantic_selector.select(
        tool_choices,
        query="What is the react method?"
    )
    
    print("\nPydantic Selector - Selected Tool:")
    if pydantic_result.selections:
        for selection in pydantic_result.selections:
            selected_tool = tool_choices[selection.index]
            print(f"  Name: {selected_tool.name}")
            print(f"  Description: {selected_tool.description}")
            print(f"  Reason: {selection.reason}")
except Exception as e:
    print(f"Pydantic selector not available or error occurred: {e}")

# Practical usage example - how to route queries based on selection
def route_query(query: str, selector_type: str = "single"):
    """
    Route a query to the appropriate tool based on selector choice
    """
    if selector_type == "single":
        selector = LLMSingleSelector.from_defaults()
    elif selector_type == "multi":
        selector = LLMMultiSelector.from_defaults()
    else:
        selector = LLMSingleSelector.from_defaults()
    
    try:
        result = selector.select(tool_choices, query=query)
        
        if result.selections:
            selected_tools = []
            for selection in result.selections:
                tool = tool_choices[selection.index]
                selected_tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'reason': selection.reason
                })
            return selected_tools
        else:
            return [{'name': 'default', 'description': 'Default fallback tool', 'reason': 'No specific tool matched'}]
    except Exception as e:
        print(f"Routing error: {e}")
        return [{'name': 'error', 'description': 'Error handling tool', 'reason': str(e)}]

# Example usage
print("\n=== PRACTICAL ROUTING EXAMPLE ===")
queries = [
    "Tell me more about rewriting methods",
    "Compare routing and react methods",
    "Unknown query about something else"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    routed_tools = route_query(query, "multi")
    for tool in routed_tools:
        print(f"  -> Routed to: {tool['name']} ({tool['reason']})")