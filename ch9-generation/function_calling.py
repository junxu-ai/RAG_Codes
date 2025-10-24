import openai
import json
import os

# Initialize the OpenAI API client
openai.api_key = os.environ["OPENAI_API_KEY"]  

# Define the function schema
functions = [
    {
        "name": "create_user_profile",
        "description": "Create a user profile with name, age, and occupation.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The user's name"},
                "age": {"type": "integer", "description": "The user's age"},
                "occupation": {"type": "string", "description": "The user's occupation"}
            },
            "required": ["name", "age", "occupation"]
        }
    }
]

# Define the user query
query = "I met John, a 28-year-old doctor."

# Generate a response from the LLM
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": query}],
    functions=functions,
    function_call={"name": "create_user_profile"}
)

print(response)
# Extract the function call from the response
function_call = response.to_dict()['choices'][0]['message']['function_call']

# Parse the arguments
arguments = json.loads(function_call['arguments'])

# Output the structured data
print(arguments)
