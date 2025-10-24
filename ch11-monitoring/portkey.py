# pip install -qU portkey-ai

from portkey_ai import Portkey
import os

# Initialize the Portkey client with your Portkey API key and specify the router configuration.
client = Portkey(
    provider="openai",                # Specify the provider (could be 'openai', 'anthropic', etc.).
    Authorization=os.getenv("OPENAI_API_KEY"),  # Use your llm provider API key.
    router="test"                     # This corresponds to a custom router configuration on your Portkey dashboard.
)

# Make a completion call using a unified API.
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "What is a large language model?"}],
    model="gpt-4o-mini"               # Use your preferred model; fallback routing occurs per router configuration.
)

# Access and display the selected (or optimal) model as determined by the gateway routing.
print(f"Answer: {response.choices[0].model_dump()['message']['content']}")
