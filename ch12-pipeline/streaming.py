# openai streaming
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Initiate the chat completion request with streaming enabled
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me how LLM works."}
    ],
    stream=True
)

# Process the streamed response
for chunk in response:
    if 'choices' in chunk:
        content = chunk['choices'][0].get('delta', {}).get('content', '')
        print(content, end='', flush=True)



# langchain streaming

from langchain.llms import OpenAI

# Initialize the OpenAI LLM with streaming enabled
llm = OpenAI(
    model_name="gpt-4",
    streaming=True,
    openai_api_key='your-api-key'
)

# Define a callback function to process each token as it's received
def process_token(token):
    print(token, end='', flush=True)

# Generate a response and stream the output
response = llm.stream("Tell me a joke.")
for token in response:
    process_token(token)


# llama_index streaming
from llama_index import GPTSimpleVectorIndex

index = GPTSimpleVectorIndex.load_from_disk('index.json')
query_engine = index.as_query_engine(streaming=True)

response = query_engine.query("Tell me how LLM works.")
for token in response.response_gen:
    print(token, end='', flush=True)
