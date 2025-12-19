import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
client = openai.OpenAI(api_key=MY_API_KEY)
llm = OpenAI(client)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

documents = [
"Queries such as identifying the risk factors of the highest-performing ridesharing company in Singapore integrate data from different sources to provide comprehensive insights
"Tell me about the long-term and short-term sustainable strategy of SCB and HSCB, where long-term is the plan more than 5 years, and short-term is less than 5 years. Make a table to list the information and then make a conclusion!"
]

# Extract keywords
keywords = kw_model.extract_keywords(documents)

