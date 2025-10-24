from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize

# Define the context text about Bitcoin 
text = """ Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user on the peer-to-peer bitcoin network without the need for intermediaries. Transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain. Bitcoin was invented in 2008 by an unknown person or group of people using the name Satoshi Nakamoto. The currency began use in 2009 when its implementation was released as open-source software. """

# NOTE: we add an extra tone_name variable here
qa_prompt_tmpl = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query in the tone of {tone_name}.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt = PromptTemplate(qa_prompt_tmpl)

# initialize response synthesizer
summarizer = TreeSummarize(verbose=True, summary_template=qa_prompt)

# Define the query and desired tone 
query = "What is Bitcoin?" 
tone = "Donald Trump" 

# Generate the response 
response = summarizer.get_response(query, [text], tone_name=tone) 
# Output the response 
print(response)
