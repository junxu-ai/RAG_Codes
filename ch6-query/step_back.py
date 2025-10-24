from llama_index.core.indices import  SimpleKeywordTableIndex
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.keyword_table import KeywordTableGPTRetriever 
from llama_index.core import PromptTemplate

# Simulated documents for testing
documents = [
    Document(text="LangChain is a framework for building applications with LLMs, including prompt management and chains."),
    Document(text="LlamaIndex provides tools for data retrieval and processing in LLM workflows."),
    Document(text="Both LangChain and LlamaIndex help build AI applications but focus on different aspects.")
]

llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# Create index and retriever
index = SimpleKeywordTableIndex.from_documents(documents)
retriever = KeywordTableGPTRetriever(index=index)

# Define few-shot examples
examples = [
    {
        "input": "What are the key features of LangChain for building AI applications?",
        "output": "What is LangChain used for?"
    },
    {
        "input": "How does LlamaIndex simplify data retrieval for AI models?",
        "output": "What is the purpose of LlamaIndex?"
    }
]

# Prepare few-shot examples by joining them into a single string.
few_shot_examples = "\n\n".join(
    [f"Human: {example['input']}\nAI: {example['output']}" for example in examples]
)

# Define a prompt template for rephrasing the input question into a more general form.
step_back_prompt = PromptTemplate(
    "You are an expert in AI frameworks (LangChain and LlamaIndex). "
    "Your task is to rephrase a question into a more general form. Examples:\n"
    "{few_shot_examples}\n\n"
    "Current question: {question}\n\n"
    "Step-back version:"
)

# Define the original input question.
question = "Can LangChain be used for building document-based retrieval pipelines?"

# Use the LLM to predict the step-back version of the question.
step_back_response = llm.predict(
    step_back_prompt,
    question=question,
    few_shot_examples=few_shot_examples
)

step_back_question = step_back_response.strip()
print(f"Step-back question: {step_back_question}")

# Retrieve context for both the original and step-back questions.
normal_retrievals = retriever.retrieve(question)
step_back_retrievals = retriever.retrieve(step_back_question)

# Combine the retrieved document texts from each retrieval stage.
normal_context = "\n\n".join([doc.text for doc in normal_retrievals])
step_back_context = "\n\n".join([doc.text for doc in step_back_retrievals])

# Define a QA prompt template that combines both contexts and asks for an answer.
qa_prompt = PromptTemplate(
    "Combine context from both retrieval stages:\n"
    "=== Original Context ===\n"
    "{normal_context}\n\n"
    "=== Step-back Context ===\n"
    "{step_back_context}\n\n"
    "Answer the original question: {question}"
)


# Use the LLM to generate the final answer.
final_answer = llm.predict(
    qa_prompt,
    normal_context=normal_context,
    step_back_context=step_back_context,
    question="Can LangChain be used for building document-based retrieval pipelines?"
)
print(f"\nFinal Answer:\n{final_answer}")