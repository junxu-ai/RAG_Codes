# !pip install graphrag

# assume the data files, e.g., txt, are stored in data folder
graphrag init --root ./data

# assume that you already configured LLM, e.g., openai
# build the index
graphrag index --root ./data

# query using --method global or local.
graphrag query \
--root ./ragtest \
--method global \
--query "What are the top themes in this story?"
