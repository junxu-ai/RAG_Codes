# to be verified

import os
from llama_index.finetuning import (
    SentenceTransformersFinetuneEngine,
    EmbeddingQAFinetuneDataset,
)

# ----------------------------
# Create a synthetic training dataset
# ----------------------------
train_corpus = {
    "doc1": (
        "Artificial intelligence is the simulation of human intelligence "
        "processes by machines, especially computer systems."
    ),
    "doc2": (
        "Machine learning is a subset of artificial intelligence that focuses "
        "on building systems that learn from data."
    ),
}

train_queries = {
    "q1": "What is artificial intelligence?",
    "q2": "Explain machine learning.",
}

train_relevant_docs = {
    "q1": ["doc1"],
    "q2": ["doc2"],
}

train_dataset = EmbeddingQAFinetuneDataset(
    corpus=train_corpus,
    queries=train_queries,
    relevant_docs=train_relevant_docs,
)

# ----------------------------
# Create a synthetic validation dataset
# ----------------------------
val_corpus = {
    "doc3": (
        "Deep learning is a branch of machine learning that uses neural networks "
        "with many layers."
    ),
    "doc4": (
        "Natural language processing enables computers to understand and interpret "
        "human language."
    ),
}

val_queries = {
    "q3": "What is deep learning?",
    "q4": "Define natural language processing.",
}

val_relevant_docs = {
    "q3": ["doc3"],
    "q4": ["doc4"],
}

val_dataset = EmbeddingQAFinetuneDataset(
    corpus=val_corpus,
    queries=val_queries,
    relevant_docs=val_relevant_docs,
)

# ----------------------------
# Set up the fine-tuning engine
# ----------------------------
# Here, we use the open-source "BAAI/bge-small-en" model.
# The finetuning engine is configured with the training dataset, an output path,
# and the optional validation dataset.
# For demonstration, we run only 1 epoch.
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
    epochs=1,
)

# ----------------------------
# Run the fine-tuning process
# ----------------------------
print("Starting fine-tuning...")
finetune_engine.finetune()
print("Fine-tuning completed.")

# ----------------------------
# Retrieve the finetuned model
# ----------------------------
embed_model = finetune_engine.get_finetuned_model()

# ----------------------------
# Display the finetuned model details
# ----------------------------
print("Finetuned Model:")
print(embed_model)
