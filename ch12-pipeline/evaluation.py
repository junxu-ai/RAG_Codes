# to be verified

import numpy as np
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_dataset
import spacy
from sentence_transformers import SentenceTransformer, util

# Load evaluation utilities
nlp = spacy.load("en_core_web_sm")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def _generate_answer(self, question, oracle_doc, distractor_doc):
        context = f"Question: {question}\nDocuments:\n- {oracle_doc}\n- {distractor_doc}"
        inputs = self.tokenizer(context, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _calculate_similarity(self, text1, text2):
        emb1 = similarity_model.encode(text1, convert_to_tensor=True)
        emb2 = similarity_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()

    def _extract_entities(self, text):
        doc = nlp(text)
        return {ent.text.lower() for ent in doc.ents}

    def evaluate_sample(self, example):
        # Generate answer
        generated_answer = self._generate_answer(
            example["question"],
            example["oracle_docs"],
            example["distractor_docs"]
        )
        
        # Split answer from context
        answer = generated_answer.split("Answer:")[-1].strip()
        
        # Calculate BLEU score
        bleu_score = corpus_bleu(
            [[ref.split()] for ref in [example["answer"]]],
            answer.split()
        )
        
        # Calculate retrieval accuracy
        oracle_sim = self._calculate_similarity(answer, example["oracle_docs"])
        distractor_sim = self._calculate_similarity(answer, example["distractor_docs"])
        retrieval_acc = 1 if oracle_sim > distractor_sim else 0
        
        # Calculate hallucination
        context = f"{example['oracle_docs']} {example['distractor_docs']}"
        context_entities = self._extract_entities(context)
        answer_entities = self._extract_entities(answer)
        hallucinated_entities = answer_entities - context_entities
        hallucination = 1 if len(hallucinated_entities) > 0 else 0
        
        return {
            "bleu": bleu_score,
            "retrieval_acc": retrieval_acc,
            "hallucination": hallucination,
            "answer": answer,
            "generated": generated_answer
        }

    def evaluate_dataset(self, dataset):
        results = {
            "bleu_scores": [],
            "retrieval_accs": [],
            "hallucinations": []
        }
        
        for example in dataset:
            metrics = self.evaluate_sample(example)
            results["bleu_scores"].append(metrics["bleu"])
            results["retrieval_accs"].append(metrics["retrieval_acc"])
            results["hallucinations"].append(metrics["hallucination"])
        
        return {
            "avg_bleu": np.mean(results["bleu_scores"]),
            "retrieval_accuracy": np.mean(results["retrieval_accs"]),
            "hallucination_rate": np.mean(results["hallucinations"]),
            "num_samples": len(dataset)
        }

# Load test dataset
test_data = load_dataset("your_dataset_repo/medical_qa_eval", split="test")

# Initialize evaluator
evaluator = RAGEvaluator(model, tokenizer)

# Run evaluation
metrics = evaluator.evaluate_dataset(test_data)

# Print results
print(f"""
Evaluation Results:
--------------------------------
- Avg BLEU Score: {metrics['avg_bleu']:.2f}
- Retrieval Accuracy: {metrics['retrieval_accuracy']:.2%}
- Hallucination Rate: {metrics['hallucination_rate']:.2%}
- Total Samples: {metrics['num_samples']}
""")

# Example individual sample analysis
sample_result = evaluator.evaluate_sample(test_data[0])
print(f"\nSample Analysis:\nQuestion: {test_data[0]['question']}")
print(f"Generated Answer: {sample_result['answer']}")
print(f"Reference Answer: {test_data[0]['answer']}")
print(f"BLEU: {sample_result['bleu']:.2f}, Retrieval Correct: {bool(sample_result['retrieval_acc'])}, Hallucination: {bool(sample_result['hallucination'])}")