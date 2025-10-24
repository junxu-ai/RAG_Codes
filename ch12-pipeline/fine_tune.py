# to be verified

import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments  
from datasets import Dataset  

# Sample synthetic data  
data = {  
    "question": [  
        "What are the first-line treatments for Parkinson's disease?",  
        "How does metformin affect renal function?"  
    ],  
    "oracle_docs": [  
        "Parkinson's disease is managed with levodopa, dopamine agonists, and MAO-B inhibitors...",  
        "Metformin is contraindicated in severe renal impairment (eGFR <30 mL/min)...",  
    ],  
    "distractor_docs": [  
        "Vitamin C supplements show no benefit in Parkinson's...",  
        "Aspirin is safe for most patients with CKD..."  
    ],  
    "answer": [  
        "First-line treatments include levodopa, dopamine agonists, and MAO-B inhibitors.",  
        "Metformin is contraindicated if eGFR <30 mL/min due to lactic acidosis risk."  
    ]  
}  

dataset = Dataset.from_dict(data)  
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")  
tokenizer.pad_token = tokenizer.eos_token  

"""
 Model Configuration with LoRA
"""
from peft import LoraConfig, get_peft_model  

model = AutoModelForCausalLM.from_pretrained(  
    "meta-llama/Llama-2-7b-chat-hf",  
    load_in_8bit=True,  # Quantization for memory efficiency  
    device_map="auto"  
)  

# Apply LoRA  
lora_config = LoraConfig(  
    r=8,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,  
    bias="none"  
)  
model = get_peft_model(model, lora_config)  
model.print_trainable_parameters()  # Output: ~4M trainable parameters  



"""
training loop
"""

def format_prompt(example):  
    # Combine question, oracle, and distractors  
    context = f"Question: {example['question']}\nDocuments:\n- {example['oracle_docs']}\n- {example['distractor_docs']}"  
    prompt = f"{context}\nAnswer: {example['answer']}"  
    return {"text": prompt}  

dataset = dataset.map(format_prompt)  

training_args = TrainingArguments(  
    output_dir="./results",  
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,  
    num_train_epochs=3,  
    learning_rate=2e-5,  
    fp16=True,  
    logging_steps=10,  
)  

from trl import SFTTrainer  

trainer = SFTTrainer(  
    model=model,  
    args=training_args,  
    train_dataset=dataset,  
    dataset_text_field="text",  
    max_seq_length=512,  
)  

trainer.train()  

"""
 Inference and Evaluation
 """

def generate_answer(question, oracle_doc, distractor_doc):  
    context = f"Question: {question}\nDocuments:\n- {oracle_doc}\n- {distractor_doc}"  
    inputs = tokenizer(context, return_tensors="pt").to("cuda")  
    outputs = model.generate(**inputs, max_length=512)  
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  

# Test  
question = "What is the treatment for hypertensive crisis?"  
oracle_doc = "Hypertensive crisis requires IV labetalol or nitroprusside..."  
distractor_doc = "Omega-3 supplements may mildly reduce blood pressure."  

answer = generate_answer(question, oracle_doc, distractor_doc)  
print(answer)  # Output: "Hypertensive crisis is managed with IV labetalol or nitroprusside..."  

"""
 Evaluation""" 
