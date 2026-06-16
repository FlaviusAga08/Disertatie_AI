from datasets import load_dataset, DatasetDict
import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    GenerationConfig,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
import json
from typing import Dict, List
import random


# 1. ENVIRONMENT & SETUP
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B for cleaner training

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 2. LEGAL-SPECIFIC OPTIMIZATIONS
class LegalTokenizerPreprocessor:
    """Custom preprocessor for legal documents"""
    
    def __init__(self, tokenizer, max_input=512, max_target=200):
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
        
    def preprocess_legal(self, examples):
        """
        Legal-specific preprocessing:
        - Add domain prefix for better legal understanding
        - Extract key sections if possible
        - Preserve legal terminology
        """
        # Add domain-specific prefix for better performance
        inputs = [
            f"summarize legal document: {doc[:self.max_input * 4]}" 
            for doc in examples["text"]
        ]
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.max_input, 
            truncation=True,
            padding=False,  # Use dynamic padding with collator
        )
        
        labels = self.tokenizer(
            text_target=examples["summary"], 
            max_length=self.max_target, 
            truncation=True,
            padding=False,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# 3. ADVANCED QUANTIZATION & PEFT (LoRA)
def create_quantized_model_with_lora(model_name: str):
    """
    Create 4-bit quantized model with LoRA for efficient fine-tuning
    - 4-bit quantization saves ~75% memory
    - LoRA adds only ~0.1% trainable parameters
    - Achieves comparable performance to full fine-tuning
    """
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Double quantization
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Load quantized model
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-base", 
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA config - efficient fine-tuning
    lora_config = LoraConfig(
        r=12,  
        lora_alpha=24,
        target_modules=["q", "v", "k"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# 4. ADVANCED GENERATION CONFIG
def create_optimized_generation_config():
    """
    Optimized generation for legal summaries
    - Balanced beam search for quality/speed
    - Domain-aware decoding
    """
    return GenerationConfig(
        max_length=200,
        min_length=30,  # Legal summaries need minimum detail
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,  # Prevent repetition
        repetition_penalty=1.2,
        # Note: temperature and top_p are for sampling (do_sample=True)
        # We use deterministic beam search for legal summarization
    )


# 5. MAIN TRAINING PIPELINE
def run_training(
    output_dir: str = "./t5_legal_results",
    model_name: str = "t5-base",
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 3e-4,  # Slightly higher for LoRA
):
    """
    Complete optimized training pipeline
    """
    
    print("=" * 80)
    print("OPTIMIZED T5 LEGAL SUMMARIZER - DISSERTATION EDITION")
    print("=" * 80)
    
    # 1. Load and prepare data
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset("billsum")
    
    # 2. Tokenizer and preprocessor
    print("[2/5] Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocessor = LegalTokenizerPreprocessor(tokenizer, max_input=512, max_target=200)
    
    # Tokenize datasets
    print("    Preprocessing legal documents...")
    tokenized_datasets = dataset.map(
        preprocessor.preprocess_legal,
        batched=True,
        num_proc=1,  # Windows-safe: use 1 process
        remove_columns=dataset["train"].column_names,
    )
    
    # 3. Create quantized model with LoRA
    print("[3/5] Creating quantized model with LoRA...")
    model = create_quantized_model_with_lora(model_name)
    
    generation_config = create_optimized_generation_config()
    
    # 4. Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    
    # 5. Advanced training arguments
    print("[4/5] Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # === OPTIMIZATION ===
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  
        
        # === LEARNING RATE SCHEDULING ===
        learning_rate=learning_rate,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        
        # === EVALUATION & SAVING ===
        eval_strategy="steps",
        eval_steps=400,  # More frequent evaluation
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,  # Keep only 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # === GPU STABILITY (Windows/RTX 5070) ===
        fp16=False,  
        bf16=True,  
        tf32=False,
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # Save memory
        
        # === WINDOWS SPECIFIC ===
        dataloader_num_workers=0,  # Critical for Windows
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        
        # === GENERATION & PREDICTION ===
        predict_with_generate=True,
        generation_config=generation_config,
        
        # === LOGGING ===
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        
        # === MISC ===
        weight_decay=0.01,
        seed=42,
    )
    
    # 6. Trainer with early stopping
    print("[5/5] Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.0001,
            )
        ],
    )
    
    # 7. Execute training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    trainer.train()
    
    # 8. Save final model
    print("\nSaving final model...")
    model.save_pretrained(f"{output_dir}_final")  # ← Changed
    tokenizer.save_pretrained(f"{output_dir}_final")  # ← Added
    generation_config.save_pretrained(f"{output_dir}_final")
    
    print("\n" + "=" * 80)
    print(f"✓ Training complete! Model saved to: {output_dir}_final")
    print("=" * 80)



# 6. INFERENCE & EVALUATION
def load_optimized_model(model_path: str):
    """Load the trained model for inference"""
    from peft import AutoPeftModelForSeq2SeqLM
    
    # Load LoRA model
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    
    return model, tokenizer, generation_config


def summarize_document(
    document: str,
    model,
    tokenizer,
    generation_config,
    max_length: int = 512
):

    input_text = f"summarize legal document: {document[:max_length]}"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    summary_ids = model.generate(
        inputs,
        generation_config=generation_config,
        max_length=200,
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



# 7. ENTRY POINT
if __name__ == "__main__":
    # Run training
    run_training(
        output_dir="./t5_legal_results",
        model_name="t5-base",
        num_epochs=4,
        batch_size=6,
        learning_rate=8e-4,
    )
