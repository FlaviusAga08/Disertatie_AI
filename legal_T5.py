from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


dataset = load_dataset("billsum")


model_name = "t5-small"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
max_input_length = 512
max_target_length = 150

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)


model = T5ForConditionalGeneration.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="./t5_legal_results",
    #evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    #predict_with_generate=True,
    logging_steps=100,
    fp16=True, 
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["validation"],
)


trainer.train()


trainer.evaluate(tokenized_datasets["test"])

#Aici se salveaza
model.save_pretrained("./t5_legal_results")
tokenizer.save_pretrained("./t5_legal_results")