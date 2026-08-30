from datasets import load_dataset
from transformers import Seq2SeqTrainer , Seq2SeqTrainingArguments
from src.utils.config_loader import load_config
from src.utils.data_preprocessing import get_tokenizer , preprocess_data , get_data_collator
from src.models.bart_model import load_bart_model   
# from src.eval.rouge_eval import compute_metrics_fn
import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from rouge_score import rouge_scorer
import numpy as np 

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token)


config = load_config("configs/bart.yaml")

# Load the dataset 
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

# Load the tokenizer
tokenizer = get_tokenizer(config["model_checkpoint"])

# Split the dataset into train and validation sets
train_subset = dataset["train"].shuffle(seed=42).select(range(config["train_size"]))
validation_subset = dataset["validation"].shuffle(seed=42).select(range(config["validation_size"]))

# Apply tokenization to the train dataset and the validation dataset
tokenize_fn = lambda examples: preprocess_data(
    config["model_checkpoint"], tokenizer, examples,
    config["max_input_length"], config["max_output_length"]
)

tokenized_train = train_subset.map(
    tokenize_fn, batched=True, remove_columns=train_subset.column_names
)

tokenized_val = validation_subset.map(
    tokenize_fn, batched=True, remove_columns=validation_subset.column_names
)

# load the model and create the data collator

model = load_bart_model(config["model_checkpoint"]) 
data_collator = get_data_collator(model, tokenizer)

# Define the compute_metrics function for evaluation
# compute_metrics = compute_metrics_fn(tokenizer)


# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    per_device_eval_batch_size=config["per_device_eval_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    warmup_steps=config["warmup_steps"],
    weight_decay=config["weight_decay"],
    learning_rate=config["learning_rate"],
    logging_steps=config["logging_steps"],
    eval_strategy=config["eval_strategy"],
    eval_steps=config["eval_steps"],
    save_steps=config["save_steps"],
    save_total_limit=config["save_total_limit"],
    generation_max_length=config["generation_max_length"],
    predict_with_generate=config["predict_with_generate"],
    fp16= config["fp16"],
    report_to = config["report_to"] ,
    # metric_for_best_model= config["metric_for_best_model"],
    # greater_is_better= config["greater_is_better"] ,
    # load_best_model_at_end= config["load_best_model_at_end"],
    hub_model_id=config["hub_model_id"],
    push_to_hub=True,
    eval_accumulation_steps=config["eval_accumulation_steps"],

)

print("=== DEBUG ===")
print("predict_with_generate:", training_args.predict_with_generate, type(training_args.predict_with_generate))
print("metric_for_best_model:", training_args.metric_for_best_model)
print("=============")

# Build trainer 
trainer = Seq2SeqTrainer(
    model= model ,
    args = training_args , 
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    processing_class=tokenizer,

)

# Train the model
train_result = trainer.train()

# Save the final model and tokenizer
final_model_dir = f"{config['output_dir']}/final_model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"Training complete. Final model saved to: {final_model_dir}")
print(train_result)



test_sample = dataset["test"].select(range(100))
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for example in test_sample:
    inputs = tokenizer(example["article"], return_tensors="pt", truncation=True, max_length=config["max_input_length"]).to(model.device)
    summary_ids = model.generate(**inputs, max_length=config["max_output_length"], num_beams=4, no_repeat_ngram_size=3)
    generated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    reference = example["highlights"]

    scores = scorer.score(reference, generated)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

print(f"Final ROUGE-1 avg: {np.mean(rouge1_scores):.4f}")
print(f"Final ROUGE-2 avg: {np.mean(rouge2_scores):.4f}")
print(f"Final ROUGE-L avg: {np.mean(rougeL_scores):.4f}")