from datasets import load_dataset
from transformers import Seq2SeqTrainer , Seq2SeqTrainingArguments
from src.utils.config_loader import load_config
from src.utils.data_preprocessing import get_tokenizer , preprocess_data , get_data_collator
from src.models.bart_model import load_bart_model   
# from src.eval.rouge_eval import compute_metrics_fn
import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token)


config = load_config("configs/bart_test.yaml")

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


import evaluate
import numpy as np


def compute_metrics_fn(tokenizer):
    """
    Create a function to compute the ROUGE score for the predicted summaries against the reference summaries.
    Args:
        tokenizer: The tokenizer object for the specified model 
    Returns:
        function: A function that takes in a tuple of predicted summaries and reference summaries and returns a dictionary containing the ROUGE scores for the predicted summaries against the reference summaries
    """

    rouge_score  = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        """
        Compute the ROUGE score for the predicted summaries against the reference summaries.
        
        Args:
        eval_pred (tuple): A tuple containing the predicted summaries and the reference summaries
        Returns:
        dict: A dictionary containing the ROUGE scores for the predicted summaries against the reference summaries 
        """
        print(">>> compute_metrics WAS CALLED <<<")
        predictions,labels = eval_pred 

        print("DEBUG predictions type:", type(predictions))
        if isinstance(predictions, tuple):
            print("DEBUG predictions is tuple, length:", len(predictions))
            predictions = predictions[0]

        predictions = np.where(predictions != -100 , predictions , tokenizer.pad_token_id)
        labels = np.where(labels != -100 , labels , tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        return { k: v*100 for k , v in result.items()}  