from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq

def get_tokenizer(model_name: str):
    """
    Load the tokenizer for the specified model 

    Args:
    model name(str): name of the checkpoint model to load the tokenizer 
    Returns:    
    tokenizer: The tokenizer object for the specified model
    """ 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def preprocess_data(model_checkpoint: str , tokenizer, examples: dict , max_input_length: int , max_output_length: int) -> dict:
   """
   Make tokenization of the input and the output text for the model
   
   Args:
    model_checkpoint (str): The model checkpoint name to load the tokenizer 
    tokenizer: the tokenizer object for the specified model
    examples (dict): A dataset example containing the input and output text to be tokenized
    max_input_length (int): The maximum length of the input sequence after tokenization
    max_output_length (int): The maximum length of the output sequence after tokenization
    
   Returns:
    dict: A dictionary containing tokenized input_ids, attention_mask, and labels for the model
   """
   inputs = examples["article"]
   targets = examples["highlights"]

   if "t5" in model_checkpoint:
        inputs = ["summarize: " + inp for inp in inputs]

   model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
    )

   labels = tokenizer(
        text_target=targets,
        max_length=max_output_length,
        truncation=True,
    )

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

def get_data_collator(model , tokenizer):
    """
    Create a data collator for the specified model and tokenizer which make dynamic padding of the input and output sequences to the maximum length in the batch.

    Args:
        model: The model object for which the data collator is to be created
        tokenizer: The tokenizer object for the specified model
    Returns:
        DataCollatorForSeq2Seq: A data collator object for the specified model and tokenizer
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
