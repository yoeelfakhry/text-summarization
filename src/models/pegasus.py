from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def load_pegasus_model(model_checkpoint: str):
    """
    Load the Pegasus model from the specified checkpoint (zero-shot, no fine-tuning required).

    Args:
        model_checkpoint (str): The model checkpoint name to load the Pegasus model
            (e.g. "google/pegasus-cnn_dailymail")

    Returns:
        model: The Pegasus model object loaded from the specified checkpoint
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.to(device)

    return model


def get_pegasus_tokenizer(model_checkpoint: str):
    """
    Load the tokenizer for the specified Pegasus checkpoint.

    Args:
        model_checkpoint (str): The model checkpoint name to load the tokenizer

    Returns:
        tokenizer: The tokenizer object for the specified Pegasus checkpoint
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)