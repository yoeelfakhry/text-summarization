from transformers import AutoModelForSeq2SeqLM
import torch


def load_bart_model(model_checkpoint: str):
    """
    Load the Bart model from the specified checkpoint

    Args:
    model_checkpoint (str): The model checkpoint name to load the Bart model

    Returns:
    model: The Bart model object loaded from the specified checkpoint
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model.to(device)

    return model