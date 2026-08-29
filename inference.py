# inference.py

from src.models.bart_model import load_bart_model
from src.utils.data_preprocessing import get_tokenizer
from src.models.pegasus import load_pegasus_model, get_pegasus_tokenizer

# Load the two model one the app start 
BART_PATH = "yoeel/bart-cnn-summarizer"  # name of the model in Hugging Face Hub
bart_tokenizer = get_tokenizer(BART_PATH)
bart_model = load_bart_model(BART_PATH)

PEGASUS_CHECKPOINT = "google/pegasus-cnn_dailymail"
pegasus_tokenizer = get_pegasus_tokenizer(PEGASUS_CHECKPOINT)
pegasus_model = load_pegasus_model(PEGASUS_CHECKPOINT)


def summarize_with_bart(text: str, max_length: int = 128) -> str:
    if not text or not text.strip():
        return "Please enter valid text."
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(bart_model.device)
    summary_ids = bart_model.generate(**inputs, max_length=max_length, num_beams=4, no_repeat_ngram_size=3)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_with_pegasus(text: str, max_length: int = 128) -> str:
    if not text or not text.strip():
        return "Please enter valid text."
    inputs = pegasus_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(pegasus_model.device)
    summary_ids = pegasus_model.generate(**inputs, max_length=max_length, num_beams=4)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)