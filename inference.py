from src.models.bart_model import load_bart_model
from src.utils.data_preprocessing import get_tokenizer
from src.models.pegasus import load_pegasus_model, get_pegasus_tokenizer

# Load the two model one the app start 
BART_PATH = "yoeel/bart-cnn-summarizer-20k-3ep"  
bart_tokenizer = get_tokenizer(BART_PATH)
bart_model = load_bart_model(BART_PATH)


def summarize_with_bart(text: str, max_length: int = 128, num_beams: int = 4) -> str:
    if not text or not text.strip():
        return "Please enter valid text."
    inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(bart_model.device)
    summary_ids = bart_model.generate(**inputs, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=3)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
