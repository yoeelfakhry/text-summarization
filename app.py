import gradio as gr
from inference import summarize_with_bart, summarize_with_pegasus


def summarize_both(article_text):
    """
    Run the input article through both BART (fine-tuned) and Pegasus (zero-shot),
    returning both summaries side by side for direct comparison.
    """
    bart_summary = summarize_with_bart(article_text)
    pegasus_summary = summarize_with_pegasus(article_text)
    return bart_summary, pegasus_summary


demo = gr.Interface(
    fn=summarize_both,
    inputs=gr.Textbox(
        lines=10,
        label="Article Text",
        placeholder="Paste a news article here to summarize...",
    ),
    outputs=[
        gr.Textbox(label="BART Summary (fine-tuned on CNN/DailyMail, 20k examples)"),
        gr.Textbox(label="Pegasus Summary (zero-shot, pretrained for summarization)"),
    ],
    title="News Article Summarizer — BART vs. Pegasus",
    description=(
        "Compares a BART-base model I fine-tuned on CNN/DailyMail against "
        "Google's Pegasus model (zero-shot, pretrained specifically for summarization). "
        "See the project README for the full training and evaluation methodology."
    ),
)

if __name__ == "__main__":
    demo.launch()