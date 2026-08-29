from bert_score import score


def compute_bertscore(predictions: list, references: list, lang: str = "en") -> dict:
    """
    Compute BERTScore (Precision, Recall, F1) between generated summaries and reference summaries.

    Args:
        predictions (list): List of generated summary strings
        references (list): List of reference summary strings
        lang (str): Language code for the BERTScore model (default "en")

    Returns:
        dict: A dictionary containing the average Precision, Recall, and F1 BERTScore values
    """
    P, R, F1 = score(predictions, references, lang=lang, verbose=True)

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }