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