import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def compute_metrics(p, id2tag):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    precision = results["overall_precision"]
    recall = results["overall_recall"]
    beta = 0.2
    beta_squared = beta**2
    
    if (beta_squared * precision + recall) == 0:
        f0_2_score = 0.0
    else:
        f0_2_score = (1 + beta_squared) * (precision * recall) / ((beta_squared * precision) + recall)

    flat_results = {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": results["overall_f1"],
        "overall_f0.2": f0_2_score, # Add the new score to the results
        "overall_accuracy": results["overall_accuracy"],
    }
    
    for key, value in results.items():
        if isinstance(value, dict) and 'f1' in value:
            flat_results[f"{key}_precision"] = value['precision']
            flat_results[f"{key}_recall"] = value['recall']
            flat_results[f"{key}_f1"] = value['f1']

    return flat_results
