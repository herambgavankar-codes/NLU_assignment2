"""
metrics.py
-----------------------------------------
Evaluation metrics for generated names.

Metrics:
1. Novelty Rate:
   Percentage of generated names that do not
   appear in the training set.

2. Diversity:
   Number of unique generated names divided by
   total generated names.
-----------------------------------------
"""


def compute_novelty_rate(generated_names, training_names):
    """
    Compute novelty rate.

    Args:
        generated_names: list of generated names
        training_names: list/set of training names

    Returns:
        float: novelty rate in percentage
    """
    if len(generated_names) == 0:
        return 0.0

    training_set = set(training_names)
    novel_count = sum(1 for name in generated_names if name not in training_set)

    return (novel_count / len(generated_names)) * 100.0


def compute_diversity(generated_names):
    """
    Compute diversity score.

    Args:
        generated_names: list of generated names

    Returns:
        float: diversity score
    """
    if len(generated_names) == 0:
        return 0.0

    unique_count = len(set(generated_names))
    return unique_count / len(generated_names)