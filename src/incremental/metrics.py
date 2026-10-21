"""
Standard class-incremental learning metrics:

- Accuracy matrix R, where R[i][j] = accuracy on task j's test set after
  having trained through task i (only defined for j <= i).
- Average forgetting: for each task j, how much accuracy dropped between
  its peak (right after learning it) and the final model. Averaged over
  all tasks except the last.

These match the standard definitions used in class-incremental learning
papers (e.g. iCaRL, EWC follow-ups) so results here are comparable to
literature you'd cite in a thesis-adjacent writeup.
"""

import numpy as np


def compute_forgetting(accuracy_matrix: np.ndarray) -> float:
    """
    Args:
        accuracy_matrix: (T, T) array, accuracy_matrix[i, j] = accuracy on
            task j after training through task i. Entries where j > i
            should be np.nan (undefined -- task j not learned yet).

    Returns:
        Average forgetting across all tasks except the last.
    """
    T = accuracy_matrix.shape[0]
    forgettings = []
    for j in range(T - 1):  # last task has no "future" step to forget by
        seen = accuracy_matrix[j:, j]
        seen = seen[~np.isnan(seen)]
        if len(seen) < 2:
            continue
        peak = seen[0]  # accuracy right after learning task j
        final = seen[-1]  # accuracy after learning all subsequent tasks
        forgettings.append(peak - final)
    return float(np.mean(forgettings)) if forgettings else 0.0


def average_incremental_accuracy(accuracy_matrix: np.ndarray) -> float:
    """Mean of the final row (accuracy on all tasks after the last one)."""
    final_row = accuracy_matrix[-1]
    final_row = final_row[~np.isnan(final_row)]
    return float(np.mean(final_row))
