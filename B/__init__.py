# Import various utility functions from the 'utils' module. These functions include:
# - infer: Function to run inference using specific model types.
# - get_preds: Function to extract predictions based on provided scores and cutoffs.
# - combine_prediction_cols: Combines prediction columns from different sources into one, removing duplicates.
# - acc: Calculates the accuracy of predictions.
# - generate_submission: Likely a function to prepare submission data formatted for specific requirements.
# - row_wise_f1_score: Calculates the F1 score, precision, and recall row-wise for given true and predicted labels.

from .utils import (
    infer,
    get_preds,
    combine_prediction_cols,
    acc,
    generate_submission,
    row_wise_f1_score,
)
from .gen_embed import generate_embedding_file


def img_collate_fn(batch):
    return list(batch)
