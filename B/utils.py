import numpy as np
import pandas as pd
from tqdm import tqdm

# Combines three prediction lists into one, removing any duplicate predictions.
def combine_prediction_cols(row):
    x = list(row.preds1) + list(row.preds2) + list(row.preds3)  # + list(row.preds5)
    return list(set(x)) # Returns unique elements only

# Calculates the accuracy of the predictions by considering the intersection of the predicted and target sets.
def acc(row):
    n = len(np.intersect1d(row.target, row.preds))
    return 2 * n / (len(row.target) + len(row.preds))

# Computes the F1 score, precision, and recall for each row in the dataset.
def row_wise_f1_score(y_true, y_pred):
    tp = np.array([len(np.intersect1d(x[0], x[1])) for x in zip(y_true, y_pred)])
    fp = y_pred.apply(lambda x: len(x)).values - tp # False positives: incorrectly predicted elements
    fn = y_true.apply(lambda x: len(x)).values - tp # False negatives: elements missed in the predictions

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1, precision, recall


def get_preds(
    df: pd.DataFrame,
    total_scores: np.ndarray,
    cutoff: float,
    pred_col: str,
    src_df: pd.DataFrame = None,
):
    """Generate prediction values

    Args:
        df (pd.DataFrame): input dataframe
        total_scores (np.ndarray): scores you want to use for prediction process
        cutoff (float): cutoff value for score
        pred_col (str): name of the prediction column

    Returns:
        pd.DataFrame: output dataframe with prediction column
    """
    if src_df is None:
        src_df = df

    predictions = []

    total_scores = total_scores >= cutoff
    for idx in tqdm(range(df.shape[0]), leave=False):
        all_scores = total_scores[idx, :]
        preds: pd.Series = src_df.loc[all_scores, "posting_id"]

        predictions.append(list(set(preds.values.tolist())))

    df[pred_col] = pd.Series(predictions, index=df.index)
    return df


def infer(
    train_df: pd.DataFrame,
    model_type: str,
) -> pd.DataFrame:
    assert model_type in [
        "old",
        "new",
    ], "Model Type have to be `old` (original) or `new` (fine-tuned)!!!"

    # original model's embedding
    all_image_embeds = np.load(
        file=f"Datasets/cached/{model_type}/all_image_embeds.npy"
    )
    all_text_embeds = np.load(file=f"Datasets/cached/{model_type}/all_text_embeds.npy")

    # calculate scores between image and image/ text and text
    ivi_scores = all_image_embeds @ all_image_embeds.T
    tvt_scores = all_text_embeds @ all_text_embeds.T

    # Sets cutoff values for similarity scores to filter predictions. The cutoffs differ depending on whether the
    # model is the original or a fine-tuned version.
    if model_type == "new":
        ivi_cutoff = 0.96 # Cutoff for new model image-to-image scores
        tvt_cutoff = 0.9675 # Cutoff for new model text-to-text scores
    else:
        ivi_cutoff = 0.95 # Cutoff for old model image-to-image scores
        tvt_cutoff = 0.92 # Cutoff for old model text-to-text scores

    # Filters and predicts based on the image-to-image similarity scores, adding the predictions as a new column to the dataframe.
    df = get_preds(
        df=train_df,
        total_scores=ivi_scores,
        cutoff=ivi_cutoff,
        pred_col="preds1",
    )

    # Filters and predicts based on the text-to-text similarity scores, updating the predictions in another column of the dataframe.
    df = get_preds(
        df=df,
        total_scores=tvt_scores,
        cutoff=tvt_cutoff,
        pred_col="preds2",
    )

    return df


def generate_submission(
    test_df: pd.DataFrame,
    model_type: str,
    test_image_embeds: np.ndarray,
    test_text_embeds: np.ndarray,
    submission_file_path: str,
    src_df: pd.DataFrame = None,
) -> pd.DataFrame:
    assert model_type in [
        "old",
        "new",
    ], "Model Type have to be `old` (original) or `new` (fine-tuned)!!!"

    # original model's embedding
    all_image_embeds = np.load(
        file=f"Datasets/cached/{model_type}/all_image_embeds.npy"
    )
    all_text_embeds = np.load(file=f"Datasets/cached/{model_type}/all_text_embeds.npy")

    # concat embedding
    all_image_embeds = np.concatenate([all_image_embeds, test_image_embeds], axis=0)
    all_text_embeds = np.concatenate([all_text_embeds, test_text_embeds], axis=0)

    # calculate scores between image and image/ text and text
    ivi_scores = test_image_embeds @ all_image_embeds.T
    tvt_scores = test_text_embeds @ all_text_embeds.T

    # cutoff threshold for new and old model
    if model_type == "new":
        ivi_cutoff = 0.96
        tvt_cutoff = 0.9675
    else:
        ivi_cutoff = 0.95
        tvt_cutoff = 0.92

    # predict use image vs image score
    df = get_preds(
        df=test_df,
        total_scores=ivi_scores,
        cutoff=ivi_cutoff,
        pred_col="preds1",
        src_df=src_df,
    )

    # predict use text vs text score
    df = get_preds(
        df=df,
        total_scores=tvt_scores,
        cutoff=tvt_cutoff,
        pred_col="preds2",
        src_df=src_df,
    )

    # create prediction column use `image_phash` column
    tmp = test_df.groupby("image_phash").posting_id.agg("unique").to_dict()
    df["preds3"] = df.image_phash.map(tmp)

    test_df["matches"] = df.apply(combine_prediction_cols, axis=1)
    test_df["matches"] = test_df["matches"].str.join(" ")

    test_df = test_df.drop(["preds1", "preds2", "preds3"], axis=1)

    # write down the result to file csv name "result.csv"
    test_df.to_csv(submission_file_path, index=False)
    print(
        f"Sucessfully generate the submission file store at {submission_file_path}..."
    )
