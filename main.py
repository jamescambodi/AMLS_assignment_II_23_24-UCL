import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

NAME_CONVERTER = {"old": "base", "new": "fine-tuned"}

#import functions from folder B
from B import (
    infer,
    combine_prediction_cols,
    acc,
    row_wise_f1_score,
    generate_submission,
)

# read training & testing dataset
train_df = pd.read_csv("Datasets/train.csv")
test_df = pd.read_csv("Datasets/test.csv")

# concat train + test dataframe -> create source matching dataframe
src_df = pd.concat([train_df.drop("label_group", axis=1), test_df], axis=0)

# create target column
tmp = train_df.groupby("label_group").posting_id.agg("unique").to_dict()
train_df["target"] = train_df.label_group.map(tmp)

# create prediction column use `image_phash` column
tmp = train_df.groupby("image_phash").posting_id.agg("unique").to_dict()
train_df["preds3"] = train_df.image_phash.map(tmp)

# load the embedding vector for Testing data
test_image_embeds = np.load(file="Datasets/cached/test/all_image_embeds.npy")
test_text_embeds = np.load(file="Datasets/cached/test/all_text_embeds.npy")

# evaluate pre-trained model vs fine-tuned model on training dataset
for model_type in ["old", "new"]:
    df: pd.DataFrame = infer(train_df=train_df, model_type=model_type)
    df["preds"] = df.apply(combine_prediction_cols, axis=1)
    f1, precision, recall = row_wise_f1_score(y_true=df["target"], y_pred=df["preds"])

    print(
        f"{NAME_CONVERTER[model_type].upper()} Model - F1: {f1.mean():.2f} - Precision: {precision.mean():.2f} - Recall: {recall.mean():.2f}"
    )

# generate submission file
generate_submission(
    test_df=test_df,
    model_type="old",
    test_image_embeds=test_image_embeds,
    test_text_embeds=test_text_embeds,
    submission_file_path="result.csv",
    src_df=src_df,
)
