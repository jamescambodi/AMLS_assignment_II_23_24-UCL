import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from torch.utils.data import DataLoader
from Datasets.data_util import ImageDataset, TextDataset, img_collate_fn
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
)

#Generate the embedding for each CLIP model
def generate_embedding_file(
    checkpoint_path: str,
    train_df: pd.DataFrame,
    image_folder: str,
    image_embed_save_path: str,
    text_embed_save_path: str,
):
    model = CLIPModel.from_pretrained(checkpoint_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    device = "mps" #Use mps due to arm chip
    print(f"We're currently using {device}...")

    model.to(device)
    model.eval()

    img_paths = (f"{image_folder}/" + train_df["image"]).values.tolist()
    title_pools = train_df["title"].str.lower().values.tolist()

    txt_dataset = TextDataset(
        texts=title_pools,
    )

    img_dataset = ImageDataset(
        image_paths=img_paths,
    )

    batch_size = 32

    # run the image embedding generate process
    img_dataloader = DataLoader(
        dataset=img_dataset, batch_size=batch_size, collate_fn=img_collate_fn
    )

    all_image_embeds = None

    for batch in tqdm(img_dataloader):
        inputs = processor(
            images=batch,
            return_tensors="pt",
        )

        inputs = inputs.to(device)
        vision_outputs = model.vision_model(**inputs)
        image_embeds = vision_outputs[1]
        image_embeds: torch.Tensor = model.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        if all_image_embeds is None:
            all_image_embeds = image_embeds.cpu().detach().numpy()
        else:
            all_image_embeds = np.concatenate(
                [all_image_embeds, image_embeds.cpu().detach().numpy()], axis=0
            )

    # run the text embedding generate process
    txt_dataloader = DataLoader(dataset=txt_dataset, batch_size=batch_size)

    all_text_embeds = None
    for batch in tqdm(txt_dataloader):
        inputs = processor(
            text=batch,
            return_tensors="pt",
            max_length=77,
            truncation=True,
            padding=True,
        )

        inputs = inputs.to(device)
        text_outputs = model.text_model(**inputs)
        text_embeds = text_outputs[1]
        text_embeds: torch.Tensor = model.text_projection(text_embeds)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        if all_text_embeds is None:
            all_text_embeds = text_embeds.cpu().detach().numpy()
        else:
            all_text_embeds = np.concatenate(
                [all_text_embeds, text_embeds.cpu().detach().numpy()], axis=0
            )

    # save all the embedding of image and text into folder
    np.save(file=image_embed_save_path, arr=all_image_embeds)
    np.save(file=text_embed_save_path, arr=all_text_embeds)
    print("Embedding Generation Process run success!!!")
