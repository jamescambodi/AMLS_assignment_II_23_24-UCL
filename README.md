# AMLS_assignment_II_23_24-UCL
# Shopee Product Matching

## Project's Structure
- Folder A will have 2 files:
    - `fine_tune_clip.py`: this file is provided by Huggingface for fine-tuning CLIP model on custom dataset.
    - `main_train.ipynb`: this is the notebook contain all steps for running the fine-tuning process.

- Folder B will have 2 files:
    - `__init__.py`: this file is required to make Python treat the folder B as the module.
    - `gen_embed.py`: this file contain function for running the Embedding Generation for both Image and Text.
    - `utils.py`: this file contain function for making prediction include:
        - getting the prediction base on Image vs Image score; Text vs Text score, etc.
        - F1, precision, recall scores calculation function.
        - Submission generation function.

- Folder Datasets will contain:
    - Folder `cached`: This folder store output of model prediction process (Image Embedding vector of all training dataset, Text Embedding, hyperparameters, etc.)
    - Folder `data_util` will have:
        - `__init__.py` as mentioned before.
        - `image_data.py`: Image Dataset class customized by using Pytorch's Datset class.
        - `text_data.py`: Text Dataset class customized by using Pytorch's Dataset class. 
    - Folder `test_images`: contain image files for testing dataset.
    - `train.csv`: training dataframe.
    - `test.csv`: testing dataframe.
- `main.py` this is the python script include all the necessary steps to run the Matching Process end-to-end.

## Packages Required

To run the code, the following packages are required:
    - `transformers` - Huggingface Package has been used in this project to load the pre-trained model from the huggingface hub.
    - `pandas` - for reading dataframe.
    - `tqdm` - for showing the progress bar.
    - `numpy` - for handling the vector/matrix operations (add, substract, divide, etc).
    - `gc` - is garbage collector $\to$ is usded for collect the allocated memory that have not been used anymore.
    - `torch.utils.data.DataLoader` - use for loading dataset by batch size.
    - `requests` - for downloading file from the specific URL.

These are libraries required before running the code. 

## How to run
### 1. Download data
- You have to download the kaggle's competition data to that folder `Datasets`.

### 2. Inference
```python
$ python main.py
```