import torch 
from datasets import Dataset, load_dataset
import numpy as np
import random as random
from torch.utils.data import DataLoader

def create_tokenized_dataset(texts, tokenizer, max_length):
    """
    Convert a list of strings into a tokenized datasets.Dataset.

    Args:
    - texts (list of str): The list of strings to be tokenized.
    - tokenizer: A tokenizer from the HuggingFace Transformers library.
    - max_length (int): Maximum length to which the tokenized strings should be truncated/padded.

    Returns:
    - A datasets.Dataset with tokenized inputs and attention masks.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a datasets.Dataset from the list of texts
    dset = Dataset.from_dict({'text': texts})

    # Define a function to tokenize the texts
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    # Tokenize the texts
    tokenized_dset = dset.map(tokenize_function, batched=True)
    tokenized_dset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return tokenized_dset

def generate_rm_inputs(cfg, tokenizer, seed=1):
    dset = load_dataset(cfg.dataset_name, split="train")
    string_list = dset['chosen'] + dset['rejected']
    tokenized_sentence_dataset = create_tokenized_dataset(string_list, tokenizer, cfg.max_length)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    token_loader = DataLoader(
        tokenized_sentence_dataset, 
        batch_size=cfg.model_batch_size, 
        shuffle=True,
        )
    return token_loader