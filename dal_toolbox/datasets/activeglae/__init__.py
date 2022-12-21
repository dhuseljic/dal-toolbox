from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_ds(args):
    print('>> Loading dataset')
    ds = load_dataset(args.dataset.name_hf)
    print('>> Starting tokenization')
    tokenizer = AutoTokenizer.from_pretrained(args.model.name_hf, use_fast=False)
    return ds, tokenizer
