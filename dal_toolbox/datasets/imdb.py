from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset 

def build_imdb(split, ds, args, return_info=False):

    ds = load_dataset('imdb')
    checkpoint = args.model.name
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def tokenize_function(batch):
        tokenizer(batch["text"], truncation=True)

    tokenized_ds = ds.map(tokenize_function, batched=True)

    if split == 'train':
        ds = tokenized_ds['train']

    elif split == 'query':
        ds = tokenized_ds['test']
    
    elif split == 'test':
        ds = tokenized_ds['test']

    return ds
