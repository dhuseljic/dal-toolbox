from datasets import load_dataset
from transformers import AutoTokenizer
#!TODO: maybe split glue and not glue dataset tokenize function
#!TODO: schönere aufteilung mit manual datasets (toxicity)
def tokenize_ds(args):
    print('>> Loading dataset')
    try:
        if args.dataset.name == 'wikitalk':
            ds = load_dataset('jigsaw_toxicity_pred', data_dir= '~/.cache/huggingface/manual_ds/jigsaw_toxicity_pred')
        else:
            ds = load_dataset(args.dataset.name_hf)

    except FileNotFoundError:
        ds = load_dataset('glue', args.dataset.name_hf)
    print('>> Starting tokenization')
    tokenizer = AutoTokenizer.from_pretrained(args.model.name_hf, use_fast=False)
    return ds, tokenizer
