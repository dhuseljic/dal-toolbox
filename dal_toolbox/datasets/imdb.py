from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset 

def build_imdb(split, ds, return_info=False):

    if split == 'train':
        ds = ds['train']

    elif split == 'query':
        ds = ds['test']
    
    elif split == 'test':
        ds = ds['test']

    if return_info == True:
        ds_info = {'n_classes': 2}
        return ds, ds_info

    return ds
