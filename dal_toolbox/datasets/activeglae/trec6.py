from transformers import AutoTokenizer
from .base import AbstractGLAE
from datasets import load_dataset


class TREC6(AbstractGLAE):
    def __init__(self, model_name, dataset_path, val_split=0.1, seed=None, pre_batch_size=1000, pre_num_proc=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super().__init__('trec', dataset_path, val_split, seed, pre_batch_size, pre_num_proc)
    
    def load_dataset(self, ds_name, cache_dir):
        ds = load_dataset(ds_name, cache_dir=cache_dir)
        return ds
      
    @property
    def num_classes(self):
        return 6

    def process_fn(self, batch):
        batch = self.tokenizer(batch['text'], truncation=True)
        return batch

    def rename_column(self, ds):
        ds = ds.rename_column('coarse_label', 'label')
        return ds






