from transformers import AutoTokenizer
from .base import AbstractGLAE
from datasets import load_dataset


class QNLI(AbstractGLAE):
    def __init__(self, model_name, dataset_path, val_split=0.1, seed=None, pre_batch_size=1000, pre_num_proc=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super().__init__('qnli', dataset_path, val_split, seed, pre_batch_size, pre_num_proc)
    
    def load_dataset(self, ds_name, cache_dir):
        ds = load_dataset('glue', ds_name, cache_dir=cache_dir)
        return ds

    def get_test_dataset(self, ds):
        return ds['validation']
   
    @property
    def num_classes(self):
        return 2

    def process_fn(self, batch):
        batch = self.tokenizer(batch['question'], batch['sentence'],truncation=True)
        return batch
    













