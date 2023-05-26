from transformers import AutoTokenizer
from .base import AbstractGLAE

class Wikitalk(AbstractGLAE):
    def __init__(self, model_name, dataset_path, val_split=0.1, seed=None, pre_batch_size=1000, pre_num_proc=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super().__init__('wikitalk', dataset_path, val_split, seed, pre_batch_size, pre_num_proc)

    @property
    def num_classes(self):
        return 2

    def process_fn(self, batch):
        batch = self.tokenizer(batch['common_text'], truncation=True)
        return batch

    def rename_column(self, ds):
        ds = ds.rename_column('toxic', 'label')
        return ds
