from transformers import AutoTokenizer
from .base import AbstractGLAE

class DBPedia(AbstractGLAE):
    def __init__(self, model_name, dataset_path, val_split=0.1, seed=None, pre_batch_size=1000, pre_num_proc=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        super().__init__('dbpedia_14', dataset_path, val_split, seed, pre_batch_size, pre_num_proc)

    @property
    def num_classes(self):
        return 14

    def process_fn(self, batch):
        batch = self.tokenizer(batch['content'], truncation=True)
        return batch