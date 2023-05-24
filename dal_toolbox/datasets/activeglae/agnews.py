from datasets import load_dataset
from transformers import AutoTokenizer
from ..base import AbstractData

class AGNews(AbstractData):
    def __init__(self, modelname, dataset_path="ag_news", val_split=0, seed=None):
        super().__init__(dataset_path, val_split, seed)
        self.modelname = modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname, use_fast=False)
    
    
    @property
    def num_classes(self):
        return 4
    
    def download_datasets(self):
        load_dataset(self.dataset_path)
            
    # def tokenize(self):
    #     # self.dataset = load_dataset(self.dataset_path)
    #     # self.tokenizer = AutoTokenizer.from_pretrained(self.modelname, use_fast=False)
    #     # return 
    #     pass

    def process(self, batch):
        batch = self.tokenizer(batch['text'], truncation=True)
        return batch


    @property
    def full_train_dataset(self):
        train_ds = load_dataset(self.dataset_path, split="train")
        train_ds = train_ds.map(
            self.process, batched=True, batch_size=1000, num_proc=4
        )
        train_ds = train_ds.remove_columns(
            list(set(train_ds['train'].column_names)-set(['label', 'input_ids', 'attention_mask']))
        )         

        return train_ds
    
    @property
    def test_dataset(self):
        test_ds = load_dataset(self.dataset_path, split="train")
        test_ds = test_ds.map(
            self.process, batched=True, batch_size=1000, num_proc=4
        )
        test_ds = test_ds.remove_columns(
            list(set(test_ds['train'].column_names)-set(['label', 'input_ids', 'attention_mask']))
        )         

        return test_ds
    
    @property
    def train_transforms(self):
        pass

    @property
    def eval_transforms(self):
        pass

    @property
    def query_transforms(self):
        pass

    @property
    def full_train_dataset_eval_transforms(self):
        return super().full_train_dataset_eval_transforms
    
    @property
    def full_train_dataset_query_transforms(self):
        return super().full_train_dataset_query_transforms
    
    @property
    def full_transform(self):
        return super().full_transform



# from multiprocess import set_start_method

# import torch

# import os
# >>>

# set_start_method("spawn")
# >>>

# def gpu_computation(example, rank):

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

#     # Your big GPU call goes here

#     return examples
# >>>

# updated_dataset = dataset.map(gpu_computation, with_rank=True)