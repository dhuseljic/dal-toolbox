from ..base import AbstractData
from datasets import load_dataset
from transformers import AutoTokenizer


class CIFAR10(AbstractData):

    def __init__(
            self,
            dataset_path: str,
            mean: tuple = (0.4914, 0.4822, 0.4465),
            std: tuple = (0.247, 0.243, 0.262),
            val_split: float = 0.1,
            seed: int = None
    ) -> None:
        self.mean = mean
        self.std = std
        super().__init__(dataset_path, val_split, seed)






# class AGNews(AbstractData):
#     def __init__(self, dataset_path, val_split, seed):
#         super().__init__(dataset_path, val_split, seed)
	

#     def tokenizer(self, modelname):
#         tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
#         return tokenizer

#     @property
#     def num_classes(self):
#         return 4

#     @property
#     def full_train_dataset(self):
#         return load_dataset("agnews", split="train")

#     def test_dataset(self):
#         return load_dataset("agnews", split="test")

#     def download_datasets(self):
#         load_dataset("agnews").map(lambda batch: self.tokenizer(batch['text']), truncation=True)






        

    








        

    








