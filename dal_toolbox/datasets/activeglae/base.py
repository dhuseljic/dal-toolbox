import torch.distributed as dist
from datasets import load_dataset

class AbstractGLAE():
    def __init__(self, dataset_name, dataset_path, val_split=0.1, seed=None, pre_batch_size=1000, pre_num_proc=4):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.val_split = val_split
        self.seed = seed
        self.pre_batch_size = pre_batch_size
        self.pre_num_proc = pre_num_proc

        print('> Loading data set..')
        self._download_datasets()
        ds = self.load_dataset(self.dataset_name, cache_dir=self.dataset_path)

        print('> Apply Tokenization on the complete dataset..')
        ds = self.rename_column(ds)
        ds = self._preprocess(ds)
        
        if 0 < val_split:
            self.split = ds['train'].train_test_split(val_split, shuffle=True, seed=self.seed)
            self.train_dataset = self.split['train']
            self.val_dataset = self.split['test']
            self.query_dataset = self.split['train']
        else:
            self.train_dataset = ds['train']
            self.val_dataset = None
            self.query_dataset = ds['train']

        self.test_dataset = self.get_test_dataset(ds)

    def process_fn(self): 
        raise ValueError("This method should be overwritten.")

    def rename_column(self, ds):
        return ds

    def get_test_dataset(self, ds):
        return ds['test']
    
    def load_dataset(self, ds_name, cache_dir):
        ds = load_dataset(ds_name, cache_dir=cache_dir)
        return ds
        
    def _download_datasets(self):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                self.download_datasets()
            dist.barrier()  # Make sure that only the process with rank 0 downloads the data
        else:
            self.download_datasets()

    def download_datasets(self):
        self.load_dataset(self.dataset_name, cache_dir=self.dataset_path)
    
    def _preprocess(self, ds):
        ds = ds.map(
            self.process_fn, batched=True, batch_size=self.pre_batch_size, num_proc=self.pre_num_proc)
        ds = ds.remove_columns(
            list(set(ds['train'].column_names)-set(['input_ids', 'attention_mask', 'label']))
        )         
        ds = ds.with_format("torch")
        return ds