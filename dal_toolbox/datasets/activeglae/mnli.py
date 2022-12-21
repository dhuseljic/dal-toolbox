def build_mnli(split, ds, return_info=False):

    if split == 'train':
        ds = ds['train']

    elif split == 'query':
        ds = ds['train']
    
    elif split == 'test':
        ds = ds['test']

    if return_info == True:
        ds_info = {'n_classes': 3}
        return ds, ds_info

    return ds