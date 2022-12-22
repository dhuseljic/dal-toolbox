from . import tokenize_ds

def build_sst2(args):
    ds, tokenizer = tokenize_ds(args)
    ds = ds.map(lambda batch: tokenizer(batch['sentence'], truncation=True))
    ds = ds.remove_columns(
            list(set(ds['train'].column_names)-set(['label', 'input_ids', 'attention_mask']))
    )
    ds_info = {'n_classes': 2, 'tokenizer': tokenizer}
    return ds, ds_info