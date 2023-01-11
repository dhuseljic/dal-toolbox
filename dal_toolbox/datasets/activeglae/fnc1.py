from . import tokenize_ds

def build_fnc1(args):
    ds, tokenizer = tokenize_ds(args)
    ds = ds.map(lambda batch: tokenizer(batch['Headline'], batch['articleBody'], truncation=True))
    ds = ds.rename_column('Stance', 'label')
    ds = ds.remove_columns(
            list(set(ds['train'].column_names)-set(['label', 'input_ids', 'attention_mask']))
    )
    ds_info = {'n_classes': 4, 'tokenizer': tokenizer}
    return ds, ds_info
    