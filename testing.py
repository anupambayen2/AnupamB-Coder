
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset
ds = load_dataset('OLMo-Coding/starcoder-python-instruct', split='train[:3]', streaming=False)
print('Columns:', ds.column_names)
print()
for i, item in enumerate(ds):
    print(f'--- Sample {i} ---')
    for k, v in item.items():
        print(f'  {k}: {str(v)[:150]}')
    print()