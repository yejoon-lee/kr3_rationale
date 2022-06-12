# tokenize and save the extracted rationales
# KR4: Korean Restaurant Reviews Rationalized with Ratings
# Uses multiprocessing

# $ python fresh_tokenize.py [name] [save_path] [--unrationale]
# example: $ python fresh_tokenize.py FT3_TopK FT3_TopK
# example: $ python fresh_tokenize.py FT3_TopK FT3_TopK-Un --unrationale

import argparse
from glob import glob
from multiprocessing import Pool
from datasets import load_from_disk
from transformers import BertTokenizer

# 0. argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", help="path to load rationales(KR4)")
parser.add_argument("save_path", help='path to save the tokenized rationales (or unrationales')
parser.add_argument("--unrationale", help='use the flag to tokenize unrationale instead of rationale', action='store_true') # default false
args = parser.parse_args()

# 1. tokenizer from hgf
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def tokenize_func(example, use_unrationale=False):
    if use_unrationale:
        return tokenizer(example['Unrationale'], truncation=True, padding=True)
    return tokenizer(example['Rationale'], truncation=True, padding=True)

# 2. function to preprocess (including tokenize) on single batch
def preprocess_batch(batch_path):
    # load batch
    batch = load_from_disk(batch_path)

    # tokenize
    batch = batch.map(tokenize_func, batched=True, fn_kwargs={'use_unrationale':args.unrationale})

    # process columns
    batch = batch.remove_columns(['Unrationale', 'Rationale'])
    batch = batch.rename_column('Rating', 'labels')
    batch.set_format(type='torch')

    # save tokenized rationale
    batch_index = batch_path[batch_path.find('batch'): ]
    batch.save_to_disk(f'kr4_tokenized/{args.save_path}/{batch_index}')

# 3. execute preprocess_batch() with multiprocessing
batch_path_list  = glob(f'kr4/{args.name}/batch_*')

with Pool(127) as p:
    p.map(preprocess_batch, batch_path_list)