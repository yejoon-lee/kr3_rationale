# tokenize and save the extracted rationales
# KR4: Korean Restaurant Reviews Rationalized with Ratings
# Uses multiprocessing

# $ python fresh_tokenize.py [name]
# example: $ python fresh_tokenize.py FT3

import argparse
from glob import glob
from multiprocessing import Pool
from datasets import load_from_disk
from transformers import BertTokenizer

# 0. argparse
parser = argparse.ArgumentParser()
parser.add_argument("name", help="path to load rationales(KR4)")
args = parser.parse_args()

# 1. tokenizer from hgf
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def tokenize_func(example):
    return tokenizer(example['Rationale'], truncation=True, padding=True)

# 2. function to preprocess (including tokenize) on single batch
def preprocess_batch(batch_path):
    # load batch
    batch = load_from_disk(batch_path)

    # tokenize
    batch = batch.map(tokenize_func, batched=True)

    # process columns
    batch = batch.remove_columns(['Unrationale', 'Rationale'])
    batch = batch.rename_column('Rating', 'labels')
    batch.set_format(type='torch')

    # save tokenized rationale
    batch_index = batch_path[batch_path.find('_') + 1 :]
    batch.save_to_disk(f'kr4_tokenized/{args.name}/batch_{batch_index}')

# 3. execute preprocess_batch() with multiprocessing
batch_path_list  = glob(f'kr4/{args.name}/batch_*')

with Pool(127) as p:
    p.map(preprocess_batch, batch_path_list)